"""
Shared evaluation runner — used by both run_eval.py and the pipeline orchestrator.
Loads the model once with NF4 quantization (fits in 32GB), runs t2v + i2v inference.
"""

import json
import logging
import re
import sys
from pathlib import Path

import torch
import yaml

log = logging.getLogger(__name__)


def find_latest_checkpoint(workspace: Path) -> tuple[Path | None, int, Path | None]:
    """Find the latest LoRA checkpoint. Returns (file, step, batch_dir)."""
    best_step = 0
    best_file = None
    best_batch = None
    for ckpt in workspace.glob("batch-*/checkpoints/lora_weights_step_*.safetensors"):
        match = re.search(r"(\d+)", ckpt.stem)
        if match:
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_file = ckpt
                best_batch = ckpt.parent.parent
    return best_file, best_step, best_batch


def find_i2v_refs(workspace: Path, batch_dir: Path | None = None) -> list[dict]:
    """Find i2v refs — check given batch first, then search backwards."""
    search_dirs = [batch_dir] if batch_dir else []
    search_dirs.extend(sorted(workspace.glob("batch-*"), reverse=True))

    for bdir in search_dirs:
        meta = bdir / "i2v" / "metadata.jsonl"
        if not meta.exists():
            continue
        refs = []
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ref = json.loads(line)
                if Path(ref["image"]).exists():
                    refs.append(ref)
        if refs:
            log.info("Found %d i2v refs from %s", len(refs), bdir.name)
            return refs
    return []


@torch.inference_mode()
def run_eval(
    config_path: str = "config.yaml",
    checkpoint: Path | None = None,
    step: int | None = None,
    batch_dir: Path | None = None,
    do_t2v: bool = True,
    do_i2v: bool = True,
):
    """
    Run t2v and/or i2v evaluation using NF4-quantized model.
    Loads model once, generates all samples, then unloads.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    workspace = Path("workspace")
    eval_cfg = cfg.get("evaluation", {})

    # Find checkpoint
    if checkpoint is None:
        checkpoint, step, batch_dir = find_latest_checkpoint(workspace)
    if checkpoint is None:
        log.error("No checkpoints found")
        return

    log.info("Eval at step %d using %s", step, checkpoint.name)

    # Output dir
    samples_dir = batch_dir / "samples" if batch_dir else workspace / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Load model components (same as training — NF4 quantized)
    sys.path.insert(0, str(Path(cfg["ltx_trainer_dir"]) / "src"))
    sys.path.insert(0, str(Path(cfg["ltx_trainer_dir"]).parent.parent / "packages" / "ltx-core" / "src"))

    from ltx_trainer.model_loader import (
        load_model, load_embeddings_processor,
    )
    from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler

    model_path = cfg["training"]["model_checkpoint"]
    text_encoder_path = cfg["training"]["text_encoder"]
    video_vae_path = cfg["training"].get("video_vae_checkpoint")
    audio_vae_path = cfg["training"].get("audio_vae_checkpoint")
    text_projection_path = cfg["training"].get("text_projection_checkpoint")

    import gc

    # Load all components (single mmap avoids repeated file opens)
    # This loads transformer + VAE + text encoder + embeddings processor in one pass
    log.info("Loading all components...")
    components = load_model(
        checkpoint_path=model_path,
        text_encoder_path=text_encoder_path,
        video_vae_path=video_vae_path,
        audio_vae_path=audio_vae_path,
        text_projection_path=text_projection_path,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=do_i2v,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=True,
    )

    transformer = components.transformer.to(dtype=torch.bfloat16)
    vae_decoder = components.video_vae_decoder
    text_encoder = components.text_encoder

    # Apply LoRA via PEFT then quantize
    log.info("Applying LoRA: %s", checkpoint.name)
    import re as _re
    from safetensors.torch import load_file
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    state_dict = load_file(str(checkpoint))
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
    target_modules = sorted({m.group(1) for k in state_dict
                            for m in [_re.match(r"(.+)\.lora_[AB]\.", k)] if m})
    lora_rank = next(v.shape[0] for k, v in state_dict.items() if "lora_A" in k and v.ndim == 2)
    log.info("LoRA rank=%d, %d target modules", lora_rank, len(target_modules))

    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank,
                             target_modules=target_modules, lora_dropout=0.0)
    transformer = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(transformer.get_base_model(), state_dict)
    del state_dict
    gc.collect()

    # Block swap: keep transformer on CPU, move blocks to GPU one at a time
    # This lets us run inference with only ~2GB VRAM for the transformer
    import types

    base_model = transformer
    if hasattr(transformer, 'get_base_model'):
        base_model = transformer.get_base_model()

    if hasattr(base_model, 'transformer_blocks'):
        num_blocks = len(base_model.transformer_blocks)
        log.info("Setting up block swap for %d transformer blocks", num_blocks)

        # Keep all blocks on CPU
        for block in base_model.transformer_blocks:
            block.to("cpu")

        # Monkey-patch _process_transformer_blocks to offload blocks
        original_process = base_model._process_transformer_blocks

        def _block_swap_process(self, video, audio, perturbations):
            device = torch.device("cuda")
            for i, block in enumerate(self.transformer_blocks):
                block.to(device, non_blocking=True)
                torch.cuda.synchronize()
                video, audio = block(video=video, audio=audio, perturbations=perturbations)
                block.to("cpu", non_blocking=True)
            return video, audio

        base_model._process_transformer_blocks = types.MethodType(_block_swap_process, base_model)

        # Also apply chunked feedforward
        def _ffn_chunked_forward(self, x):
            num_chunks = 4
            if x.shape[1] > 4096:
                chunk_size = x.shape[1] // num_chunks
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < num_chunks - 1 else x.shape[1]
                    x[:, start:end] = self.net(x[:, start:end])
                return x
            return self.net(x)

        for block in base_model.transformer_blocks:
            block.ff.forward = types.MethodType(_ffn_chunked_forward, block.ff)

        # Prevent ValidationSampler from moving entire transformer to GPU
        # (block swap handles device management per-block)
        _original_to = transformer.to
        def _noop_to(*args, **kwargs):
            # Only allow dtype changes, block device moves
            if args and isinstance(args[0], torch.dtype):
                return _original_to(*args, **kwargs)
            if 'dtype' in kwargs and 'device' not in kwargs:
                return _original_to(**kwargs)
            return transformer
        transformer.to = _noop_to

        # Move non-block parts to GPU (patchify, proj_out, adaln, etc.)
        for name, module in base_model.named_children():
            if name != 'transformer_blocks':
                module.to("cuda")

        log.info("Block swap + chunked feedforward enabled")

    gc.collect()
    log.info("VRAM after setup: %.1f GB", torch.cuda.memory_allocated() / 1024**3)

    # VAE encoder loaded lazily for i2v (below)

    # Get eval params — block swap keeps VRAM low, can use full resolution
    width = eval_cfg.get("width", 768)
    height = eval_cfg.get("height", 448)
    num_frames = eval_cfg.get("num_frames", 89)
    guidance_scale = eval_cfg.get("guidance_scale", 4.0)
    num_steps = eval_cfg.get("num_inference_steps", 30)

    # Create sampler
    from ltx_trainer.progress import StandaloneSamplingProgress
    with StandaloneSamplingProgress(num_steps=num_steps) as progress:
        sampler = ValidationSampler(
            transformer=transformer,
            vae_decoder=vae_decoder,
            vae_encoder=components.video_vae_encoder,
            text_encoder=text_encoder,
            embeddings_processor=text_encoder.embeddings_processor,
            sampling_context=progress,
        )

        # T2V eval
        if do_t2v:
            prompts = eval_cfg.get("prompts", [])
            for i, prompt in enumerate(prompts):
                out_path = samples_dir / f"step_{step:06d}_t2v_{i}.mp4"
                log.info("T2V %d: %.80s...", i, prompt)
                try:
                    video, audio = sampler.generate(
                        config=GenerationConfig(
                            prompt=prompt,
                            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                            height=height, width=width, num_frames=num_frames,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale,
                            seed=42 + i,
                            generate_audio=False,
                        ),
                        device="cuda",
                    )
                    from ltx_trainer.video_utils import save_video
                    save_video(video, str(out_path), fps=eval_cfg.get("fps", 25))
                    log.info("  Saved: %s (%.1f KB)", out_path.name, out_path.stat().st_size / 1024)
                except Exception as e:
                    log.error("  T2V %d failed: %s", i, e)

        # I2V eval
        if do_i2v:
            i2v_refs = find_i2v_refs(workspace, batch_dir)
            if not i2v_refs:
                log.warning("No i2v refs found, skipping i2v eval")
            else:
                from torchvision import transforms
                from ltx_trainer.utils import open_image_as_srgb

                for i, ref in enumerate(i2v_refs[:2]):
                    out_path = samples_dir / f"step_{step:06d}_i2v_{i}.mp4"
                    log.info("I2V %d: %s", i, Path(ref["image"]).name)
                    try:
                        image = open_image_as_srgb(ref["image"])
                        # Resize to match target resolution
                        image = image.resize((width, height))
                        condition_image = transforms.ToTensor()(image)

                        video, audio = sampler.generate(
                            config=GenerationConfig(
                                prompt=ref["prompt"],
                                negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                                height=height, width=width, num_frames=num_frames,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance_scale,
                                seed=42 + i,
                                condition_image=condition_image,
                                generate_audio=False,
                            ),
                            device="cuda",
                        )
                        from ltx_trainer.video_utils import save_video
                        save_video(video, str(out_path), fps=eval_cfg.get("fps", 25))
                        log.info("  Saved: %s (%.1f KB)", out_path.name, out_path.stat().st_size / 1024)
                    except Exception as e:
                        log.error("  I2V %d failed: %s", i, e)

    log.info("Eval complete — results in %s", samples_dir)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int)
    parser.add_argument("--i2v-only", action="store_true")
    parser.add_argument("--t2v-only", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    workspace = Path("workspace")
    ckpt, st, bdir = find_latest_checkpoint(workspace)
    if args.step:
        for c in workspace.glob(f"batch-*/checkpoints/lora_weights_step_{args.step:05d}.safetensors"):
            ckpt, st, bdir = c, args.step, c.parent.parent

    run_eval(
        config_path=args.config,
        checkpoint=ckpt, step=st, batch_dir=bdir,
        do_t2v=not args.i2v_only,
        do_i2v=not args.t2v_only,
    )

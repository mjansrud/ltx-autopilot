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
    from ltx_trainer.quantization import quantize_model
    from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler

    model_path = cfg["training"]["model_checkpoint"]
    text_encoder_path = cfg["training"]["text_encoder"]

    log.info("Loading model (NF4 quantized)...")
    components = load_model(
        checkpoint_path=model_path,
        text_encoder_path=text_encoder_path,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=do_i2v,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=True,
    )

    # Quantize transformer with NF4 (same as training)
    log.info("Quantizing transformer with NF4...")
    transformer = components.transformer.to(dtype=torch.bfloat16)
    transformer = quantize_model(transformer, precision="nf4-bnb")

    # Apply LoRA
    log.info("Loading LoRA: %s", checkpoint.name)
    from ltx_trainer.model_loader import load_lora_weights
    transformer = load_lora_weights(transformer, str(checkpoint))

    # Get eval params
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
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            text_encoder=components.text_encoder,
            embeddings_processor=components.text_encoder.embeddings_processor,
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

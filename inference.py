import argparse
import math

import torch
import torchaudio

from ai_osu_maps.config import Config
from ai_osu_maps.config import FlowConfig
from ai_osu_maps.config import ModelConfig
from ai_osu_maps.inference.postprocessor import vectors_to_osu
from ai_osu_maps.inference.sampler import sample
from ai_osu_maps.model.audio_encoder import AudioEncoder
from ai_osu_maps.model.flow_transformer import FlowTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an osu! beatmap from audio")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output_path", type=str, default="output.osu", help="Output .osu file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--difficulty", type=float, default=5.0, help="Target star rating")
    parser.add_argument("--cs", type=float, default=4.0, help="Circle Size")
    parser.add_argument("--ar", type=float, default=4.0, help="Approach Rate")
    parser.add_argument("--od", type=float, default=4.0, help="Overall Difficulty")
    parser.add_argument("--hp", type=float, default=4.0, help="HP Drain Rate")
    parser.add_argument("--mapper_id", type=int, default=None, help="Mapper ID for style conditioning")
    parser.add_argument("--num_steps", type=int, default=32, help="Number of sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_objects", type=int, default=None, help="Number of hit objects (auto if not set)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[FlowTransformer, AudioEncoder, ModelConfig]:
    """Load model weights from a checkpoint file.

    Expects a checkpoint dict with keys: model_state_dict, ema_state_dict,
    audio_encoder_state_dict, config.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config: Config = checkpoint["config"]
    model_config = config.model

    model = FlowTransformer(model_config)
    # Prefer EMA weights if available
    state_key = "ema_state_dict" if "ema_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    model.to(device)

    audio_encoder = AudioEncoder(d_model=model_config.d_model)
    if "audio_encoder_state_dict" in checkpoint:
        audio_encoder.load_state_dict(checkpoint["audio_encoder_state_dict"])
    audio_encoder.to(device)

    return model, audio_encoder, model_config


def get_audio_duration_ms(audio_path: str) -> float:
    """Get audio duration in milliseconds."""
    info = torchaudio.info(audio_path)
    return (info.num_frames / info.sample_rate) * 1000.0


def build_conditioning(
    args: argparse.Namespace, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build the conditioning dict from CLI arguments."""
    mapper_id = args.mapper_id if args.mapper_id is not None else 0
    return {
        "difficulty": torch.tensor([args.difficulty], device=device),
        "cs": torch.tensor([args.cs], device=device),
        "ar": torch.tensor([args.ar], device=device),
        "od": torch.tensor([args.od], device=device),
        "hp": torch.tensor([args.hp], device=device),
        "mapper_id": torch.tensor([mapper_id], dtype=torch.long, device=device),
        "year": torch.tensor([2024.0], device=device),
    }


def predict_num_objects(
    model: FlowTransformer, audio_features: torch.Tensor
) -> int:
    """Predict the number of hit objects from audio features."""
    with torch.no_grad():
        log_count = model.object_count_predictor(audio_features)
    return max(1, round(math.exp(log_count.item())))


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load models
    print(f"Loading checkpoint: {args.checkpoint}")
    model, audio_encoder, model_config = load_checkpoint(args.checkpoint, device)
    model.eval()
    audio_encoder.eval()

    # Load and encode audio
    print(f"Processing audio: {args.audio_path}")
    waveform = AudioEncoder.load_audio(args.audio_path)
    waveform = waveform.to(device)

    with torch.no_grad():
        audio_features = audio_encoder(waveform)  # (1, T_audio, d_model)

    duration_ms = get_audio_duration_ms(args.audio_path)

    # Determine number of objects
    if args.num_objects is not None:
        num_objects = args.num_objects
    else:
        num_objects = predict_num_objects(model, audio_features)
    print(f"Generating {num_objects} hit objects")

    # Build conditioning and flow config
    cond = build_conditioning(args, device)
    flow_config = FlowConfig(
        n_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
    )

    # Sample
    print(f"Sampling with {flow_config.n_steps} steps, CFG scale {flow_config.cfg_scale}")
    vectors = sample(
        model=model,
        audio_features=audio_features,
        num_objects=num_objects,
        cond=cond,
        config=flow_config,
        device=device,
    )

    # Convert to .osu file
    vectors_np = vectors.squeeze(0).cpu().numpy()
    osu_content = vectors_to_osu(
        vectors_np,
        audio_path=args.audio_path,
        duration_ms=duration_ms,
        difficulty=args.difficulty,
        cs=args.cs,
        ar=args.ar,
        od=args.od,
        hp=args.hp,
    )

    # Write output
    with open(args.output_path, "w") as f:
        f.write(osu_content)

    print(f"Wrote {args.output_path}")
    print(f"  Objects: {num_objects}")
    print(f"  Duration: {duration_ms / 1000:.1f}s")
    print(f"  Difficulty: {args.difficulty}*")


if __name__ == "__main__":
    main()

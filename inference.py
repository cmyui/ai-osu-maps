import argparse
import logging
import os

import torch

from ai_osu_maps.config import ModelConfig, GenerationConfig
from ai_osu_maps.data.osu_parser import tokens_to_events
from ai_osu_maps.data.tokenizer import Tokenizer
from ai_osu_maps.inference.postprocessor import BeatmapConfig, Postprocessor
from ai_osu_maps.inference.sampler import sample_autoregressively
from ai_osu_maps.model.transformer import Transformer
from ai_osu_maps.model.audio_encoder import AudioEncoder

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate osu! beatmaps from audio using autoregressive model"
    )
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for style")
    parser.add_argument("--difficulty", type=float, default=5.0, help="Star rating")
    parser.add_argument("--cs", type=float, default=4.0, help="Circle size")
    parser.add_argument("--ar", type=float, default=9.0, help="Approach rate")
    parser.add_argument("--od", type=float, default=8.0, help="Overall difficulty")
    parser.add_argument("--hp", type=float, default=5.0, help="HP drain")
    parser.add_argument("--bpm", type=float, default=120.0, help="BPM for timing")
    parser.add_argument("--offset", type=int, default=0, help="Timing offset in ms")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--osz", action="store_true", help="Export as .osz")
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stderr")
    parser.add_argument("--mapper", type=int, default=0, help="Mapper ID (0 = unknown)")
    parser.add_argument("--year", type=float, default=0.0, help="Year condition (0.0 = unknown)")
    parser.add_argument("--no_ema", action="store_true", help="Use trained weights instead of EMA")
    parser.add_argument(
        "--audio_encoder", type=str, default=None,
        help="Path to audio_encoder.pt (overrides checkpoint-embedded state)",
    )
    return parser.parse_args()


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(
    checkpoint_path: str, device: torch.device, use_ema: bool = True,
) -> tuple[Transformer, ModelConfig, dict | None]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config: ModelConfig = ckpt["model_config"]

    tokenizer = Tokenizer()
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        dropout=0.0,  # no dropout at inference
        max_seq_len=model_config.max_seq_len,
        mert_dim=model_config.mert_dim,
        text_dim=model_config.text_dim,
        n_text_tokens=model_config.n_text_tokens,
        num_mappers=model_config.num_mappers,
    ).to(device)

    if use_ema and "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
    else:
        state_dict = ckpt["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    audio_encoder_state = ckpt.get("audio_encoder_state_dict")

    return model, model_config, audio_encoder_state


def generate(args: argparse.Namespace) -> None:
    device = torch.device(args.device or _default_device())

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("Using device: %s", device)

    # Load model
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, model_config, audio_encoder_state = load_checkpoint(
        args.checkpoint, device, use_ema=not args.no_ema,
    )

    # Tokenizer
    tokenizer = Tokenizer()

    # Encode audio
    logger.info("Encoding audio: %s", args.audio_path)
    audio_encoder = AudioEncoder(d_model=model_config.d_model).to(device)
    if args.audio_encoder is not None:
        state = torch.load(args.audio_encoder, map_location=device, weights_only=True)
        audio_encoder.load_state_dict(state)
        logger.info("Loaded audio encoder weights from %s", args.audio_encoder)
    elif audio_encoder_state is not None:
        audio_encoder.load_state_dict(audio_encoder_state)
        logger.info("Loaded audio encoder weights from checkpoint")
    else:
        logger.warning(
            "No audio encoder state in checkpoint - using random projection "
            "(features will not match training data)"
        )
    audio_encoder.eval()
    waveform = AudioEncoder.load_audio(args.audio_path).to(device)
    with torch.no_grad():
        audio_features = audio_encoder(waveform)  # (1, T, d_model)
    audio_mask = torch.ones(
        1, audio_features.shape[1], dtype=torch.bool, device=device
    )

    # Encode text prompt
    text_emb = None
    if args.prompt:
        logger.info("Encoding text prompt: %s", args.prompt)
        from ai_osu_maps.model.text_encoder import TextEncoder

        text_enc = TextEncoder(d_model=model_config.d_model).to(device)
        text_emb = text_enc.encode_text([args.prompt]).to(device)

    # Conditioning
    difficulty = torch.tensor([args.difficulty], dtype=torch.float32, device=device)
    cs_val = torch.tensor([args.cs], dtype=torch.float32, device=device)
    ar_val = torch.tensor([args.ar], dtype=torch.float32, device=device)
    od_val = torch.tensor([args.od], dtype=torch.float32, device=device)
    hp_val = torch.tensor([args.hp], dtype=torch.float32, device=device)
    mapper_id = torch.tensor([args.mapper], dtype=torch.long, device=device)
    year_val = torch.tensor([args.year], dtype=torch.float32, device=device)

    # Generation config
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        max_tokens=args.max_tokens,
    )

    # Generate
    logger.info("Generating beatmap...")
    token_ids = sample_autoregressively(
        model, tokenizer, audio_features,
        difficulty, cs_val, ar_val, od_val, hp_val,
        mapper_id, year_val,
        gen_config,
        audio_mask=audio_mask,
        text_emb=text_emb,
        device=device,
        stream=args.stream,
    )
    logger.info("Generated %d tokens", len(token_ids))

    # Convert tokens to events
    events = tokens_to_events(token_ids, tokenizer)
    logger.info("Decoded %d events", len(events))

    # Post-process to .osu
    postprocessor = Postprocessor(bpm=args.bpm, offset=args.offset)

    # Try to generate timing from events first
    timing = postprocessor.generate_timing(events)

    audio_filename = os.path.basename(args.audio_path)
    beatmap_config = BeatmapConfig(
        audio_filename=audio_filename,
        title="Generated Beatmap",
        title_unicode="Generated Beatmap",
        artist="AI",
        artist_unicode="AI",
        creator="osumaps-ar",
        version=f"AR {args.difficulty:.1f}*",
        hp_drain_rate=args.hp,
        circle_size=args.cs,
        overall_difficulty=args.od,
        approach_rate=args.ar,
        bpm=args.bpm,
        offset=args.offset,
    )

    result = postprocessor.generate(events, beatmap_config, timing or None)

    osu_path = postprocessor.write_result(result, args.output_dir)
    logger.info("Wrote .osu file: %s", osu_path)

    if args.osz:
        osz_path = postprocessor.export_osz(osu_path, args.audio_path, args.output_dir)
        logger.info("Wrote .osz file: %s", osz_path)


if __name__ == "__main__":
    generate(parse_args())

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor

MERT_MODEL_NAME = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE = 24000
MERT_HIDDEN_DIM = 768
MERT_NUM_LAYERS = 13


# 30s at 24kHz; MERT produces ~75Hz frames, so 30s -> ~2250 frames
MERT_CHUNK_SAMPLES = 30 * MERT_SAMPLE_RATE
MAX_AUDIO_FRAMES = 512


class AudioEncoder(nn.Module):
    def __init__(self, d_model: int = 512, max_frames: int = MAX_AUDIO_FRAMES) -> None:
        super().__init__()

        self.max_frames = max_frames

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_MODEL_NAME,
            trust_remote_code=True,
        )
        self.mert = AutoModel.from_pretrained(
            MERT_MODEL_NAME,
            trust_remote_code=True,
        )

        for param in self.mert.parameters():
            param.requires_grad = False

        self.layer_weights = nn.Parameter(torch.ones(MERT_NUM_LAYERS))
        self.projection = nn.Linear(MERT_HIDDEN_DIM, d_model)

    def _encode_chunk(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run MERT on a single chunk, return weighted hidden states (B, T, H)."""
        outputs = self.mert(waveform, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (L, B, T, H)
        weights = F.softmax(self.layer_weights, dim=0)
        return torch.einsum("l,lbth->bth", weights, hidden_states)

    def _split_into_chunks(self, waveform_1d: torch.Tensor) -> list[torch.Tensor]:
        """Split a 1D waveform into MERT-sized chunks.

        Returns at least one chunk. Skips very short tails (<1s).
        """
        if waveform_1d.shape[0] < MERT_SAMPLE_RATE:
            waveform_1d = F.pad(
                waveform_1d,
                (0, MERT_SAMPLE_RATE - waveform_1d.shape[0]),
            )

        T_samples = waveform_1d.shape[0]
        chunks = []
        for start in range(0, T_samples, MERT_CHUNK_SAMPLES):
            end = min(start + MERT_CHUNK_SAMPLES, T_samples)
            if end - start < MERT_SAMPLE_RATE:
                break
            chunks.append(waveform_1d[start:end])

        if not chunks:
            chunks.append(waveform_1d)

        return chunks

    def _pool_to_fixed_length(self, features: torch.Tensor) -> torch.Tensor:
        """Pool variable-length frame features to max_frames via mean pooling.

        Args:
            features: (T_frames, d_model)

        Returns:
            (min(T_frames, max_frames), d_model)
        """
        T = features.shape[0]
        if T <= self.max_frames:
            return features
        indices = torch.linspace(0, T, self.max_frames + 1, dtype=torch.long)
        pooled = []
        for i in range(self.max_frames):
            pooled.append(features[indices[i] : indices[i + 1]].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _encode_single(self, waveform_1d: torch.Tensor) -> torch.Tensor:
        """Encode a single waveform (no batch dim) through chunked MERT.

        Args:
            waveform_1d: (T_samples,) at 24kHz

        Returns:
            (max_frames, d_model)
        """
        chunks = self._split_into_chunks(waveform_1d)

        chunk_outputs = []
        for chunk in chunks:
            chunk_outputs.append(self._encode_chunk(chunk.unsqueeze(0)))

        weighted = torch.cat(chunk_outputs, dim=1)  # (1, T_frames, H)
        projected = self.projection(weighted.squeeze(0))  # (T_frames, d_model)
        return self._pool_to_fixed_length(projected)

    def encode_batch(
        self,
        waveforms: list[torch.Tensor],
        *,
        chunk_batch_size: int = 8,
    ) -> list[torch.Tensor]:
        """Encode multiple waveforms with batched MERT chunk processing.

        Groups same-length chunks across songs into batched forward passes
        for better GPU utilization than processing songs one at a time.

        Args:
            waveforms: List of 1D waveforms (T_samples,) at 24kHz, on model device
            chunk_batch_size: Max MERT chunks per forward pass

        Returns:
            List of (T_frames, d_model) tensors, one per input waveform
        """
        with torch.no_grad():
            # Split all songs into chunks
            songs_chunks = [self._split_into_chunks(w) for w in waveforms]

            # Separate full-length (batchable) and partial chunks
            full_chunks: list[tuple[int, int, torch.Tensor]] = []
            partial_chunks: list[tuple[int, int, torch.Tensor]] = []

            for song_idx, chunks in enumerate(songs_chunks):
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk.shape[0] == MERT_CHUNK_SAMPLES:
                        full_chunks.append((song_idx, chunk_idx, chunk))
                    else:
                        partial_chunks.append((song_idx, chunk_idx, chunk))

            # Encode full-length chunks in batches (all same size, no padding waste)
            encoded: dict[tuple[int, int], torch.Tensor] = {}

            for i in range(0, len(full_chunks), chunk_batch_size):
                batch = full_chunks[i : i + chunk_batch_size]
                stacked = torch.stack([c for _, _, c in batch])
                weighted = self._encode_chunk(stacked)  # (B, T_frames, H)
                for j, (song_idx, chunk_idx, _) in enumerate(batch):
                    encoded[(song_idx, chunk_idx)] = weighted[j]

            # Encode partial chunks individually
            for song_idx, chunk_idx, chunk in partial_chunks:
                weighted = self._encode_chunk(chunk.unsqueeze(0))  # (1, T_frames, H)
                encoded[(song_idx, chunk_idx)] = weighted[0]

            # Reassemble per song: concatenate chunks, project, pool
            results = []
            for song_idx, chunks in enumerate(songs_chunks):
                frames = [encoded[(song_idx, i)] for i in range(len(chunks))]
                weighted = torch.cat(frames, dim=0)  # (total_frames, H)
                projected = self.projection(weighted)  # (total_frames, d_model)
                results.append(self._pool_to_fixed_length(projected))

            return results

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio waveforms into fixed-length frame features.

        Processes each item individually to avoid padding waste inside
        MERT (which is frozen/no_grad, so no batch efficiency loss).

        Args:
            waveform: (B, T_samples) at 24kHz (may be zero-padded from collation)

        Returns:
            (B, max_frames, d_model)
        """
        with torch.no_grad():
            results = []
            for i in range(waveform.shape[0]):
                # Trim trailing zeros (collation padding)
                wav_i = waveform[i]
                nonzero = wav_i.nonzero()
                if nonzero.numel() > 0:
                    actual_len = nonzero[-1].item() + 1
                    wav_i = wav_i[:actual_len]
                results.append(self._encode_single(wav_i))

        # Pad to max_frames if any are shorter
        max_t = max(r.shape[0] for r in results)
        padded = []
        for r in results:
            if r.shape[0] < max_t:
                pad = torch.zeros(max_t - r.shape[0], r.shape[1], device=r.device)
                r = torch.cat([r, pad], dim=0)
            padded.append(r)

        return torch.stack(padded, dim=0)  # (B, max_frames, d_model)

    @staticmethod
    def load_audio(audio_path: str | Path) -> torch.Tensor:
        """Load an audio file and resample to 24kHz.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Waveform tensor of shape (1, T_samples) at 24kHz.
        """
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert stereo to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != MERT_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, MERT_SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor

MERT_MODEL_NAME = "m-a-p/MERT-v1-330M"
MERT_SAMPLE_RATE = 24000
MERT_HIDDEN_DIM = 1024
MERT_NUM_LAYERS = 25


class AudioEncoder(nn.Module):
    def __init__(self, d_model: int = 512) -> None:
        super().__init__()

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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform into frame-level features.

        Args:
            waveform: (B, T_samples) at 24kHz

        Returns:
            (B, T_audio, d_model) at ~75Hz
        """
        with torch.no_grad():
            outputs = self.mert(waveform, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (B, T_audio, 1024) for each layer
        hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (L, B, T, H)

        weights = F.softmax(self.layer_weights, dim=0)  # (L,)
        weighted = torch.einsum("l,lbth->bth", weights, hidden_states)  # (B, T, H)

        return self.projection(weighted)  # (B, T, d_model)

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

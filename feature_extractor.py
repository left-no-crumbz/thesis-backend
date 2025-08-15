import numpy as np
from transformers import ASTFeatureExtractor
import torch
from transformers.utils import is_speech_available
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

if is_speech_available():
    import torchaudio.compliance.kaldi as ta_kaldi

# based on the following literature that uses the Hamming window:
    # https://arxiv.org/pdf/2505.15136
    # https://arxiv.org/pdf/2409.05924
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11007653
class ASTFeatureExtractorHamming(ASTFeatureExtractor):
    """
    Custom AST Feature Extractor that uses Hamming window instead of Hann/Hanning.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override the window for numpy-based processing (when torchaudio is not available)
        if not is_speech_available():
            # Recalculate mel filters and window with hamming
            mel_filters = mel_filter_bank(
                num_frequency_bins=257,
                num_mel_filters=self.num_mel_bins,
                min_frequency=20,
                max_frequency=self.sampling_rate // 2,
                sampling_rate=self.sampling_rate,
                norm=None,
                mel_scale="kaldi",
                triangularize_in_mel_space=True,
            )
            self.mel_filters = mel_filters
            # Use hamming window instead of hann
            self.window = window_function(400, "hamming", periodic=False)
    
    def _extract_fbank_features(self, waveform: np.ndarray, max_length: int) -> np.ndarray:
        """
        Override to use hamming window type in torchaudio.compliance.kaldi.fbank
        """
        if is_speech_available():
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform,
                sample_frequency=self.sampling_rate,
                window_type="hamming",  # Changed from "hanning" to "hamming"
                num_mel_bins=self.num_mel_bins,
            )
        else:
            # Use numpy implementation with hamming window
            waveform = np.squeeze(waveform)
            fbank = spectrogram(
                waveform,
                self.window,  # This is now hamming window from __init__
                frame_length=400, # this follows the 25 ms frame length used in the paper (16000mhz * 0.025 = 400)
                hop_length=160, # this follows the hop length used in the paper (16000mhz * 0.01 = 160)
                fft_length=512,
                power=2.0,
                center=False,
                preemphasis=0.97,
                mel_filters=self.mel_filters,
                log_mel="log",
                mel_floor=1.192092955078125e-07,
                remove_dc_offset=True,
            ).T
            fbank = torch.from_numpy(fbank)

        n_frames = fbank.shape[0]
        difference = max_length - n_frames

        # pad or truncate, depending on difference
        if difference > 0:
            pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))
            fbank = pad_module(fbank)
        elif difference < 0:
            fbank = fbank[0:max_length, :]

        fbank = fbank.numpy()
        return fbank
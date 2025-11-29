
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import torch

# Mock chatterbox module before importing processor
sys.modules["chatterbox"] = MagicMock()
sys.modules["chatterbox.tts"] = MagicMock()
sys.modules["soundfile"] = MagicMock()

# Now import the processor
from tts_engine.chatterbox_processor import ChatterboxTTSProcessor

class TestChatterboxOptimization(unittest.TestCase):
    def setUp(self):
        # Mock _ChatterboxTTS to avoid ImportError in __init__
        self.mock_tts_cls = MagicMock()
        self.mock_tts_cls.from_pretrained.return_value = MagicMock(sr=24000)
        
        # Patch the module-level variable in chatterbox_processor
        self.patcher = patch("tts_engine.chatterbox_processor._ChatterboxTTS", self.mock_tts_cls)
        self.patcher.start()
        
        self.torch_patcher = patch("tts_engine.chatterbox_processor.torch", torch)
        self.torch_patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.torch_patcher.stop()

    @patch("tts_engine.chatterbox_processor.torchaudio")
    def test_prepare_audio_prompt_resampling(self, mock_torchaudio):
        # Setup mock torchaudio.load to return 48kHz audio (needs resampling to 24kHz)
        # 2 channels (stereo), 5 seconds
        original_sr = 48000
        duration_sec = 5
        waveform = torch.randn(2, original_sr * duration_sec)
        mock_torchaudio.load.return_value = (waveform, original_sr)
        
        # Mock transforms.Resample
        mock_resampler = MagicMock()
        mock_resampler.return_value = torch.randn(2, 24000 * duration_sec) # Mocked output
        mock_torchaudio.transforms.Resample.return_value = mock_resampler

        # Initialize processor
        processor = ChatterboxTTSProcessor(device="cpu")
        
        # Create a dummy file path
        dummy_path = "dummy_input.wav"
        with patch("os.path.exists", return_value=True):
            optimized_path = processor._prepare_audio_prompt(dummy_path)

        # Verify load called
        mock_torchaudio.load.assert_called_with(dummy_path)
        
        # Verify Resample initialized and called
        mock_torchaudio.transforms.Resample.assert_called_with(orig_freq=48000, new_freq=24000)
        mock_resampler.assert_called()
        
        # Verify save called with target SR (24000)
        self.assertTrue(mock_torchaudio.save.called)
        args, _ = mock_torchaudio.save.call_args
        self.assertEqual(args[2], 24000) # 3rd arg is sample_rate
        
        print(f"Verified: Audio resampled from 48000 to 24000 and saved to {optimized_path}")

    @patch("tts_engine.chatterbox_processor.torchaudio")
    def test_prepare_audio_prompt_no_resampling(self, mock_torchaudio):
        # Setup mock torchaudio.load to return 24kHz audio (no resampling needed)
        original_sr = 24000
        duration_sec = 5
        waveform = torch.randn(1, original_sr * duration_sec)
        mock_torchaudio.load.return_value = (waveform, original_sr)

        # Initialize processor
        processor = ChatterboxTTSProcessor(device="cpu")
        
        dummy_path = "dummy_input_24k.wav"
        with patch("os.path.exists", return_value=True):
            optimized_path = processor._prepare_audio_prompt(dummy_path)

        # Verify Resample NOT called
        mock_torchaudio.transforms.Resample.assert_not_called()
        
        # Verify save called (still saves to temp file for consistency/mono mix)
        self.assertTrue(mock_torchaudio.save.called)
        args, _ = mock_torchaudio.save.call_args
        self.assertEqual(args[2], 24000)

        print(f"Verified: Audio with correct SR (24000) processed without resampling.")

if __name__ == "__main__":
    unittest.main()

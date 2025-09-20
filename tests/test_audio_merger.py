import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from pydub import AudioSegment

# We need to import the function from the module we are testing.
# Since the module is in the parent directory, we need to adjust the path.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.audio_merger import merge_audio_files

class TestAudioMerger(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and dummy audio files for testing."""
        self.test_dir = "temp_test_audio"
        os.makedirs(self.test_dir, exist_ok=True)
        self.audio_file1 = os.path.join(self.test_dir, "audio1.wav")
        self.audio_file2 = os.path.join(self.test_dir, "audio2.wav")
        self.output_file = os.path.join(self.test_dir, "merged.wav")

        # Create silent 100ms WAV files for testing
        silent_segment = AudioSegment.silent(duration=100)
        silent_segment.export(self.audio_file1, format="wav")
        silent_segment.export(self.audio_file2, format="wav")

    def tearDown(self):
        """Clean up the temporary directory and files after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_merge_audio_files_success(self):
        """Test that audio files are merged successfully under normal conditions."""
        # This test relies on ffmpeg being in the system's PATH

        # Ensure the converter path is not hardcoded
        # We can't easily test the "before" state without modifying the source,
        # but we can test the "after" state thoroughly.
        # To be safe, we'll unset the converter path if it was set by another test.
        if hasattr(AudioSegment, 'converter'):
            AudioSegment.converter = None

        file_paths = [self.audio_file1, self.audio_file2]
        result = merge_audio_files(file_paths, self.output_file)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.output_file))

        # Check that the merged file is roughly the sum of the two parts
        merged_audio = AudioSegment.from_wav(self.output_file)
        self.assertAlmostEqual(len(merged_audio), 200, delta=10)

    @patch('pydub.AudioSegment.export')
    def test_export_failure_is_handled_gracefully(self, mock_export):
        """
        Test that if pydub's export method fails, our function returns False.
        This simulates the error condition caused by a missing ffmpeg.
        """
        # We mock the export method itself to raise an exception.
        mock_export.side_effect = Exception("Could not find ffmpeg")

        file_paths = [self.audio_file1, self.audio_file2]

        # Our function should catch the exception and return False.
        result = merge_audio_files(file_paths, self.output_file)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()

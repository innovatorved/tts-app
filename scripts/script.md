
## Example Usage of the Script
```bash
python main.py --text_file "example/text1.txt" --output_dir "example/my_audio" --device "mps" --num-workers 2 --merge_output
```

## Example Usage of the Conversation Script
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py --conversation example/conversation.txt --output_dir example/conversation_output --output_filename_base conversation --male_voice am_michael --female_voice af_heart --merge_output --threads 4 
```

## Example Usage with Chatterbox Turbo (Default - No Voice Cloning)
```bash
# Uses Chatterbox Turbo TTS without voice cloning (default behavior)
python main.py --text "Hello, this is a test of Chatterbox Turbo TTS!" --output_dir "example/turbo_audio" --device "mps" --engine chatterbox
```

## Example Usage with Chatterbox Turbo + Voice Cloning
```bash
# Uses Chatterbox Turbo TTS with voice cloning from a reference audio file
python main.py --text_file "example/text1.txt" --output_dir "example/cloned_audio" --device "mps" --engine chatterbox --cb_voice_cloning --cb_audio_prompt "path/to/reference_audio.wav" --cb_exaggeration 0.5 --cb_cfg_weight 0.5
```

## Example Usage of the WebUI
```bash
python main.py --text_file "example/text1.txt" --output_dir "example/my_audio" --device "cpu" --num-workers 1 --merge_output --engine chatterbox --cb_exaggeration 0.5 --cb_cfg_weight 0.5 --cb_voice_cloning --cb_audio_prompt output.wav
```

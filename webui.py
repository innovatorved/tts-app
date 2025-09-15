import gradio as gr
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, List

# Import our TTS modules
from tts_engine.processor import KokoroTTSProcessor
from tts_engine.chatterbox_processor import ChatterboxTTSProcessor
from utils.pdf_parser import extract_text_from_pdf
from utils.file_handler import ensure_dir_exists
from utils.text_file_parser import extract_text_from_txt
from utils.audio_merger import merge_audio_files
from utils.conversation_parser import (
    extract_conversation_from_text,
    get_voice_for_speaker,
)
from utils.split_text import split_text_into_chunks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSWebUI:
    def __init__(self):
        self.kokoro_processor = None
        self.chatterbox_processor = None
        self.temp_dir = tempfile.mkdtemp()
        
    def initialize_processors(self, engine: str, device: str = None):
        """Initialize TTS processors based on selected engine."""
        try:
            if engine == "kokoro":
                if self.kokoro_processor is None:
                    self.kokoro_processor = KokoroTTSProcessor(device=device)
                return "Kokoro processor initialized successfully!"
            elif engine == "chatterbox":
                if self.chatterbox_processor is None:
                    self.chatterbox_processor = ChatterboxTTSProcessor(device=device)
                return "Chatterbox processor initialized successfully!"
        except Exception as e:
            return f"Error initializing {engine} processor: {str(e)}"
    
    def process_text_input(self, 
                          text: str, 
                          engine: str,
                          output_dir: str,
                          lang: str = "a",
                          voice: str = "af_heart",
                          speed: float = 1.0,
                          device: str = None,
                          merge_output: bool = False,
                          audio_prompt_path: str = None) -> tuple:
        """Process direct text input."""
        if not text.strip():
            return "Please enter some text to convert.", None
        
        try:
            # Initialize processor
            init_msg = self.initialize_processors(engine, device)
            if "Error" in init_msg:
                return init_msg, None
            
            # Set output directory
            if not output_dir:
                output_dir = os.path.join(self.temp_dir, "output")
            ensure_dir_exists(output_dir)
            
            base_filename = "tts_output"
            
            if engine == "kokoro":
                processor = self.kokoro_processor
                processor.set_generation_params(voice=voice, speed=speed)
                audio_files = processor.text_to_speech(
                    text=text,
                    output_dir=output_dir,
                    base_filename=base_filename,
                    voice=voice,
                    speed=speed
                )
            else:  # chatterbox
                processor = self.chatterbox_processor
                # For Chatterbox, audio_prompt_path is required
                if not audio_prompt_path:
                    return "Chatterbox requires a reference audio file. Please upload one.", None
                
                audio_files = processor.text_to_speech(
                    text=text,
                    output_dir=output_dir,
                    base_filename=base_filename,
                    audio_prompt_path=audio_prompt_path
                )
            
            if not audio_files:
                return "No audio files were generated.", None
            
            # Merge if requested
            if merge_output and len(audio_files) > 1:
                merged_file = os.path.join(output_dir, f"{base_filename}_merged.wav")
                merge_audio_files(audio_files, merged_file)
                return f"Generated {len(audio_files)} audio segments and merged into single file.", merged_file
            
            return f"Generated {len(audio_files)} audio segments.", audio_files[0] if len(audio_files) == 1 else None
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return f"Error processing text: {str(e)}", None
    
    def process_file_input(self,
                          file_obj,
                          engine: str,
                          output_dir: str,
                          lang: str = "a",
                          voice: str = "af_heart",
                          speed: float = 1.0,
                          device: str = None,
                          merge_output: bool = False,
                          threads: int = 1,
                          paragraphs_per_chunk: int = 30,
                          audio_prompt_path: str = None) -> tuple:
        """Process file input (PDF or text)."""
        if file_obj is None:
            return "Please upload a file.", None
        
        try:
            file_path = file_obj.name
            file_ext = Path(file_path).suffix.lower()
            
            # Extract text based on file type
            if file_ext == '.pdf':
                text = extract_text_from_pdf(file_path)
                input_type = "PDF"
            elif file_ext in ['.txt', '.md']:
                text = extract_text_from_txt(file_path)
                input_type = "Text file"
            else:
                return "Unsupported file type. Please upload PDF or text files.", None
            
            if not text.strip():
                return f"No text could be extracted from the {input_type.lower()}.", None
            
            # Split into chunks if needed
            chunks = split_text_into_chunks(text, paragraphs_per_chunk)
            
            # Initialize processor
            init_msg = self.initialize_processors(engine, device)
            if "Error" in init_msg:
                return init_msg, None
            
            # Set output directory
            if not output_dir:
                output_dir = os.path.join(self.temp_dir, "output")
            ensure_dir_exists(output_dir)
            
            base_filename = Path(file_path).stem
            
            all_audio_files = []
            
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{base_filename}_chunk_{i}"
                
                if engine == "kokoro":
                    processor = self.kokoro_processor
                    processor.set_generation_params(voice=voice, speed=speed)
                    audio_files = processor.text_to_speech(
                        text=chunk,
                        output_dir=output_dir,
                        base_filename=chunk_filename,
                        voice=voice,
                        speed=speed
                    )
                else:  # chatterbox
                    processor = self.chatterbox_processor
                    # For Chatterbox, audio_prompt_path is required
                    if not audio_prompt_path:
                        return "Chatterbox requires a reference audio file. Please upload one.", None
                    
                    audio_files = processor.text_to_speech(
                        text=chunk,
                        output_dir=output_dir,
                        base_filename=chunk_filename,
                        audio_prompt_path=audio_prompt_path
                    )
                
                all_audio_files.extend(audio_files)
            
            if not all_audio_files:
                return "No audio files were generated.", None
            
            # Merge if requested
            if merge_output and len(all_audio_files) > 1:
                merged_file = os.path.join(output_dir, f"{base_filename}_merged.wav")
                merge_audio_files(all_audio_files, merged_file)
                return f"Processed {input_type} into {len(all_audio_files)} audio segments and merged.", merged_file
            
            return f"Processed {input_type} into {len(all_audio_files)} audio segments.", all_audio_files[0] if len(all_audio_files) == 1 else None
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return f"Error processing file: {str(e)}", None
    
    def process_conversation(self,
                            file_obj,
                            engine: str,
                            output_dir: str,
                            male_voice: str = "am_adam",
                            female_voice: str = "af_heart",
                            speed: float = 1.0,
                            device: str = None,
                            merge_output: bool = False,
                            audio_prompt_male_path: str | None = None,
                            audio_prompt_female_path: str | None = None) -> tuple:
        """Process conversation input."""
        if file_obj is None:
            return "Please upload a conversation file.", None
        
        try:
            file_path = file_obj.name
            text = extract_text_from_txt(file_path)
            
            if not text.strip():
                return "No text could be extracted from the conversation file.", None
            
            # Parse conversation
            conversation_parts = extract_conversation_from_text(text)
            
            if not conversation_parts:
                return "No conversation parts found. Please ensure the file contains 'Man:' and 'Woman:' prefixes.", None
            
            # Initialize processor
            init_msg = self.initialize_processors(engine, device)
            if "Error" in init_msg:
                return init_msg, None
            
            # Set output directory
            if not output_dir:
                output_dir = os.path.join(self.temp_dir, "output")
            ensure_dir_exists(output_dir)
            
            base_filename = Path(file_path).stem
            
            all_audio_files = []
            
            for i, (speaker, line) in enumerate(conversation_parts):
                if engine == "kokoro":
                    voice = male_voice if speaker == "Man" else female_voice
                    processor = self.kokoro_processor
                    processor.set_generation_params(voice=voice, speed=speed)
                    audio_files = processor.text_to_speech(
                        text=line,
                        output_dir=output_dir,
                        base_filename=f"{base_filename}_{speaker.lower()}_{i:03d}",
                        voice=voice,
                        speed=speed
                    )
                else:  # chatterbox
                    processor = self.chatterbox_processor
                    # For Chatterbox conversation, both male and female prompts are required
                    if not audio_prompt_male_path or not audio_prompt_female_path:
                        return (
                            "Chatterbox conversation requires two reference audio files: Male and Female.",
                            None,
                        )

                    chosen_prompt = (
                        audio_prompt_male_path if speaker == "Man" else audio_prompt_female_path
                    )

                    audio_files = processor.text_to_speech(
                        text=line,
                        output_dir=output_dir,
                        base_filename=f"{base_filename}_{speaker.lower()}_{i:03d}",
                        audio_prompt_path=chosen_prompt
                    )
                
                all_audio_files.extend(audio_files)
            
            if not all_audio_files:
                return "No audio files were generated.", None
            
            # Merge if requested
            if merge_output:
                merged_file = os.path.join(output_dir, f"{base_filename}_merged.wav")
                merge_audio_files(all_audio_files, merged_file)
                return f"Processed conversation into {len(all_audio_files)} audio segments and merged.", merged_file
            
            return f"Processed conversation into {len(all_audio_files)} audio segments.", None
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return f"Error processing conversation: {str(e)}", None

def create_ui():
    tts_ui = TTSWebUI()
    
    with gr.Blocks(title="TTS App Web UI", theme=gr.themes.Monochrome()) as interface:
        gr.Markdown("# 🎵 TTS: Text/PDF/Conversation to Audio Converter")
        gr.Markdown("Convert text, PDF files, or conversations to speech using Kokoro or Chatterbox TTS engines.")
        
        with gr.Tabs():
            # Text Input Tab
            with gr.TabItem("Text Input"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Enter Text",
                            placeholder="Type or paste your text here...",
                            lines=10
                        )
                        engine_text = gr.Radio(
                            ["kokoro", "chatterbox"],
                            label="TTS Engine",
                            value="kokoro"
                        )
                        with gr.Row():
                            lang_text = gr.Textbox(label="Language Code", value="a", visible=True)
                            voice_text = gr.Textbox(label="Voice", value="af_heart")
                            speed_text = gr.Number(label="Speed", value=1.0, minimum=0.5, maximum=2.0)
                        audio_prompt_text = gr.File(
                            label="Reference Audio (Required for Chatterbox)", 
                            file_types=[".wav", ".mp3", ".flac"],
                            visible=False
                        )
                        device_text = gr.Textbox(label="Device (cpu/cuda/mps)", placeholder="auto")
                        output_dir_text = gr.Textbox(label="Output Directory", placeholder="./output_audio")
                        merge_text = gr.Checkbox(label="Merge Output", value=False)
                        
                    with gr.Column():
                        text_status = gr.Textbox(label="Status", interactive=False)
                        text_audio = gr.Audio(label="Generated Audio", visible=False)
                        text_btn = gr.Button("Generate Speech", variant="primary")
            
            # File Input Tab
            with gr.TabItem("File Input (PDF/TXT)"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload File", file_types=[".pdf", ".txt", ".md"])
                        engine_file = gr.Radio(
                            ["kokoro", "chatterbox"],
                            label="TTS Engine",
                            value="kokoro"
                        )
                        with gr.Row():
                            lang_file = gr.Textbox(label="Language Code", value="a", visible=True)
                            voice_file = gr.Textbox(label="Voice", value="af_heart")
                            speed_file = gr.Number(label="Speed", value=1.0, minimum=0.5, maximum=2.0)
                        audio_prompt_file = gr.File(
                            label="Reference Audio (Required for Chatterbox)", 
                            file_types=[".wav", ".mp3", ".flac"],
                            visible=False
                        )
                        device_file = gr.Textbox(label="Device (cpu/cuda/mps)", placeholder="auto")
                        output_dir_file = gr.Textbox(label="Output Directory", placeholder="./output_audio")
                        with gr.Row():
                            threads_file = gr.Number(label="Threads", value=1, minimum=1, maximum=8)
                            paragraphs_file = gr.Number(label="Paragraphs per Chunk", value=30, minimum=1)
                        merge_file = gr.Checkbox(label="Merge Output", value=False)
                        
                    with gr.Column():
                        file_status = gr.Textbox(label="Status", interactive=False)
                        file_audio = gr.Audio(label="Generated Audio", visible=False)
                        file_btn = gr.Button("Process File", variant="primary")
            
            # Conversation Tab
            with gr.TabItem("Conversation"):
                with gr.Row():
                    with gr.Column():
                        conv_input = gr.File(label="Upload Conversation File", file_types=[".txt", ".md"])
                        engine_conv = gr.Radio(
                            ["kokoro", "chatterbox"],
                            label="TTS Engine",
                            value="kokoro"
                        )
                        with gr.Row():
                            male_voice_conv = gr.Textbox(label="Male Voice", value="am_adam")
                            female_voice_conv = gr.Textbox(label="Female Voice", value="af_heart")
                        with gr.Row():
                            audio_prompt_male_conv = gr.File(
                                label="Male Reference Audio (Chatterbox)", 
                                file_types=[".wav", ".mp3", ".flac"],
                                visible=False
                            )
                            audio_prompt_female_conv = gr.File(
                                label="Female Reference Audio (Chatterbox)", 
                                file_types=[".wav", ".mp3", ".flac"],
                                visible=False
                            )
                        speed_conv = gr.Number(label="Speed", value=1.0, minimum=0.5, maximum=2.0)
                        device_conv = gr.Textbox(label="Device (cpu/cuda/mps)", placeholder="auto")
                        output_dir_conv = gr.Textbox(label="Output Directory", placeholder="./output_audio")
                        merge_conv = gr.Checkbox(label="Merge Output", value=True)
                        
                    with gr.Column():
                        conv_status = gr.Textbox(label="Status", interactive=False)
                        conv_audio = gr.Audio(label="Generated Audio", visible=False)
                        conv_btn = gr.Button("Process Conversation", variant="primary")
        
        # Event handlers
        def update_engine_visibility(engine):
            if engine == "kokoro":
                # Show text/voice fields, hide audio prompts
                return (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                )
            else:
                # Hide text/voice fields, show audio prompts
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
        
        # Update visibility based on engine selection
        engine_text.change(update_engine_visibility, inputs=engine_text, outputs=[lang_text, voice_text, audio_prompt_text])
        engine_file.change(update_engine_visibility, inputs=engine_file, outputs=[lang_file, voice_file, audio_prompt_file])
        # For conversation, we need to toggle two prompt fields at once; return value goes to the first, we mirror for second
        def update_conv_visibility(engine):
            show_voices, show_voices2, show_prompts = update_engine_visibility(engine)
            return show_voices, show_voices2, show_prompts, show_prompts

        engine_conv.change(
            update_conv_visibility,
            inputs=engine_conv,
            outputs=[male_voice_conv, female_voice_conv, audio_prompt_male_conv, audio_prompt_female_conv],
        )
        
        # Text processing
        def process_text(text, engine, lang, voice, speed, audio_prompt, device, output_dir, merge):
            audio_prompt_path = audio_prompt.name if audio_prompt else None
            status, audio_file = tts_ui.process_text_input(
                text, engine, output_dir, lang, voice, speed, device, merge, audio_prompt_path
            )
            if audio_file and os.path.exists(audio_file):
                return status, gr.update(value=audio_file, visible=True)
            else:
                return status, gr.update(visible=False)
        
        text_btn.click(
            process_text,
            inputs=[text_input, engine_text, lang_text, voice_text, speed_text, audio_prompt_text, device_text, output_dir_text, merge_text],
            outputs=[text_status, text_audio]
        )
        
        # File processing
        def process_file(file, engine, lang, voice, speed, audio_prompt, device, output_dir, threads, paragraphs, merge):
            audio_prompt_path = audio_prompt.name if audio_prompt else None
            status, audio_file = tts_ui.process_file_input(
                file, engine, output_dir, lang, voice, speed, device, merge, threads, paragraphs, audio_prompt_path
            )
            if audio_file and os.path.exists(audio_file):
                return status, gr.update(value=audio_file, visible=True)
            else:
                return status, gr.update(visible=False)
        
        file_btn.click(
            process_file,
            inputs=[file_input, engine_file, lang_file, voice_file, speed_file, audio_prompt_file, device_file, output_dir_file, threads_file, paragraphs_file, merge_file],
            outputs=[file_status, file_audio]
        )
        
        # Conversation processing
        def process_conv(file, engine, male_voice, female_voice, audio_prompt_male, audio_prompt_female, speed, device, output_dir, merge):
            audio_prompt_male_path = audio_prompt_male.name if audio_prompt_male else None
            audio_prompt_female_path = audio_prompt_female.name if audio_prompt_female else None
            status, audio_file = tts_ui.process_conversation(
                file,
                engine,
                output_dir,
                male_voice,
                female_voice,
                speed,
                device,
                merge,
                audio_prompt_male_path,
                audio_prompt_female_path,
            )
            if audio_file and os.path.exists(audio_file):
                return status, gr.update(value=audio_file, visible=True)
            else:
                return status, gr.update(visible=False)
        
        conv_btn.click(
            process_conv,
            inputs=[
                conv_input,
                engine_conv,
                male_voice_conv,
                female_voice_conv,
                audio_prompt_male_conv,
                audio_prompt_female_conv,
                speed_conv,
                device_conv,
                output_dir_conv,
                merge_conv,
            ],
            outputs=[conv_status, conv_audio],
        )
    
    return interface

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)

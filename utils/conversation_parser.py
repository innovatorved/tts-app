import logging
import re
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def extract_conversation_from_text(text_content: str) -> List[Tuple[str, str]]:
    """
    Extracts conversation parts from text content.

    Args:
        text_content: String containing conversation text with "Man:" and "Woman:" prefixes.

    Returns:
        List of tuples containing (speaker_type, line_text)
    """
    if not text_content or not text_content.strip():
        logger.warning("No text content provided for conversation extraction.")
        return []

    # Normalize line endings
    normalized_text = text_content.replace("\r\n", "\n").replace("\r", "\n")

    # Split by newlines
    lines = normalized_text.split("\n")

    conversation_parts = []
    current_speaker = None
    current_text = []

    # Process each line to extract speaker and text
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for speaker patterns
        man_match = re.match(r"^(?:Man|MAN|man):(.*)$", line)
        woman_match = re.match(r"^(?:Woman|WOMAN|woman):(.*)$", line)

        if man_match:
            # If we have accumulated text for a previous speaker, save it
            if current_speaker and current_text:
                conversation_parts.append((current_speaker, " ".join(current_text)))
                current_text = []

            current_speaker = "Man"
            text_content = man_match.group(1).strip()
            if text_content:
                current_text.append(text_content)
        elif woman_match:
            # If we have accumulated text for a previous speaker, save it
            if current_speaker and current_text:
                conversation_parts.append((current_speaker, " ".join(current_text)))
                current_text = []

            current_speaker = "Woman"
            text_content = woman_match.group(1).strip()
            if text_content:
                current_text.append(text_content)
        elif current_speaker:  # Continue with the current speaker
            current_text.append(line)

    # Add the last part if there's any
    if current_speaker and current_text:
        conversation_parts.append((current_speaker, " ".join(current_text)))

    logger.info(f"Extracted {len(conversation_parts)} conversation parts")
    return conversation_parts


def get_voice_for_speaker(speaker: str, voice_config: Dict = None) -> str:
    """
    Returns appropriate voice model for the given speaker.

    Args:
        speaker: Speaker identifier ('Man' or 'Woman')
        voice_config: Optional configuration with custom voice mappings

    Returns:
        Voice model name suitable for the Kokoro TTS engine
    """
    # Use provided mapping or fallback to defaults
    if voice_config and speaker in voice_config:
        return voice_config[speaker]

    # Default mappings
    voice_mapping = {
        "Man": "am_adam",  # American male voice - strong and confident
        "Woman": "af_heart",  # American female voice
    }

    return voice_mapping.get(speaker, "af_heart")  # Default to female voice if unknown

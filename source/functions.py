import re

def remove_numeric(text):
    return re.sub(r'\d+', '', text)

def segment_text (text):
    segment_length = len(text) // 5
    segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]
    return segments[:5]
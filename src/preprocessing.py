import re
import html

def clean_text(text):
    """Normalize review text while preserving negation and useful words."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    # Keep apostrophes so don't/isn't/can't remain meaningful.
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

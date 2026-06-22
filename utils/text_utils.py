import re

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("", "")
    text = re.sub(r"[。]{2,}", "。", text)
    text = re.sub(r"[；]{2,}", "；", text)
    text = re.sub(r"[，]{2,}", "，", text)
    if not re.search(r"[a-zA-Z0-9一-龥]", text):
        return ""
    punct_ratio = len(re.findall(r"[。，“”；！？,.!?;]", text)) / max(len(text), 1)
    if punct_ratio > 0.6:
        return ""
    return text.strip()


def is_valid_text(text):
    text = text.strip()
    if len(text) < 10:
        return False
    if not re.search(r"[a-zA-Z0-9一-龥]", text):
        return False
    punct_ratio = len(re.findall(r"[。，“”；；！？,.!?;]", text)) / len(text)
    if punct_ratio > 0.5:
        return False
    return True

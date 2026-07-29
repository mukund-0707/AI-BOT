import re


def clean_text(raw_text: str) -> str:
    """
    Clean extracted document text without changing its meaning.
    """

    if not raw_text:
        return ""

    # Remove leading and trailing whitespace
    text = raw_text.strip()

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3+ blank lines with only 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text

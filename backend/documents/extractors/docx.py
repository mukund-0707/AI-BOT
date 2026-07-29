from docx import Document


def extract_docx(file_path):
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return [
        {
            "page_number": None,
            "text": "\n".join(paragraphs),
        }
    ]

def extract_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return [
        {
            "page_number": None,
            "text": text,
        }
    ]

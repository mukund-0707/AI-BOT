from pypdf import PdfReader


def extract_pdf(file_path):
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        pages.append(
            {
                "page_number": page_number,
                "text": page.extract_text() or "",
            }
        )

    return pages

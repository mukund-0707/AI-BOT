from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def extract_docx(file_path):
    document = Document(file_path)

    blocks = []

    for child in document.element.body.iterchildren():

        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()

            if text:
                # Prefix headings so chunks retain section context after splitting.
                level = (paragraph.style.name or "").removeprefix("Heading ")

                if level.isdigit():
                    text = f"{'#' * int(level)} {text}"

                blocks.append(text)

        elif child.tag.endswith("}tbl"):
            table = Table(child, document)

            rows = [
                [cell.text.strip().replace("\n", " ") for cell in row.cells]
                for row in table.rows
            ]

            # Repeat headers in wide tables to preserve context across chunks.
            header = rows[0] if len(table.columns) >= 3 else None

            for index, row in enumerate(rows):

                if header and index:
                    values = [
                        f"{head}: {value}"
                        for head, value in zip(header, row)
                        if value
                    ]
                else:
                    values = [value for value in row if value]

                # Merged cells repeat the same text across the span.
                cells = list(dict.fromkeys(values))

                if cells:
                    blocks.append(" | ".join(cells))

    return [
        {
            "page_number": None,
            "text": "\n".join(blocks),
        }
    ]

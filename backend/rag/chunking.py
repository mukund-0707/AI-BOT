def create_chunks(
    pages,
    chunk_size=1000,
    overlap=150,
):
    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space != -1:
                    end = last_space

            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "page_number": page["page_number"],
                        "text": chunk_text.strip(),
                    }
                )
                chunk_index += 1
            start = end - overlap
    return chunks

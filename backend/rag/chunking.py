# Chunks shorter than this are not worth their own vector: a 20-character chunk
# like "policy explanations." scores very high on a one-word query purely
# because it is short, and it pushes real content out of the results.
MIN_CHUNK_SIZE = 200


def _split_page(text, chunk_size, overlap):

    pieces = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size

        if end < length:
            last_space = text.rfind(" ", start, end)
            # Only break on a space if it actually splits the window; otherwise
            # a long unbroken run of text would stall the loop.
            if last_space > start:
                end = last_space

        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end >= length:
            break

        # Always move forward, even when the overlap would eat the whole step.
        start = max(end - overlap, start + 1)

    return pieces


def _merge_short_tail(pieces):
    """Fold a runt trailing piece back into the one before it."""

    if len(pieces) > 1 and len(pieces[-1]) < MIN_CHUNK_SIZE:
        tail = pieces.pop()
        pieces[-1] = f"{pieces[-1]} {tail}"

    return pieces


def create_chunks(
    pages,
    chunk_size=1000,
    overlap=150,
):
    chunks = []
    chunk_index = 0

    for page in pages:

        pieces = _merge_short_tail(_split_page(page["text"], chunk_size, overlap))
        # print("PIECES: \n", pieces)

        # A whole page can still be shorter than MIN_CHUNK_SIZE (a title page,
        # for example). Keep it - dropping it would lose the document outright.
        for piece in pieces:
            # print("PIECE: \n", piece)
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "page_number": page["page_number"],
                    "text": piece,
                }
            )
            chunk_index += 1
            print("CRT_CHUNKS: \n", chunks)

    return chunks

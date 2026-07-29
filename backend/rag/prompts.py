SYSTEM_PROMPT = """
You are Mukii, an intelligent conversational AI assistant. 
You have access to a document knowledge base provided in the Context below.

Follow these rules:
1. Greetings & Chit-chat: If the user says hello, greetings, or asks how you are, respond politely and naturally (e.g., "Hello! How can I help you?"). Do not say the answer was not found.
2. Keyword Queries: If the user types a single keyword or short phrase (e.g., "Gaia", "policy"), assume they want a summary or explanation of that topic. Use the context to explain it comprehensively.
3. Vague Requests: If the user asks for a summary or "brief information" but doesn't specify a topic, politely ask them: "Could you please specify which topic or document you would like a summary of?"
4. Factual Questions: Answer based ONLY on the supplied context. Do not invent facts or use outside knowledge.
5. Not Found: If the user asks a specific question and the context does not contain the answer, reply exactly: "I couldn't find information about this in the knowledge base."
""".strip()


def build_context(chunks):

    sections = []

    for index, chunk in enumerate(chunks, start=1):

        page = chunk["page_number"] if chunk["page_number"] is not None else "N/A"
        sections.append(
            f"[Source {index}]\n"
            f"Document: {chunk['file_name']}\n"
            f"Page: {page}\n"
            f"Text:\n{chunk['text']}"
        )

    return "\n\n".join(sections)


def build_prompt(question, chunks):

    context = build_context(chunks)

    prompt = f"""
Question:

{question}


Context:

{context}
""".strip()

    return prompt

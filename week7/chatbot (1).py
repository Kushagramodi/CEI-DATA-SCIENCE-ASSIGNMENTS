import cohere
import uuid


class Chatbot:
    def __init__(self, vectorstore, cohere_api_key, chat_model="command-a-03-2025"):
        self.vectorstore     = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co              = cohere.Client(cohere_api_key)
        self.chat_model      = chat_model

    # ── Stage 7: Answer Generation ───────────────────────────────────────
    def respond(self, user_message: str):
        # Stage 6 — retrieve relevant chunks
        retrieved_docs = self.vectorstore.retrieve(user_message)

        # System instruction — forces grounded answers only
        preamble = """You are a helpful document assistant.
Answer ONLY using the information provided in the document context below.
If the answer is not present in the documents, say:
'I could not find this information in the provided document.'
Always be concise, accurate, and reference the document content in your answer."""

        # Stage 7 — stream grounded response
        response = self.co.chat_stream(
            message=user_message,
            model=self.chat_model,
            documents=retrieved_docs,
            preamble=preamble,
            conversation_id=self.conversation_id,
        )

        return response, retrieved_docs

import cohere
import fitz
import time
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset


class VectorStore:
    def __init__(self, pdf_path, cohere_api_key, pinecone_api_key,
                 chunk_size=1000, use_rerank=True, embed_model="embed-english-v3.0",use_hybrid=True):
        self.embed_model      = embed_model
        self.use_rerank       = use_rerank
        self.use_hybrid = use_hybrid
        self.pdf_path         = pdf_path
        self.co               = cohere.Client(cohere_api_key)
        self.pinecone_api_key = pinecone_api_key
        self.chunks           = []
        self.embeddings       = []
        self.retrieve_top_k   = 10
        self.rerank_top_k     = 3

        self.load_document()
        self.split_text(chunk_size)
        self.embed_chunks()
        self.index_chunks()

    def load_document(self):

        if self.pdf_path.lower().endswith(".pdf"):

            self.pdf_text = self.extract_text_from_pdf(self.pdf_path)

        elif self.pdf_path.lower().endswith(".txt"):

            self.pdf_text = self.extract_text_from_txt(self.pdf_path)

        elif self.pdf_path.startswith("hf://"):
            dataset_name = self.pdf_path.replace("hf://", "")

            if "/" not in dataset_name:
                raise ValueError(
                    f"Invalid dataset name '{dataset_name}'. "
                    f"Use 'namespace/name' format e.g. 'rajpurkar/squad'"
                )

            self.pdf_text = self.extract_text_from_huggingface(dataset_name)

        else:

            raise ValueError("Unsupported document format.")



    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
        return text

    def extract_text_from_txt(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_text_from_huggingface(
            self,
            dataset_name,
            split="train",
            max_samples=100,
    ):
        # Validate namespace/name format
        if "/" not in dataset_name:
            raise ValueError(
                f"Invalid dataset '{dataset_name}'. "
                f"Use 'namespace/name' format e.g. 'rajpurkar/squad'"
            )

        dataset = load_dataset(dataset_name, split=split)

        # Auto-detect text column
        possible_columns = ["text", "context", "content",
                            "passage", "document", "documents", "answer"]
        text_column = None
        for col in possible_columns:
            if col in dataset.column_names:
                text_column = col
                break

        # Fallback to first column
        if text_column is None:
            text_column = dataset.column_names[0]

        texts = []
        max_samples = min(max_samples, len(dataset))
        for sample in dataset.select(range(max_samples)):
            value = sample[text_column]
            if isinstance(value, list):
                texts.append(" ".join(str(v) for v in value))
            elif isinstance(value, dict):
                texts.append(" ".join(str(v) for v in value.values()))
            else:
                texts.append(str(value))

        return "\n".join(texts)


    def split_text(self, chunk_size=1000):
        sentences     = self.pdf_text.split(". ")
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():               # filter empty chunks
                    self.chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk.strip():                       # flush last chunk
            self.chunks.append(current_chunk.strip())

    def keyword_score(self, query: str, chunks: list) -> list:
        """Simple keyword overlap scoring for hybrid search."""
        query_words = set(query.lower().split())
        scored = []
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            scored.append((overlap, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:5]]

    def embed_chunks(self, batch_size=90):
        total_chunks = len(self.chunks)
        for i in range(0, total_chunks, batch_size):
            batch = self.chunks[i:min(i + batch_size, total_chunks)]
            batch_embeddings = self.co.embed(
                texts=batch,
                input_type="search_document",
                model=self.embed_model          # configurable model
            ).embeddings
            self.embeddings.extend(batch_embeddings)

    def index_chunks(self):
        pc         = Pinecone(api_key=self.pinecone_api_key)
        index_name = "rag-qa-bot"

        # Delete old index to avoid dimension mismatch on model change
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            time.sleep(3)                       # wait for async deletion

        pc.create_index(
            name=index_name,
            dimension=len(self.embeddings[0]),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        self.index      = pc.Index(index_name)
        chunks_metadata = [{"text": chunk} for chunk in self.chunks]
        ids             = [str(i) for i in range(len(self.chunks))]
        self.index.upsert(vectors=zip(ids, self.embeddings, chunks_metadata))

    def retrieve(self, query: str) -> list:
        # Embed query
        query_emb = self.co.embed(
            texts=[query],
            model=self.embed_model,
            input_type="search_query",
        ).embeddings

        # Vector search
        res = self.index.query(
            vector=query_emb,
            top_k=self.retrieve_top_k,
            include_metadata=True,
        )

        vector_docs = [match["metadata"]["text"] for match in res["matches"]]

        # Hybrid search — keyword + vector
        if self.use_hybrid:
            keyword_indices = self.keyword_score(query, vector_docs)
            vector_indices = list(range(len(vector_docs)))
            combined_indices = list(dict.fromkeys(keyword_indices + vector_indices))
            docs_to_rerank = [vector_docs[i] for i in combined_indices]
        else:
            docs_to_rerank = vector_docs

        # Reranking
        if self.use_rerank:
            rerank_results = self.co.rerank(
                query=query,
                documents=docs_to_rerank,
                top_n=self.rerank_top_k,
                model="rerank-english-v3.0",
            )
            return [res["matches"][r.index]["metadata"] for r in rerank_results.results]
        else:
            return [match["metadata"] for match in res["matches"][:self.rerank_top_k]]

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """Weighted Reciprocal Rank Fusion. Dedup berdasarkan page_content."""
    scores = {}
    for retriever, weight in zip(retrievers, weights):
        for rank, doc in enumerate(retriever.invoke(query), start=1):
            key = doc.page_content
            prev = scores[key][0] if key in scores else 0.0
            scores[key] = (prev + weight / (rrf_k + rank), doc)

    ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports "
        "gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the "
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
        "rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. "
        "Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel "
        "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
        "alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]


class HybridRetriever:
    def __init__(self, documents, bm25_weight=0.5, k=4):
        if not 0 <= bm25_weight <= 1:
            raise ValueError("bm25_weight harus antara 0 dan 1")

        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

        embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            check_embedding_ctx_length=False,
        )
        self.vectorstore = Chroma.from_documents(
            documents, embeddings, collection_name="hybrid_search"
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    def search(self, query):
        return hybrid_retrieve(
            query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k,
        )

    def add_documents(self, new_docs):
        self.vectorstore.add_documents(new_docs)
        stored = self.vectorstore.get()
        all_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(stored["documents"], stored["metadatas"])
        ]
        # BM25Retriever tidak bisa menambah dokumen, jadi dibuat ulang
        self.bm25_retriever = BM25Retriever.from_documents(all_docs, k=self.k)


if __name__ == "__main__":
    retriever = HybridRetriever(documents, bm25_weight=0.5, k=3)
    for query in [
        "SKU-7742X specifications",
        "E_CONN_REFUSED error",
        "How do I authenticate?",
        "WCAG compliance",
    ]:
        print(f"\n{query}")
        for doc in retriever.search(query):
            print(f"  - {doc.page_content[:80]}")

from __future__ import annotations

from pathlib import Path
import hashlib
import math
import re
from typing import Iterable

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency fallback
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import fallback for constrained environments
    SentenceTransformer = None


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "travel_data.txt"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "travel_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TravelRAG:
    def __init__(self) -> None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self.model = None

        self.documents = self._load_documents()
        self.document_embeddings = self._embed_texts(self.documents)
        self.collection = None

        if chromadb is not None:
            self.client = chromadb.PersistentClient(path=str(PERSIST_DIR))

            try:
                self.client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass

            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            self._ingest_documents(self.documents)
        else:
            self.client = None

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True).tolist()

        return [self._fallback_embedding(text) for text in texts]

    @staticmethod
    def _fallback_embedding(text: str) -> list[float]:
        vector_size = 384
        values = [0.0] * vector_size

        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % vector_size
            direction = 1.0 if digest[4] % 2 == 0 else -1.0
            values[index] += direction

        magnitude = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / magnitude for value in values]

    def _load_documents(self) -> list[str]:
        if DATA_FILE.exists():
            raw_lines = DATA_FILE.read_text(encoding="utf-8").splitlines()
            documents = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
            if documents:
                return documents

        return [
            "Goa | Beaches, nightlife, water sports, best time Nov-Feb, budget 8k-15k",
            "Manali | Mountains, snow, best time Dec-Feb, budget 10k-20k",
            "Jaipur | Heritage, forts, best time Oct-Mar, budget 6k-12k",
            "Kerala | Backwaters, nature, best time Sep-Mar, budget 12k-25k",
        ]

    def _ingest_documents(self, documents: Iterable[str]) -> None:
        if self.collection is None:
            return

        docs = list(documents)
        if not docs:
            return

        embeddings = self._embed_texts(docs)
        ids = [f"travel-doc-{index}" for index in range(len(docs))]
        metadatas = [{"source": "travel_data.txt", "rank": index} for index in range(len(docs))]

        self.collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        query = query.strip()
        if not query:
            return "No travel context matched the query."

        if self.collection is None:
            return self._retrieve_context_from_memory(query, top_k)

        query_embedding = self._embed_texts([query])
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        documents = results.get("documents", [[]])[0]

        if not documents:
            return "No travel context matched the query."

        return "\n".join(documents)

    def _retrieve_context_from_memory(self, query: str, top_k: int) -> str:
        query_embedding = self._embed_texts([query])[0]
        scored_documents: list[tuple[float, str]] = []

        for document, embedding in zip(self.documents, self.document_embeddings):
            score = self._cosine_similarity(query_embedding, embedding)
            scored_documents.append((score, document))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        top_documents = [document for score, document in scored_documents[:top_k] if score > 0]

        if not top_documents:
            return "No travel context matched the query."

        return "\n".join(top_documents)

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_magnitude = math.sqrt(sum(value * value for value in left)) or 1.0
        right_magnitude = math.sqrt(sum(value * value for value in right)) or 1.0
        return numerator / (left_magnitude * right_magnitude)
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    - Loads knowledge base from .txt files in a folder
    - Uses sentence-transformers for embeddings (runs locally, no API cost)
    - ChromaDB as vector store (persistent)
    - Semantic search to find relevant context before generating response
    """

    def __init__(
        self,
        docs_folder: str = "./knowledge_base",
        collection_name: str = "ecommerce_kb",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        print("🔧 Initializing RAG Engine...")

        self.docs_folder = Path(docs_folder)
        self.chunk_size = chunk_size        # characters per chunk
        self.chunk_overlap = chunk_overlap  # overlap between chunks

        # Load embedding model locally
        print("📦 Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("C:/models/all-MiniLM-L6-v2")

        # Initialize ChromaDB (persistent vector store)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name

        # Build or load the knowledge base
        self._init_collection()
        print("✅ RAG Engine ready!")

    # ── File Loading ────────────────────────────────────────────────────────

    def _load_txt_files(self) -> list[dict]:
        """
        Load all .txt files from the docs folder.
        Each file is split into overlapping chunks.
        Returns a list of {id, content, category} dicts.
        """
        if not self.docs_folder.exists():
            raise FileNotFoundError(f"Docs folder not found: {self.docs_folder}")

        txt_files = sorted(self.docs_folder.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in: {self.docs_folder}")

        print(f"📂 Found {len(txt_files)} txt files: {[f.name for f in txt_files]}")

        all_chunks = []
        for file in txt_files:
            raw_text = file.read_text(encoding="utf-8")
            category = file.stem          # e.g. "return-refund-policy"
            chunks = self._chunk_text(raw_text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{file.stem}_chunk_{i}",
                    "content": chunk,
                    "category": category,
                })

        print(f"✂️  Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks for better retrieval.
        Tries to split at newlines to avoid cutting mid-sentence.
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            # If not at the end, try to break at a newline for clean splits
            if end < text_len:
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move forward with overlap
            start = end - self.chunk_overlap

        return chunks

    # ── Collection Management ───────────────────────────────────────────────

    def _init_collection(self):
        """Create or load the vector collection."""
        existing = [c.name for c in self.client.list_collections()]

        if self.collection_name in existing:
            print(f"📂 Loading existing collection: '{self.collection_name}'")
            self.collection = self.client.get_collection(self.collection_name)
            print(f"   → {self.collection.count()} chunks already indexed")
        else:
            print(f"⚠️ Collection '{self.collection_name}' not found. Creating empty collection.")
            print("   👉 Run 'python ingest.py' to populate the knowledge base.")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def _ingest_knowledge_base(self):
        """Load txt files, embed chunks, and store in ChromaDB."""
        documents = self._load_txt_files()
        print(f"📥 Ingesting {len(documents)} chunks into vector store...")

        docs      = [d["content"]  for d in documents]
        ids       = [d["id"]       for d in documents]
        metadatas = [{"category": d["category"]} for d in documents]

        # Embed all chunks locally
        embeddings = self.embedder.encode(
            docs, show_progress_bar=True, batch_size=32
        ).tolist()

        # Store in ChromaDB in batches (avoids memory issues with large KBs)
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            self.collection.add(
                documents=docs[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                ids=ids[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        print(f"✅ Ingested {len(docs)} chunks into vector store")

    def rebuild(self):
        """
        Force re-index: deletes existing collection and rebuilds from txt files.
        Call this whenever you update your docs folder.
        """
        print("🔄 Rebuilding knowledge base from txt files...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._ingest_knowledge_base()
        print("✅ Rebuild complete!")

    # ── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Semantic search: find top-k most relevant chunks for a query.

        Steps:
        1. Embed the user query
        2. Cosine similarity search in ChromaDB
        3. Return top-k results with scores
        """
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            similarity = 1 - results["distances"][0][i]   # cosine distance → similarity
            retrieved.append({
                "content":    results["documents"][0][i],
                "category":   results["metadatas"][0][i]["category"],
                "similarity": round(similarity, 4),
            })

        return retrieved

    def format_context(self, retrieved: list[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not retrieved:
            return "No specific information found in knowledge base."

        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            context_parts.append(
                f"[Source {i} — {doc['category'].replace('-', ' ').upper()} "
                f"| Relevance: {doc['similarity']:.2f}]\n{doc['content']}"
            )

        return "\n\n".join(context_parts)
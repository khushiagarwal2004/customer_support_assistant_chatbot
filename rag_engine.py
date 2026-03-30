import re
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
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        print("🔧 Initializing RAG Engine...")

        self.docs_folder = Path(docs_folder)
        self.chunk_size = chunk_size        # characters per chunk
        self.chunk_overlap = chunk_overlap  # overlap between chunks

        # Load embedding model locally
        print("📦 Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
        Structure-aware semantic chunking that keeps logical sections together.

        Instead of blindly cutting at every N characters (which splits products
        and policy sections in half), this detects the document's natural
        structure and splits along those boundaries.

        Three strategies tried in order:
          1. Separator lines (--- / ===) → for product catalogs
          2. ALL-CAPS section headers   → for policy & FAQ documents
          3. Fixed-size with overlap    → fallback for unstructured text
        """
        # Strategy 1: Documents with visual separator lines (product catalog)
        separator_pattern = re.compile(r'^[-=]{10,}$', re.MULTILINE)
        separator_count = len(separator_pattern.findall(text))

        if separator_count >= 3:
            chunks = self._split_by_separators(text, separator_pattern)
            if chunks:
                return chunks

        # Strategy 2: Documents with ALL-CAPS section headers (policies)
        chunks = self._split_by_section_headers(text)
        if chunks:
            return chunks

        # Strategy 3: Fallback for unstructured text
        return self._fixed_size_split(text)

    # ── Strategy 1: Separator-Based Splitting ────────────────────────────────

    def _split_by_separators(self, text: str, pattern) -> list[str]:
        """
        For documents like the product catalog that use -------- or ========
        lines as visual dividers.

        How it works:
        1. Split the text at every separator line
        2. Small fragments (like "PRODUCT: Name / SKU: Code") are recognized
           as headers and merged into the NEXT content block
        3. Result: each chunk = one complete product entry with all its details

        Example — before (old fixed-size):
          Chunk 1: "PRODUCT: iPhone 16... Description: Premium..."
          Chunk 2: "Pricing: ₹79,900... Stock: In Stock..."   ← price separated!

        Example — after (this method):
          Chunk 1: "PRODUCT: iPhone 16... Description... Pricing: ₹79,900
                    Stock: In Stock... Chatbot Notes: ..."     ← everything together!
        """
        blocks = pattern.split(text)
        blocks = [b.strip() for b in blocks if b.strip()]

        chunks = []
        pending_header = None

        for block in blocks:
            # Blocks under 120 chars are typically headers (product name + SKU,
            # category titles, etc.) — hold them to merge with the next block
            if len(block) < 120:
                pending_header = (pending_header + '\n' + block) if pending_header else block
            else:
                # This is a substantial content block — attach any pending header
                chunk = (pending_header + '\n\n' + block) if pending_header else block
                pending_header = None

                if len(chunk) > 2000:
                    # Safety: if somehow a section is huge, break it down
                    chunks.extend(self._fixed_size_split(chunk))
                elif len(chunk) > 50:
                    chunks.append(chunk)

        # Don't lose trailing headers (e.g. end-of-file notes)
        if pending_header and len(pending_header) > 30:
            chunks.append(pending_header)

        return chunks

    # ── Strategy 2: Section-Header Splitting ─────────────────────────────────

    def _split_by_section_headers(self, text: str) -> list[str]:
        """
        For policy/FAQ documents that use ALL-CAPS lines as section headers.

        Detects lines like:
          STANDARD RETURN WINDOW
          REFUND TIMELINES
          EXCHANGE POLICY

        Each header + everything below it (until the next header) = one chunk.
        """
        # Match lines that are entirely uppercase letters + spaces/punctuation,
        # at least 6 characters long — these are section headers
        header_pattern = re.compile(r'^([A-Z][A-Z \-&/—:()]{5,})$', re.MULTILINE)
        headers = list(header_pattern.finditer(text))

        if len(headers) < 2:
            return []

        chunks = []
        for i, match in enumerate(headers):
            start = match.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            chunk = text[start:end].strip()

            if len(chunk) < 30:
                continue
            if len(chunk) > 2000:
                chunks.extend(self._fixed_size_split(chunk))
            else:
                chunks.append(chunk)

        return chunks

    # ── Strategy 3: Fixed-Size Fallback ──────────────────────────────────────

    def _fixed_size_split(self, text: str) -> list[str]:
        """
        Original fixed-size chunking with overlap — used as a fallback when
        no structural patterns are detected, or to break down oversized sections.
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            if end < text_len:
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

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
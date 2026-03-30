import hashlib
import re
import chromadb
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer



class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    - Loads knowledge base from .txt files in a folder
    - Uses sentence-transformers for embeddings (runs locally, no API cost)
    - ChromaDB as vector store (persistent)
    - Hybrid retrieval: dense (cosine) + sparse (BM25), merged with RRF
    """

    def __init__(
        self,
        docs_folder: str = "./knowledge_base",
        collection_name: str = "ecommerce_kb",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        rrf_k: int = 60,
    ):
        print("🔧 Initializing RAG Engine...")

        self.docs_folder = Path(docs_folder)
        self.chunk_size = chunk_size        # characters per chunk
        self.chunk_overlap = chunk_overlap  # overlap between chunks
        self.rrf_k = rrf_k                  # Reciprocal Rank Fusion constant

        # BM25 keyword index (built from Chroma docs, parallel to self._bm25_docs)
        self.bm25 = None
        self._bm25_docs: list[dict] = []

        # Load embedding model locally
        print("📦 Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB (persistent vector store)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name

        # Build or load the knowledge base
        self._init_collection()
        self._build_bm25_index()
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
        self._build_bm25_index()

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

    # ── BM25 + hybrid helpers ───────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphanumeric tokens for BM25 (SKUs like EL-PHN-029 → el, phn, 029)."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_bm25_index(self):
        """Rebuild BM25 from all documents currently in Chroma (IDs must match)."""
        self.bm25 = None
        self._bm25_docs = []

        n = self.collection.count()
        if n == 0:
            print("🔍 BM25 keyword index: empty (run ingest.py)")
            return

        data = self.collection.get(include=["documents", "metadatas"])
        ids = data["ids"]
        docs = data["documents"]
        metas = data["metadatas"]

        tokenized: list[list[str]] = []
        self._bm25_docs = []
        for i, doc in enumerate(docs):
            toks = self._tokenize(doc)
            if not toks:
                toks = ["_empty_"]
            tokenized.append(toks)
            self._bm25_docs.append({
                "id": ids[i],
                "content": doc,
                "category": metas[i]["category"],
            })

        self.bm25 = BM25Okapi(tokenized)
        print(f"🔍 BM25 keyword index ready ({len(self._bm25_docs)} documents)")

    def _chunk_id_for_content(self, content: str) -> str:
        """Map exact chunk text to Chroma id (for RRF); fallback if not found."""
        for d in self._bm25_docs:
            if d["content"] == content:
                return d["id"]
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
        return f"_hash_{digest}"

    def _semantic_search(self, query: str, n_results: int) -> list[dict]:
        """Dense retrieval: cosine similarity in ChromaDB."""
        n = self.collection.count()
        if n == 0:
            return []
        n_results = min(max(1, n_results), n)

        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        docs_row = results["documents"][0]
        for i in range(len(docs_row)):
            content = docs_row[i]
            similarity = 1 - results["distances"][0][i]
            out.append({
                "id": self._chunk_id_for_content(content),
                "content": content,
                "category": results["metadatas"][0][i]["category"],
                "similarity": round(similarity, 4),
            })
        return out

    def _keyword_search(self, query: str, n_results: int) -> list[dict]:
        """Sparse retrieval: BM25 over the same chunks as the vector store."""
        if self.bm25 is None or not self._bm25_docs:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            stripped = query.strip().lower()
            tokens = [stripped] if stripped else []

        if not tokens:
            return []

        scores = list(self.bm25.get_scores(tokens))
        # Strong boost when the normalized query appears verbatim (SKUs, exact product lines)
        q_norm = re.sub(r"\s+", " ", query.strip().lower())
        if q_norm:
            boost = 25.0
            for i in range(len(scores)):
                if q_norm in self._bm25_docs[i]["content"].lower():
                    scores[i] += boost

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = ranked[: min(n_results, len(ranked))]

        out = []
        for idx, sc in top:
            d = self._bm25_docs[idx]
            out.append({
                "id": d["id"],
                "content": d["content"],
                "category": d["category"],
                "similarity": round(float(sc), 4),
            })
        return out

    def _rrf_merge(
        self,
        semantic: list[dict],
        keyword: list[dict],
        top_k: int,
        query: str = "",
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion: combine two ranked lists without score calibration.

        score(d) = sum 1 / (k + rank_i) for each list where d appears.
        Optional: +1.0 when chunk text contains a SKU/code detected in the query.
        """
        k = self.rrf_k
        merged: dict[str, dict] = {}

        for rank, item in enumerate(semantic, start=1):
            cid = item["id"]
            merged.setdefault(cid, {"rrf": 0.0, "content": item["content"], "category": item["category"]})
            merged[cid]["rrf"] += 1.0 / (k + rank)
            merged[cid]["content"] = item["content"]
            merged[cid]["category"] = item["category"]

        for rank, item in enumerate(keyword, start=1):
            cid = item["id"]
            merged.setdefault(cid, {"rrf": 0.0, "content": item["content"], "category": item["category"]})
            merged[cid]["rrf"] += 1.0 / (k + rank)
            merged[cid]["content"] = item["content"]
            merged[cid]["category"] = item["category"]

        # Strong boost when query mentions a SKU / product code (dense search often ranks wrong phone #1)
        for sku in re.findall(r"\b[A-Z]{2,3}-[A-Z]{2,3}-\d{2,6}\b", query, re.I):
            sl = sku.lower()
            for _cid, data in merged.items():
                if sl in data["content"].lower():
                    data["rrf"] += 1.0

        ordered = sorted(merged.items(), key=lambda x: x[1]["rrf"], reverse=True)[:top_k]

        peak = ordered[0][1]["rrf"] if ordered else 1.0
        peak = peak if peak > 0 else 1.0
        out = []
        for _cid, data in ordered:
            sim = min(1.0, data["rrf"] / peak)
            out.append({
                "content": data["content"],
                "category": data["category"],
                "similarity": round(sim, 4),
            })
        return out

    # ── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Hybrid search: semantic (cosine) + keyword (BM25), merged with RRF.

        - Semantic helps paraphrases and broad intent ("good phone for photos").
        - Keyword helps exact names, SKUs, policy terms ("iPhone 16", "EL-PHN-029").
        """
        q = (query or "").strip()
        if not q:
            return []

        n = self.collection.count()
        if n == 0:
            return []

        # Broad "list all phones/mobiles" — hybrid top-K often misses many SKUs; pull catalog phone chunks directly.
        if self._is_list_phones_inventory_query(q):
            direct = self._retrieve_phone_catalog_chunks(limit=top_k)
            if direct:
                return direct

        candidate_k = max(top_k * 3, 12)
        candidate_k = min(candidate_k, n)

        semantic = self._semantic_search(q, candidate_k)

        if self.bm25 is None:
            out = semantic[:top_k]
            return self._finalize_for_phone_listing_query(q, out)

        keyword = self._keyword_search(q, candidate_k)

        if not keyword:
            merged = semantic[:top_k]
            return self._finalize_for_phone_listing_query(
                q, self._post_filter_for_listing_query(q, merged, top_k)
            )
        if not semantic:
            mx = max((item["similarity"] for item in keyword), default=0.0) or 1.0
            merged = [
                {
                    "content": item["content"],
                    "category": item["category"],
                    "similarity": round(min(1.0, item["similarity"] / mx), 4),
                }
                for item in keyword[:top_k]
            ]
            return self._finalize_for_phone_listing_query(
                q, self._post_filter_for_listing_query(q, merged, top_k)
            )

        merged = self._rrf_merge(semantic, keyword, top_k, query=q)
        return self._finalize_for_phone_listing_query(
            q, self._post_filter_for_listing_query(q, merged, top_k)
        )

    def _finalize_for_phone_listing_query(self, query: str, docs: list[dict]) -> list[dict]:
        """One chunk per phone SKU when user asked to list phones (covers all retrieve paths)."""
        if not docs:
            return docs
        if self._is_list_phones_inventory_query(query):
            return self._dedupe_chunks_by_phone_sku(docs)
        return docs

    def _is_list_phones_inventory_query(self, query: str) -> bool:
        t = query.lower()
        list_words = ("list", "show", "all", "every", "available")
        phone_words = ("mobile", "mobiles", "phone", "phones", "smartphone", "smartphones")
        return any(w in t for w in list_words) and any(w in t for w in phone_words)

    @staticmethod
    def _dedupe_chunks_by_phone_sku(docs: list[dict]) -> list[dict]:
        """
        Keep at most one chunk per mobile phone SKU (EL-PHN-xxx).

        Prevents duplicate lines in answers when the same SKU appears in multiple
        chunks (re-ingest quirks) or when post-filter mixes overlapping sources.
        Chunks without an EL-PHN- SKU line are kept as-is (e.g., comparison blurbs).
        """
        seen: set[str] = set()
        out: list[dict] = []
        for d in docs:
            content = d.get("content") or ""
            m = re.search(r"SKU:\s*(EL-PHN-\d+)", content, re.I)
            if m:
                sku = m.group(1).upper()
                if sku in seen:
                    continue
                seen.add(sku)
            out.append(d)
        return out

    def _retrieve_phone_catalog_chunks(self, limit: int) -> list[dict]:
        """
        Return product-catalog chunks that represent mobile phones (SKU prefix EL-PHN-).
        Uses the same chunk list as BM25/Chroma so IDs stay consistent with ingestion.
        """
        if not self._bm25_docs or limit <= 0:
            return []

        phones: list[dict] = []
        for d in self._bm25_docs:
            if d.get("category") != "product-catalog":
                continue
            content = d.get("content") or ""
            if "el-phn-" not in content.lower():
                continue
            phones.append(
                {
                    "content": content,
                    "category": d["category"],
                    "similarity": 0.99,
                }
            )

        phones = self._dedupe_chunks_by_phone_sku(phones)

        # Stable ordering helps the model iterate consistently (by SKU if present).
        def _sku_key(text: str) -> str:
            m = re.search(r"SKU:\s*(EL-PHN-\d+)", text, re.I)
            return m.group(1).upper() if m else text[:40]

        phones.sort(key=lambda x: _sku_key(x["content"]))
        return phones[:limit]

    def _post_filter_for_listing_query(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """
        For broad listing queries (e.g., "list all mobiles"), prioritize
        product catalog chunks and, for phone queries, prefer phone SKUs.
        """
        text = query.lower()
        list_words = ("list", "show", "all", "every", "available")
        is_listing = any(w in text for w in list_words)
        if not is_listing or not docs:
            return docs[:top_k]

        # Keep product catalog chunks first for broad inventory requests.
        catalog_first = sorted(
            docs,
            key=lambda d: (d.get("category") != "product-catalog", -d.get("similarity", 0.0)),
        )

        # If this is specifically about mobiles/phones, prefer phone product entries.
        if any(w in text for w in ("mobile", "mobiles", "phone", "phones", "smartphone", "smartphones")):
            phone_like = [
                d for d in catalog_first
                if "el-phn-" in d.get("content", "").lower()
                or "smartphone" in d.get("content", "").lower()
                or "phone" in d.get("content", "").lower()
            ]
            if phone_like:
                phone_like = self._dedupe_chunks_by_phone_sku(phone_like)
                return phone_like[:top_k]

        return catalog_first[:top_k]

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
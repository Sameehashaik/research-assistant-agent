# document_search.py
# Our first "tool" — wraps the entire RAG pipeline from Project 1 into
# something an agent can pick up and use on its own.
#
# What this does in plain English:
#   1. Loads your TXT/PDF files
#   2. Chops them into chunks (same strategy as Project 1)
#   3. Turns chunks into embeddings via OpenAI
#   4. Stores them in a FAISS index for fast search
#   5. Exposes a .search() method the agent will call
#
# The agent never sees this code — it only sees the tool's name + description.
# That description is how it decides "should I use this tool for this question?"

from langchain_core.tools import Tool
from pathlib import Path
import os
import sys
import re
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# cost_tracker lives in the project root, so we need to add it to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import CostTracker

# same model + dimensions we used in Project 1
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


class DocumentSearchTool:
    """
    Wraps our full RAG pipeline (load → chunk → embed → search) as a
    LangChain Tool the agent can decide to invoke.
    """

    def __init__(self, documents_dir: str = "data"):
        self.documents_dir = documents_dir
        self.chunks = []          # the actual text pieces after chunking
        self.chunk_sources = []   # parallel list — which file each chunk came from
        self.index = None         # FAISS index (None until load_documents is called)
        self.tracker = CostTracker(log_file="project2_costs.json")
        self._client = None       # lazy-loaded OpenAI client

    # -- OpenAI client (created once, reused) --

    def _get_client(self) -> OpenAI:
        """Lazy-create the OpenAI client so we only build it when we actually need it."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file")
            self._client = OpenAI(api_key=api_key)
        return self._client

    # -- Step 1: Load raw text from files --

    def _load_file(self, file_path: str) -> str:
        """Read a .txt or .pdf and return the raw text. Same logic as Project 1's loader."""
        path = Path(file_path)

        if path.suffix.lower() == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        elif path.suffix.lower() == ".pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            # grab text from every page, skip blanks
            return "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _clean_text(self, text: str) -> str:
        """Collapse extra whitespace so the chunker gets clean input."""
        text = re.sub(r"\n{3,}", "\n\n", text)   # 3+ newlines → 2
        text = re.sub(r" {2,}", " ", text)        # 2+ spaces → 1
        return text.strip()

    # -- Step 2: Chunk text into overlapping pieces --

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """
        Split on sentence boundaries with overlap.
        Overlap means the end of chunk N appears at the start of chunk N+1,
        so we don't lose context that sits right on a boundary.
        """
        if not text or not text.strip():
            return []

        # split after sentence-ending punctuation (.!?) followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # if adding this sentence would bust the limit, seal the chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # carry over the last few sentences as overlap for the next chunk
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) > overlap and overlap_sentences:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # don't forget whatever's left over
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # -- Step 3: Turn text into vectors --

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Call OpenAI's embedding API for a batch of texts.
        Returns a list of 1536-dim float vectors, one per input text.
        """
        if not texts:
            return []

        client = self._get_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        # log the spend so we can keep an eye on costs
        tokens_used = response.usage.total_tokens
        self.tracker.track_call(
            model="embedding-small",
            input_tokens=tokens_used,
            output_tokens=0,
            description=f"Embed {len(texts)} chunks for document search",
        )

        return [item.embedding for item in response.data]

    # -- Main public methods --

    def load_documents(self, file_paths: list[str]):
        """
        Full pipeline: load files → chunk → embed → build FAISS index.
        Call this once before searching. You can pass multiple files.
        """
        all_chunks = []
        all_sources = []

        for file_path in file_paths:
            raw_text = self._load_file(file_path)
            clean_text = self._clean_text(raw_text)
            chunks = self._chunk_text(clean_text)

            # remember which file each chunk belongs to (for citations later)
            source_name = Path(file_path).name
            all_chunks.extend(chunks)
            all_sources.extend([source_name] * len(chunks))

            print(f"  Loaded {source_name}: {len(chunks)} chunks")

        if not all_chunks:
            print("  No chunks created from documents.")
            return

        # embed everything in one batch (cheaper than one-by-one)
        embeddings = self._generate_embeddings(all_chunks)

        # build the FAISS index — IndexFlatL2 = brute-force exact search
        vectors = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        self.index.add(vectors)

        self.chunks = all_chunks
        self.chunk_sources = all_sources

        print(f"  Vector store ready: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 3) -> str:
        """
        The function the agent actually calls.
        Takes a natural-language query, embeds it, finds the closest chunks
        in FAISS, and returns a formatted string the agent can read.
        """
        if self.index is None or self.index.ntotal == 0:
            return "No documents loaded. Please upload documents first."

        # embed the query so we can compare it against stored vectors
        query_embedding = self._generate_embeddings([query])[0]
        query_vector = np.array([query_embedding], dtype=np.float32)

        # FAISS returns (distances, indices) — lower distance = more relevant
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        # build a human-readable result string for the agent
        formatted = "Document Search Results:\n\n"
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx == -1:  # FAISS pads with -1 when there aren't enough results
                continue
            formatted += f"[{rank}] From: {self.chunk_sources[idx]}\n"
            formatted += f"    {self.chunks[idx][:300]}\n"
            formatted += f"    (Relevance distance: {dist:.4f})\n\n"

        return formatted

    def as_tool(self) -> Tool:
        """
        Package this as a LangChain Tool.
        The name + description are what the agent reads to decide
        "should I use this tool right now?"
        """
        return Tool(
            name="search_documents",
            description=(
                "Search through your personal documents and notes. "
                "Use this when the question is about YOUR information, "
                "past notes, saved documents, or personal knowledge. "
                "Input should be a search query."
            ),
            func=self.search
        )

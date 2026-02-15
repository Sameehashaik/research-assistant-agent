"""
Document Search Tool - Wraps RAG from Project 1

This is a TOOL that the agent can choose to use.

Tool Schema:
- Name: "search_documents"
- Description: "Search through uploaded personal documents for information"
- Parameters: query (str)
- Returns: Relevant passages from documents

The agent will see this schema and decide when to use it!
"""

from langchain_core.tools import Tool
from pathlib import Path
import os
import sys
import re
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path for cost_tracker
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import CostTracker

# Constants (same as Project 1)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


class DocumentSearchTool:
    """
    Tool for searching personal documents using RAG.

    This wraps everything you built in Project 1:
    - Document loading (TXT, PDF)
    - Text chunking
    - Embeddings generation
    - FAISS vector search

    But now it's packaged as a TOOL that an agent can decide to use!
    """

    def __init__(self, documents_dir: str = "data"):
        self.documents_dir = documents_dir
        self.chunks = []          # The text chunks
        self.chunk_sources = []   # Which file each chunk came from
        self.index = None         # FAISS index
        self.tracker = CostTracker(log_file="project2_costs.json")
        self._client = None

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file")
            self._client = OpenAI(api_key=api_key)
        return self._client

    # ---- Document Loading (from Project 1) ----

    def _load_file(self, file_path: str) -> str:
        """Load a single file (TXT or PDF)."""
        path = Path(file_path)

        if path.suffix.lower() == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        elif path.suffix.lower() == ".pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    # ---- Chunking (from Project 1) ----

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Build overlap
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

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ---- Embeddings (from Project 1) ----

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using OpenAI."""
        if not texts:
            return []

        client = self._get_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        # Track cost
        tokens_used = response.usage.total_tokens
        self.tracker.track_call(
            model="embedding-small",
            input_tokens=tokens_used,
            output_tokens=0,
            description=f"Embed {len(texts)} chunks for document search",
        )

        return [item.embedding for item in response.data]

    # ---- Main Methods ----

    def load_documents(self, file_paths: list[str]):
        """
        Load documents into the vector store.

        This does everything Project 1 did:
        1. Load each file
        2. Chunk the text
        3. Generate embeddings
        4. Store in FAISS

        Args:
            file_paths: List of paths to documents
        """
        all_chunks = []
        all_sources = []

        for file_path in file_paths:
            # Load and clean
            raw_text = self._load_file(file_path)
            clean_text = self._clean_text(raw_text)

            # Chunk
            chunks = self._chunk_text(clean_text)

            # Track which file each chunk came from
            source_name = Path(file_path).name
            all_chunks.extend(chunks)
            all_sources.extend([source_name] * len(chunks))

            print(f"  Loaded {source_name}: {len(chunks)} chunks")

        if not all_chunks:
            print("  No chunks created from documents.")
            return

        # Generate embeddings
        embeddings = self._generate_embeddings(all_chunks)

        # Build FAISS index
        vectors = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        self.index.add(vectors)

        # Store chunks and sources
        self.chunks = all_chunks
        self.chunk_sources = all_sources

        print(f"  Vector store ready: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 3) -> str:
        """
        Search documents and return relevant passages.

        THIS is the function the agent will call!

        Args:
            query: User's search query
            k: Number of results to return

        Returns:
            Formatted string with results and sources
        """
        if self.index is None or self.index.ntotal == 0:
            return "No documents loaded. Please upload documents first."

        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Search FAISS
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        # Format results
        formatted = "Document Search Results:\n\n"
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx == -1:
                continue
            formatted += f"[{rank}] From: {self.chunk_sources[idx]}\n"
            formatted += f"    {self.chunks[idx][:300]}\n"
            formatted += f"    (Relevance distance: {dist:.4f})\n\n"

        return formatted

    def as_tool(self) -> Tool:
        """
        Convert to LangChain Tool format.

        This is what the agent actually sees!
        The name and description tell the agent WHEN to use this tool.
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

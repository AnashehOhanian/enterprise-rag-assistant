"""Document chunking, embedding, retrieval, and grounded answer generation.

SQLite stores document text and embedding vectors in a local file. A local
sentence-transformers model creates those embeddings. Answer generation
calls an OpenAI-compatible chat endpoint when ``LLM_API_KEY`` or
``OPENAI_API_KEY`` is set, and otherwise returns the retrieved passages
directly.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("enterprise_rag.engine")

_SYSTEM_PROMPT = (
    "You answer questions for an enterprise knowledge assistant. "
    "Use only the context passages. If the passages do not contain the answer, "
    "say that the indexed documents do not contain enough information. "
    "Do not invent policy, numbers, or citations. "
    "When you use a passage, mention its filename."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "Context passages:\n{context}\n\nQuestion: {question}"),
    ]
)

_CONTEXT_CHAR_LIMIT = 12000
_EMBED_BATCH_SIZE = 32
_EXCERPT_CHARS = 1000

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rag_documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT,
    chunk_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES rag_documents (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx
ON rag_chunks (document_id);

CREATE TABLE IF NOT EXISTS rag_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class RAGEngineError(Exception):
    """A pipeline or storage failure that the API can return to a client."""

    def __init__(self, message: str, status_code: int = 500, code: str = "rag_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from the environment."""

    sqlite_path: str
    embedding_model: str
    embedding_device: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_similarity: float
    max_chunks: int
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: int

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables.

        Returns:
            Validated settings.

        Raises:
            RAGEngineError: A required variable is missing or a value is out of range.
        """
        chunk_size = _int_env("CHUNK_SIZE", 800, minimum=200, maximum=4000)
        chunk_overlap = _int_env("CHUNK_OVERLAP", 120, minimum=0, maximum=1000)
        if chunk_overlap >= chunk_size:
            raise RAGEngineError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE.",
                code="config_error",
            )

        llm_api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        return cls(
            sqlite_path=str(_resolve_sqlite_path()),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
            embedding_device=_embedding_device(),
            embedding_dim=_int_env("EMBEDDING_DIM", 384, minimum=8, maximum=4096),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=_int_env("TOP_K", 4, minimum=1, maximum=20),
            min_similarity=_float_env("MIN_SIMILARITY", 0.2, minimum=0.0, maximum=1.0),
            max_chunks=_int_env("MAX_CHUNKS", 2000, minimum=1, maximum=10000),
            llm_api_base=os.getenv("LLM_API_BASE", "https://api.openai.com/v1").strip().rstrip("/"),
            llm_api_key=llm_api_key,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
            llm_timeout_seconds=_int_env("LLM_TIMEOUT_SECONDS", 60, minimum=5, maximum=300),
        )


@dataclass(frozen=True)
class IngestResult:
    """Summary of a stored document."""

    document_id: str
    filename: str
    content_type: str | None
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class Source:
    """A retrieved passage returned with an answer."""

    document_id: str
    filename: str
    chunk_index: int
    score: float
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class QueryResult:
    """An answer plus the passages that supported it."""

    answer: str
    mode: str
    sources: list[Source]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "answer": self.answer,
            "mode": self.mode,
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class _RetrievedChunk:
    """Internal retrieval row, including the full chunk text."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


class RAGEngine:
    """Chunk documents, store embeddings in SQLite, and answer questions."""

    def __init__(self, settings: Settings) -> None:
        """Store settings without opening the database or loading the model.

        Args:
            settings: Validated runtime settings.
        """
        self.settings = settings
        self._db_lock = threading.Lock()
        self._schema_ready = False
        self._model: Any = None
        self._model_lock = threading.Lock()

    @classmethod
    def from_env(cls) -> RAGEngine:
        """Create an engine from the current environment.

        Returns:
            An engine that connects on the first database call.
        """
        return cls(Settings.from_env())

    @property
    def llm_configured(self) -> bool:
        """Whether a language-model API key is available."""
        return bool(self.settings.llm_api_key)

    def check_database(self) -> bool:
        """Open the SQLite file if needed and run ``SELECT 1``.

        Returns:
            True when the database file accepts the probe.

        Raises:
            RAGEngineError: The file exists but schema setup failed.
        """
        try:
            with self._connection() as conn:
                conn.execute("SELECT 1")
            return True
        except RAGEngineError:
            raise
        except sqlite3.Error:
            logger.exception("database health check failed")
            return False

    def ingest(self, filename: str, text: str, content_type: str | None = None) -> IngestResult:
        """Chunk a document, embed the chunks, and store them.

        Args:
            filename: Name stored beside the document.
            text: Extracted document text.
            content_type: Optional MIME type.

        Returns:
            The new document id and chunk count.

        Raises:
            RAGEngineError: The text is empty, produces too many chunks, or storage fails.
        """
        cleaned = _clean_text(text)
        if not cleaned:
            raise RAGEngineError(
                "The document has no text to index.",
                status_code=400,
                code="invalid_document",
            )
        chunks = chunk_text(cleaned, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise RAGEngineError(
                "The document has no text to index.",
                status_code=400,
                code="invalid_document",
            )
        if len(chunks) > self.settings.max_chunks:
            raise RAGEngineError(
                f"The document produced {len(chunks)} chunks, above MAX_CHUNKS={self.settings.max_chunks}.",
                status_code=400,
                code="invalid_document",
            )

        vectors = self._embed(chunks)
        document_id = str(uuid.uuid4())
        stored_name = filename.strip()[:255] or "document.txt"
        stored_type = content_type[:100] if content_type else None
        rows = [
            (str(uuid.uuid4()), document_id, index, chunk, _pack_vector(vector))
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO rag_documents (id, filename, content_type, chunk_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (document_id, stored_name, stored_type, len(chunks)),
                )
                conn.executemany(
                    """
                    INSERT INTO rag_chunks (id, document_id, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except RAGEngineError:
            raise
        except sqlite3.Error as exc:
            logger.exception("failed to store document")
            raise RAGEngineError(
                "The document could not be stored.",
                status_code=503,
                code="database_error",
            ) from exc

        return IngestResult(
            document_id=str(document_id),
            filename=stored_name,
            content_type=stored_type,
            chunk_count=len(chunks),
        )

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        """Retrieve passages and answer a question from them.

        Args:
            question: Natural-language question.
            top_k: Optional number of passages. ``None`` uses ``TOP_K``.

        Returns:
            The answer, the response mode, and source excerpts.

        Raises:
            RAGEngineError: Retrieval or generation fails.
        """
        cleaned = question.strip()
        if not cleaned:
            raise RAGEngineError(
                "question must be a non-empty string.",
                status_code=400,
                code="bad_request",
            )
        if len(cleaned) > 4000:
            raise RAGEngineError(
                "question must be 4000 characters or fewer.",
                status_code=400,
                code="bad_request",
            )
        depth = self._resolve_top_k(top_k)
        matches = self._retrieve(cleaned, depth)
        kept = [match for match in matches if match.score >= self.settings.min_similarity]
        sources = [
            Source(
                document_id=match.document_id,
                filename=match.filename,
                chunk_index=match.chunk_index,
                score=round(match.score, 4),
                excerpt=_excerpt(match.content),
            )
            for match in kept
        ]
        if not kept:
            return QueryResult(
                answer=(
                    "No indexed passages were similar enough to answer that question. "
                    "Upload a relevant document or lower MIN_SIMILARITY."
                ),
                mode="none",
                sources=[],
            )
        if not self.llm_configured:
            return QueryResult(answer=_extractive_answer(cleaned, kept), mode="extractive", sources=sources)
        answer = self._generate(cleaned, kept)
        return QueryResult(answer=answer, mode="generated", sources=sources)

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the newest stored documents.

        Args:
            limit: Maximum rows, from 1 to 200.

        Returns:
            Document metadata in newest-first order.

        Raises:
            RAGEngineError: The database query fails.
        """
        bounded = self._resolve_limit(limit)
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, filename, content_type, chunk_count, created_at
                    FROM rag_documents
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (bounded,),
                ).fetchall()
        except RAGEngineError:
            raise
        except sqlite3.Error as exc:
            logger.exception("failed to list documents")
            raise RAGEngineError(
                "Documents could not be listed.",
                status_code=503,
                code="database_error",
            ) from exc
        documents: list[dict[str, Any]] = []
        for row in rows:
            created = row["created_at"]
            documents.append(
                {
                    "document_id": str(row["id"]),
                    "filename": row["filename"],
                    "content_type": row["content_type"],
                    "chunk_count": row["chunk_count"],
                    "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created),
                }
            )
        return documents

    def delete_document(self, document_id: uuid.UUID) -> None:
        """Delete a document. Chunks are removed by a foreign-key cascade.

        Args:
            document_id: Document primary key.

        Raises:
            RAGEngineError: The id does not exist or the database query fails.
        """
        try:
            with self._connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM rag_documents WHERE id = ?",
                    (str(document_id),),
                )
                deleted = cursor.rowcount
        except RAGEngineError:
            raise
        except sqlite3.Error as exc:
            logger.exception("failed to delete document")
            raise RAGEngineError(
                "The document could not be deleted.",
                status_code=503,
                code="database_error",
            ) from exc
        if deleted == 0:
            raise RAGEngineError("Document not found.", status_code=404, code="not_found")

    def close(self) -> None:
        """Forget schema state. SQLite connections are opened per request."""
        with self._db_lock:
            self._schema_ready = False

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection, creating the schema on first use.

        The connection commits when the block finishes and rolls back when
        the block raises. A lock keeps writes from overlapping in one process.

        Yields:
            An open connection with foreign keys enabled.

        Raises:
            RAGEngineError: The database file or schema cannot be prepared.
        """
        with self._db_lock:
            self._ensure_schema()
            try:
                conn = sqlite3.connect(self.settings.sqlite_path, timeout=30)
            except sqlite3.Error as exc:
                logger.exception("failed to open sqlite database")
                raise RAGEngineError(
                    "The SQLite database could not be opened. Check SQLITE_PATH.",
                    status_code=503,
                    code="database_unavailable",
                ) from exc
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _ensure_schema(self) -> None:
        """Create tables once and record the embedding dimension.

        Caller must hold ``_db_lock``.

        Raises:
            RAGEngineError: The file cannot be created or its dimension does not match settings.
        """
        if self._schema_ready:
            return
        path = Path(self.settings.sqlite_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path, timeout=30)
        except sqlite3.Error as exc:
            logger.exception("failed to create sqlite database")
            raise RAGEngineError(
                "The SQLite database could not be created. Check SQLITE_PATH.",
                status_code=503,
                code="database_unavailable",
            ) from exc
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(_SCHEMA_SQL)
            row = conn.execute(
                "SELECT value FROM rag_meta WHERE key = 'embedding_dim'"
            ).fetchone()
            expected = str(self.settings.embedding_dim)
            if row is None:
                conn.execute(
                    "INSERT INTO rag_meta (key, value) VALUES ('embedding_dim', ?)",
                    (expected,),
                )
            elif str(row["value"]) != expected:
                raise RAGEngineError(
                    f"The SQLite file was built with embedding dimension {row['value']}, "
                    f"while EMBEDDING_DIM is {expected}. Delete {path.name} before changing dimensions.",
                    code="config_error",
                )
            conn.commit()
        except RAGEngineError:
            conn.rollback()
            raise
        except sqlite3.Error as exc:
            conn.rollback()
            logger.exception("schema initialization failed")
            raise RAGEngineError(
                "The SQLite schema could not be created.",
                status_code=503,
                code="database_error",
            ) from exc
        finally:
            conn.close()
        self._schema_ready = True

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with the local embedding model.

        Args:
            texts: Non-empty chunk strings.

        Returns:
            One normalized vector per input string.

        Raises:
            RAGEngineError: The model cannot be loaded or returns an unexpected shape.
        """
        model = self._embedding_model()
        try:
            with self._model_lock:
                encoded = model.encode(
                    texts,
                    batch_size=_EMBED_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        except RAGEngineError:
            raise
        except Exception as exc:
            logger.exception("embedding failed")
            raise RAGEngineError(
                "The embedding model could not encode the text.",
                status_code=503,
                code="model_unavailable",
            ) from exc
        vectors = encoded.tolist() if hasattr(encoded, "tolist") else [list(row) for row in encoded]
        if len(vectors) != len(texts):
            raise RAGEngineError(
                "The embedding model returned an unexpected number of vectors.",
                code="model_unavailable",
                status_code=503,
            )
        expected = self.settings.embedding_dim
        cleaned: list[list[float]] = []
        for vector in vectors:
            as_floats = [float(value) for value in vector]
            if len(as_floats) != expected:
                raise RAGEngineError(
                    f"The embedding model returned dimension {len(as_floats)}, expected {expected}.",
                    code="config_error",
                )
            cleaned.append(as_floats)
        return cleaned

    def _embedding_model(self) -> Any:
        """Load the sentence-transformers model once.

        Returns:
            The loaded model.

        Raises:
            RAGEngineError: Weights cannot be downloaded or the dimension disagrees with settings.
        """
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                )
            except Exception as exc:
                logger.exception("embedding model load failed")
                raise RAGEngineError(
                    "The embedding model could not be loaded. "
                    "Check EMBEDDING_MODEL and that this machine can reach Hugging Face.",
                    status_code=503,
                    code="model_unavailable",
                ) from exc
            try:
                dimension = int(model.get_sentence_embedding_dimension())
            except Exception as exc:
                logger.exception("embedding dimension lookup failed")
                raise RAGEngineError(
                    "The embedding model did not report its vector dimension.",
                    status_code=503,
                    code="model_unavailable",
                ) from exc
            if dimension != self.settings.embedding_dim:
                raise RAGEngineError(
                    f"EMBEDDING_DIM is {self.settings.embedding_dim}, but {self.settings.embedding_model} "
                    f"produces {dimension}-dimensional vectors.",
                    code="config_error",
                )
            self._model = model
            return model

    def _retrieve(self, question: str, top_k: int) -> list[_RetrievedChunk]:
        """Embed the question and return the nearest stored chunks.

        Args:
            question: Question text.
            top_k: Maximum number of rows.

        Returns:
            Chunks ordered from most similar to least similar.

        Raises:
            RAGEngineError: Embedding or the database query fails.
        """
        embedding = self._embed([question])[0]
        try:
            with self._connection() as conn:
                fetched = conn.execute(
                    """
                    SELECT
                        c.id AS chunk_id,
                        c.document_id,
                        d.filename,
                        c.chunk_index,
                        c.content,
                        c.embedding
                    FROM rag_chunks AS c
                    JOIN rag_documents AS d ON d.id = c.document_id
                    """
                ).fetchall()
                rows = [
                    {
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "filename": row["filename"],
                        "chunk_index": row["chunk_index"],
                        "content": row["content"],
                        "embedding": bytes(row["embedding"]),
                    }
                    for row in fetched
                ]
        except RAGEngineError:
            raise
        except sqlite3.Error as exc:
            logger.exception("vector retrieval failed")
            raise RAGEngineError(
                "Similar passages could not be retrieved.",
                status_code=503,
                code="database_error",
            ) from exc
        return _rank_chunks(rows, embedding, top_k)

    def _generate(self, question: str, chunks: list[_RetrievedChunk]) -> str:
        """Ask the configured chat model to answer from the retrieved passages.

        Args:
            question: User question.
            chunks: Passages that passed the similarity threshold.

        Returns:
            The model's answer text.

        Raises:
            RAGEngineError: The language-model request fails or the response is empty.
        """
        context = _build_context(chunks)
        messages = []
        for message in _PROMPT.format_messages(context=context, question=question):
            content = message.content
            if not isinstance(content, str):
                raise RAGEngineError(
                    "The prompt renderer produced a non-text message.",
                    status_code=502,
                    code="llm_error",
                )
            role = {"system": "system", "human": "user", "ai": "assistant"}.get(message.type)
            if role is None:
                raise RAGEngineError(
                    "The prompt renderer produced an unsupported message role.",
                    status_code=502,
                    code="llm_error",
                )
            messages.append({"role": role, "content": content})

        payload = {
            "model": self.settings.llm_model,
            "temperature": 0.1,
            "max_tokens": 800,
            "messages": messages,
        }
        raw_body = json.dumps(payload).encode("utf-8")
        http_request = urlrequest.Request(
            f"{self.settings.llm_api_base}/chat/completions",
            data=raw_body,
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "enterprise-rag-assistant",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(http_request, timeout=self.settings.llm_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            logger.error("language model HTTP %s: %s", exc.code, detail)
            if exc.code in {401, 403}:
                raise RAGEngineError(
                    "The language model rejected the API key.",
                    status_code=502,
                    code="llm_auth_error",
                ) from exc
            raise RAGEngineError(
                "The language model request failed.",
                status_code=502,
                code="llm_error",
            ) from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            logger.exception("language model request failed")
            raise RAGEngineError(
                "The language model request failed.",
                status_code=502,
                code="llm_error",
            ) from exc

        try:
            parsed = json.loads(raw)
            answer = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            logger.exception("language model returned an unexpected payload")
            raise RAGEngineError(
                "The language model returned an unreadable response.",
                status_code=502,
                code="llm_error",
            ) from exc
        if not isinstance(answer, str) or not answer.strip():
            raise RAGEngineError(
                "The language model returned an empty answer.",
                status_code=502,
                code="llm_error",
            )
        return answer.strip()

    def _resolve_top_k(self, top_k: int | None) -> int:
        """Return a retrieval depth inside the supported range.

        Args:
            top_k: Caller-supplied depth, or ``None`` for the default.

        Returns:
            An integer from 1 to 20.

        Raises:
            RAGEngineError: The override is not an integer in range.
        """
        if top_k is None:
            return self.settings.top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise RAGEngineError(
                "top_k must be an integer from 1 to 20.",
                status_code=400,
                code="bad_request",
            )
        return top_k

    @staticmethod
    def _resolve_limit(limit: int) -> int:
        """Clamp a document-list limit to the API range.

        Args:
            limit: Requested page size.

        Returns:
            An integer from 1 to 200.

        Raises:
            RAGEngineError: The limit is not an integer in range.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise RAGEngineError(
                "limit must be an integer from 1 to 200.",
                status_code=400,
                code="bad_request",
            )
        return limit


def extract_text(filename: str, data: bytes) -> str:
    """Decode a ``.txt``, ``.md``, or ``.pdf`` upload into plain text.

    Args:
        filename: Sanitized filename used to choose a parser.
        data: Raw file bytes.

    Returns:
        Cleaned document text.

    Raises:
        RAGEngineError: The file is empty, not UTF-8, or not a readable PDF.
    """
    if not data:
        raise RAGEngineError("The uploaded file is empty.", status_code=400, code="invalid_document")
    extension = filename[filename.rfind("."):].lower() if "." in filename else ""
    if extension == ".pdf":
        text = _extract_pdf(data)
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RAGEngineError(
                "Text files must be UTF-8.",
                status_code=400,
                code="invalid_document",
            ) from exc
    cleaned = _clean_text(text)
    if not cleaned:
        raise RAGEngineError(
            "The document has no extractable text. Scanned PDFs need OCR, which this service does not run.",
            status_code=400,
            code="invalid_document",
        )
    return cleaned


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split document text into overlapping character chunks.

    Args:
        text: Cleaned document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters repeated between adjacent chunks.

    Returns:
        Non-empty chunks in document order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [part.strip() for part in splitter.split_text(text) if part.strip()]


def _extract_pdf(data: bytes) -> str:
    """Extract text from a digital PDF.

    Args:
        data: PDF bytes.

    Returns:
        Text joined across pages.

    Raises:
        RAGEngineError: The bytes are not a readable PDF or pypdf is missing.
    """
    if not data.startswith(b"%PDF"):
        raise RAGEngineError("The file is not a PDF.", status_code=400, code="invalid_document")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RAGEngineError(
            "PDF support is unavailable because pypdf is not installed.",
            status_code=503,
            code="dependency_missing",
        ) from exc
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        logger.info("pdf parse failed: %s", exc)
        raise RAGEngineError(
            "The PDF could not be read.",
            status_code=400,
            code="invalid_document",
        ) from exc
    return "\n\n".join(pages)


def _clean_text(text: str) -> str:
    """Normalize newlines and drop null bytes.

    Args:
        text: Raw extracted text.

    Returns:
        Stripped text using Unix newlines.
    """
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    """Collapse whitespace and shorten a passage for an API response.

    Args:
        text: Full chunk text.
        limit: Maximum characters, including the ellipsis.

    Returns:
        A single-line excerpt.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _build_context(chunks: list[_RetrievedChunk], limit: int = _CONTEXT_CHAR_LIMIT) -> str:
    """Join passages until the character budget is spent.

    Args:
        chunks: Retrieved passages in rank order.
        limit: Maximum characters of context sent to the language model.

    Returns:
        A labeled context block.
    """
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        block = f"[{chunk.filename} chunk {chunk.chunk_index}]\n{chunk.content.strip()}"
        if parts and used + len(block) > limit:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _extractive_answer(question: str, chunks: list[_RetrievedChunk]) -> str:
    """Build a retrieval-only answer when no language-model key is configured.

    Args:
        question: User question, included so the response stands alone.
        chunks: Passages that passed the similarity threshold.

    Returns:
        A plain-text answer made of the retrieved passages.
    """
    lines = [
        "No language model API key is configured, so this answer quotes the closest matching passages.",
        f"Question: {question}",
        "",
    ]
    for chunk in chunks:
        lines.append(f"[{chunk.filename} chunk {chunk.chunk_index}]")
        lines.append(chunk.content.strip())
        lines.append("")
    return "\n".join(lines).strip()


def _embedding_device() -> str:
    """Read and validate ``EMBEDDING_DEVICE``.

    Returns:
        ``cpu``, ``cuda``, or ``mps``.

    Raises:
        RAGEngineError: The variable names an unsupported device.
    """
    device = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower() or "cpu"
    if device not in {"cpu", "cuda", "mps"}:
        raise RAGEngineError(
            "EMBEDDING_DEVICE must be cpu, cuda, or mps.",
            code="config_error",
        )
    return device


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read an integer environment variable inside an inclusive range.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.
        minimum: Lowest accepted value.
        maximum: Highest accepted value.

    Returns:
        The parsed integer.

    Raises:
        RAGEngineError: The value is not an integer in range.
    """
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RAGEngineError(f"{name} must be an integer.", code="config_error") from exc
    if not minimum <= value <= maximum:
        raise RAGEngineError(
            f"{name} must be between {minimum} and {maximum}.",
            code="config_error",
        )
    return value


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    """Read a float environment variable inside an inclusive range.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.
        minimum: Lowest accepted value.
        maximum: Highest accepted value.

    Returns:
        The parsed float.

    Raises:
        RAGEngineError: The value is not a number in range.
    """
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RAGEngineError(f"{name} must be a number.", code="config_error") from exc
    if not minimum <= value <= maximum:
        raise RAGEngineError(
            f"{name} must be between {minimum} and {maximum}.",
            code="config_error",
        )
    return value


def _resolve_sqlite_path() -> Path:
    """Resolve ``SQLITE_PATH`` to an absolute file path.

    A relative path is resolved from the project directory, not the shell's
    current directory, so the API and a manual run use the same file.

    Returns:
        Absolute path of the SQLite database file.
    """
    raw = os.getenv("SQLITE_PATH", "rag.sqlite").strip() or "rag.sqlite"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _pack_vector(vector: list[float]) -> bytes:
    """Pack a normalized embedding into a float32 blob.

    Args:
        vector: Embedding values.

    Returns:
        Little-endian float32 bytes.
    """
    return np.asarray(vector, dtype=np.float32).tobytes()


def _rank_chunks(rows: list[dict[str, Any]], query: list[float], top_k: int) -> list[_RetrievedChunk]:
    """Rank stored chunks by cosine similarity to a normalized query.

    Stored embeddings are normalized at insert time, so the dot product is
    the cosine similarity.

    Args:
        rows: Chunk dictionaries that include an ``embedding`` blob.
        query: Normalized query embedding.
        top_k: Maximum number of rows to return.

    Returns:
        The closest chunks, highest score first.

    Raises:
        RAGEngineError: A stored vector does not match the query dimension.
    """
    if not rows:
        return []
    query_vec = np.asarray(query, dtype=np.float32)
    matrix = np.empty((len(rows), query_vec.shape[0]), dtype=np.float32)
    for index, row in enumerate(rows):
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.shape != query_vec.shape:
            raise RAGEngineError(
                "A stored embedding does not match EMBEDDING_DIM. "
                "Delete the SQLite file and upload the documents again.",
                code="config_error",
            )
        matrix[index] = vector
    scores = matrix @ query_vec
    count = min(top_k, len(scores))
    chosen = np.argpartition(scores, -count)[-count:]
    order = chosen[np.argsort(scores[chosen])[::-1]]
    matches: list[_RetrievedChunk] = []
    for index in order:
        row = rows[int(index)]
        score = max(-1.0, min(1.0, float(scores[int(index)])))
        matches.append(
            _RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                filename=str(row["filename"]),
                chunk_index=int(row["chunk_index"]),
                content=str(row["content"]),
                score=score,
            )
        )
    return matches

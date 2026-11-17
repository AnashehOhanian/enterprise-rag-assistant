"""Flask REST API for the enterprise RAG assistant.

Endpoints:
    GET  /
    GET  /api
    GET  /health
    POST /upload
    POST /query
    GET  /documents
    DELETE /documents/<document_id>
"""

from __future__ import annotations

import atexit
import logging
import os
import secrets
import threading
import uuid
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from rag_engine import RAGEngine, RAGEngineError, extract_text

load_dotenv()

logger = logging.getLogger("enterprise_rag.api")

_ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
_MAX_QUESTION_CHARS = 4000
_MAX_TOP_K = 20

_engine: RAGEngine | None = None
_engine_lock = threading.Lock()


class AppError(Exception):
    """A request error with a stable client-facing code and HTTP status."""

    def __init__(self, message: str, status_code: int = 400, code: str = "bad_request") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def create_app() -> Flask:
    """Build the Flask application and register API routes.

    Returns:
        A configured Flask application. The database pool and embedding
        model stay unloaded until a route needs them.
    """
    _configure_logging()
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config["MAX_CONTENT_LENGTH"] = _require_positive_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    app.config["JSON_SORT_KEYS"] = False

    if not os.getenv("APP_API_KEY", "").strip():
        app.logger.warning("APP_API_KEY is not set. Document and query routes are unauthenticated.")

    @app.get("/")
    def index() -> Any:
        """Render the chat page."""
        return render_template("index.html")

    @app.get("/api")
    def api_index() -> Any:
        """Return the service name and the routes a client can call."""
        return jsonify(
            {
                "service": "enterprise-rag-assistant",
                "endpoints": {
                    "ui": "/",
                    "health": "/health",
                    "upload": "/upload",
                    "query": "/query",
                    "documents": "/documents",
                },
            }
        )

    @app.get("/health")
    def health() -> Any:
        """Report whether configuration and the SQLite file are usable."""
        try:
            engine = get_engine()
            database_ok = engine.check_database()
        except RAGEngineError as exc:
            return _error_body(exc.code, str(exc), {"status": "degraded", "database": False}), exc.status_code
        status = "ok" if database_ok else "degraded"
        body = {
            "status": status,
            "database": database_ok,
            "embedding_model": engine.settings.embedding_model,
            "llm_configured": engine.llm_configured,
        }
        return jsonify(body), 200 if database_ok else 503

    @app.post("/upload")
    def upload() -> Any:
        """Ingest a text document, Markdown file, PDF, or raw JSON text."""
        _require_api_key()
        filename, text, content_type = _read_upload()
        result = get_engine().ingest(filename, text, content_type)
        logger.info(
            "ingested document_id=%s chunks=%s filename=%s",
            result.document_id,
            result.chunk_count,
            result.filename,
        )
        return jsonify(result.to_dict()), 201

    @app.post("/query")
    def query() -> Any:
        """Answer a question from indexed document chunks."""
        _require_api_key()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise AppError("Request body must be a JSON object.")
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise AppError("JSON body must include a non-empty 'question' string.")
        question = question.strip()
        if len(question) > _MAX_QUESTION_CHARS:
            raise AppError(f"question must be {_MAX_QUESTION_CHARS} characters or fewer.")
        top_k = _optional_top_k(payload.get("top_k", None))
        result = get_engine().query(question, top_k)
        logger.info(
            "query chars=%s top_k=%s mode=%s sources=%s",
            len(question),
            top_k if top_k is not None else "default",
            result.mode,
            len(result.sources),
        )
        return jsonify(result.to_dict())

    @app.get("/documents")
    def list_documents() -> Any:
        """List recently ingested documents."""
        _require_api_key()
        limit = _optional_limit(request.args.get("limit"))
        documents = get_engine().list_documents(limit)
        return jsonify({"documents": documents})

    @app.delete("/documents/<document_id>")
    def delete_document(document_id: str) -> Any:
        """Delete one document and its chunks."""
        _require_api_key()
        try:
            parsed = uuid.UUID(document_id)
        except ValueError as exc:
            raise AppError("document_id must be a UUID.") from exc
        get_engine().delete_document(parsed)
        return jsonify({"deleted": True, "document_id": str(parsed)})

    @app.errorhandler(AppError)
    def handle_app_error(exc: AppError) -> Any:
        return _error_body(exc.code, str(exc)), exc.status_code

    @app.errorhandler(RAGEngineError)
    def handle_rag_error(exc: RAGEngineError) -> Any:
        if exc.status_code >= 500:
            log = logger.warning if exc.code == "config_error" else logger.error
            log("rag error code=%s message=%s", exc.code, exc)
        return _error_body(exc.code, str(exc)), exc.status_code

    @app.errorhandler(HTTPException)
    def handle_http(exc: HTTPException) -> Any:
        descriptions = {
            404: "Resource not found.",
            405: "Method not allowed.",
            413: "Upload exceeds MAX_UPLOAD_BYTES.",
        }
        codes = {
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
        }
        status = exc.code or 500
        return _error_body(codes.get(status, "http_error"), descriptions.get(status, "Request failed.")), status

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception) -> Any:
        if isinstance(exc, HTTPException):
            return handle_http(exc)
        logger.exception("unhandled API error")
        return _error_body("internal_error", "Internal server error."), 500

    atexit.register(_shutdown_engine)
    return app


def get_engine() -> RAGEngine:
    """Return the process-wide RAG engine, creating it on first use.

    Returns:
        The shared engine instance.

    Raises:
        RAGEngineError: Configuration is missing or invalid.
    """
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = RAGEngine.from_env()
        return _engine


def _shutdown_engine() -> None:
    """Close the database pool when the process exits."""
    global _engine
    with _engine_lock:
        engine = _engine
        _engine = None
    if engine is not None:
        engine.close()


def _configure_logging() -> None:
    """Send logs to stderr once, without disturbing a host process logger."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _require_positive_int(name: str, default: int) -> int:
    """Read a positive integer environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        The parsed integer.

    Raises:
        RuntimeError: The variable is set to a non-positive integer.
    """
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def _require_api_key() -> None:
    """Reject the request when APP_API_KEY is set and the caller omits it.

    Health checks stay open so orchestrators can probe them. When the
    variable is empty, local development proceeds without a key.

    Raises:
        AppError: The configured key is missing from the request.
    """
    expected = os.getenv("APP_API_KEY", "").strip()
    if not expected:
        return
    provided = request.headers.get("X-API-Key", "").strip()
    if not provided:
        authorization = request.headers.get("Authorization", "").strip()
        if authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise AppError("Unauthorized.", 401, "unauthorized")


def _read_upload() -> tuple[str, str, str | None]:
    """Read either a multipart file or a JSON text body.

    Returns:
        Filename, extracted text, and optional MIME type.

    Raises:
        AppError: The body is missing, empty, or an unsupported file type.
        RAGEngineError: A file could not be decoded.
    """
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise AppError("Request body must be a JSON object.")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise AppError("JSON body must include a non-empty 'text' string.")
        filename = payload.get("filename", "upload.txt")
        if not isinstance(filename, str) or not filename.strip():
            raise AppError("'filename' must be a non-empty string.")
        return _safe_filename(filename, fallback="upload.txt"), text.strip(), "text/plain"

    storage = request.files.get("file")
    if storage is None or not storage.filename:
        raise AppError(
            "Provide a multipart file field named 'file', or a JSON body with 'text'.",
            415,
            "unsupported_media",
        )
    extension = _extension(storage.filename)
    if extension not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise AppError(f"Unsupported file type. Allowed extensions: {allowed}.")
    filename = _safe_filename(storage.filename, fallback=f"document{extension}")
    data = storage.read()
    if not data:
        raise AppError("The uploaded file is empty.")
    text = extract_text(filename, data)
    content_type = storage.mimetype or None
    return filename, text, content_type


def _safe_filename(name: str, *, fallback: str) -> str:
    """Return a basename that is safe to store and display.

    Args:
        name: Client-supplied filename.
        fallback: Name used when sanitizing removes every character.

    Returns:
        A sanitized filename of at most 255 characters.
    """
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    extension = _extension(base)
    cleaned = secure_filename(base)
    if not cleaned:
        cleaned = fallback
    elif extension and not cleaned.lower().endswith(extension):
        stem = cleaned[: max(1, 255 - len(extension))]
        cleaned = f"{stem}{extension}"
    return cleaned[:255]


def _extension(filename: str) -> str:
    """Return the lowercase extension, including the leading dot.

    The extension is taken from the last path segment. Names such as
    ``notes.txt.exe`` keep ``.exe``, so the upload allow-list sees the
    real suffix before the stored name is sanitized.
    """
    base = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    dot = base.rfind(".")
    if dot <= 0:
        return ""
    return base[dot:].lower()


def _optional_top_k(value: Any) -> int | None:
    """Validate an optional retrieval depth from a JSON body.

    Args:
        value: Raw ``top_k`` value. ``None`` selects the configured default.

    Returns:
        The validated depth, or ``None`` when the client did not send one.

    Raises:
        AppError: The value is not an integer in the allowed range.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_TOP_K:
        raise AppError(f"top_k must be an integer from 1 to {_MAX_TOP_K}.")
    return value


def _optional_limit(value: str | None) -> int:
    """Validate the document list page size.

    Args:
        value: Raw query-string value.

    Returns:
        A limit from 1 to 200. Missing values default to 50.

    Raises:
        AppError: The query string is not an integer in range.
    """
    if value is None or value == "":
        return 50
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AppError("limit must be an integer from 1 to 200.") from exc
    if not 1 <= parsed <= 200:
        raise AppError("limit must be an integer from 1 to 200.")
    return parsed


def _error_body(code: str, message: str, extra: dict[str, Any] | None = None) -> Any:
    """Build a JSON error response body.

    Args:
        code: Stable machine-readable error code.
        message: Client-facing explanation.
        extra: Optional top-level fields, used by the health route.

    Returns:
        A Flask JSON response without a status code. Callers attach the status.
    """
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if extra:
        body = {**extra, **body}
    return jsonify(body)


app = create_app()


def main() -> None:
    """Run the development server."""
    try:
        port = int(os.getenv("PORT", "5000"))
    except ValueError as exc:
        raise SystemExit("PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be between 1 and 65535.")
    host = os.getenv("HOST", "127.0.0.1")
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

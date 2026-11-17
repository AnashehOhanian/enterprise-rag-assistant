"""Streamlit chat client for the Flask RAG API.

Run the API first, then start this interface:

    python app.py
    streamlit run app_ui.py
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib import error, request

import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:5000"
REQUEST_TIMEOUT_SECONDS = 300


class ApiError(Exception):
    """A response the Flask API rejected or could not complete."""


def main() -> None:
    """Render the document sidebar and the question chat."""
    st.set_page_config(page_title="Enterprise RAG Assistant", page_icon=":material/forum:", layout="wide")
    _init_state()

    with st.sidebar:
        st.header("Documents")
        api_base = st.text_input("API base URL", value=DEFAULT_API_BASE)
        api_key = st.text_input("API key", type="password", help="Required only when APP_API_KEY is set on the API.")
        top_k = st.slider("Passages to retrieve", min_value=1, max_value=20, value=4)
        _show_health(api_base)
        _upload_form(api_base, api_key)
        _show_documents(api_base, api_key)

    st.title("Enterprise RAG Assistant")
    st.caption("Answers come from the documents indexed by the Flask API.")

    for message in st.session_state.messages:
        _render_message(message)

    question = st.chat_input("Ask a question about your documents")
    if question:
        _ask(api_base, api_key, question, top_k)


def _init_state() -> None:
    """Create the chat history the first time the page loads."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _show_health(api_base: str) -> None:
    """Show whether the Flask process accepts a health check."""
    try:
        status, body = _request("GET", api_base, "/health")
    except ApiError as exc:
        st.error(f"API unavailable: {exc}")
        return
    if status == 200 and body.get("status") == "ok":
        st.success("API connected")
    else:
        message = body.get("error", {}).get("message", "The API is degraded.")
        st.warning(message)


def _upload_form(api_base: str, api_key: str) -> None:
    """Upload one PDF or text file through the sidebar."""
    uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if st.button("Index document", type="primary", use_container_width=True, disabled=uploaded is None):
        try:
            with st.spinner("Indexing document…"):
                _status, body = _upload_file(api_base, api_key, uploaded.name, uploaded.getvalue())
        except ApiError as exc:
            st.error(str(exc))
            return
        st.success(f"Indexed {body.get('filename', uploaded.name)} in {body.get('chunk_count', 0)} chunks.")
        st.rerun()


def _show_documents(api_base: str, api_key: str) -> None:
    """List documents currently stored by the API."""
    st.subheader("Indexed")
    try:
        _status, body = _request("GET", api_base, "/documents", api_key=api_key)
    except ApiError as exc:
        st.caption(str(exc))
        return
    documents = body.get("documents", [])
    if not documents:
        st.caption("No documents yet.")
        return
    for document in documents:
        st.write(f"{document.get('filename', 'document')} · {document.get('chunk_count', 0)} chunks")


def _ask(api_base: str, api_key: str, question: str, top_k: int) -> None:
    """Send a question and append both turns to the chat."""
    st.session_state.messages.append({"role": "user", "content": question})
    _render_message(st.session_state.messages[-1])
    try:
        with st.spinner("Searching indexed documents…"):
            _status, body = _request(
                "POST",
                api_base,
                "/query",
                api_key=api_key,
                payload={"question": question, "top_k": top_k},
            )
    except ApiError as exc:
        answer = {"role": "assistant", "content": str(exc), "sources": [], "error": True}
    else:
        answer = {
            "role": "assistant",
            "content": body.get("answer", "No answer was returned."),
            "sources": body.get("sources", []),
            "mode": body.get("mode"),
            "error": False,
        }
    st.session_state.messages.append(answer)
    _render_message(answer)


def _render_message(message: dict[str, Any]) -> None:
    """Draw one chat turn, including source passages for an answer."""
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        mode = message.get("mode")
        sources = message.get("sources") or []
        if sources:
            label = f"Sources · {mode}" if mode else "Sources"
            with st.expander(label, expanded=True):
                for source in sources:
                    filename = source.get("filename", "document")
                    chunk_index = source.get("chunk_index", 0)
                    score = source.get("score", "")
                    st.markdown(f"**{filename}** · chunk {chunk_index} · score {score}")
                    st.write(source.get("excerpt", ""))


def _upload_file(api_base: str, api_key: str, filename: str, data: bytes) -> tuple[int, dict[str, Any]]:
    """Post a multipart file to ``/upload``.

    Args:
        api_base: Flask origin, without a trailing path.
        api_key: Optional ``APP_API_KEY`` value.
        filename: Original file name.
        data: Raw file bytes.

    Returns:
        HTTP status and the parsed JSON body.
    """
    boundary = uuid.uuid4().hex
    disposition = f'Content-Disposition: form-data; name="file"; filename="{filename}"'
    preamble = f"--{boundary}\r\n{disposition}\r\nContent-Type: application/octet-stream\r\n\r\n".encode("utf-8")
    ending = f"\r\n--{boundary}--\r\n".encode("utf-8")
    payload = preamble + data + ending
    headers = _auth_headers(api_key)
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    return _open_request(api_base, "/upload", payload, headers, "POST")


def _request(
    method: str,
    api_base: str,
    path: str,
    api_key: str = "",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Call a JSON endpoint on the Flask API.

    Args:
        method: HTTP method.
        api_base: Flask origin, without a trailing path.
        path: Absolute path such as ``/query``.
        api_key: Optional ``APP_API_KEY`` value.
        payload: JSON body for POST requests.

    Returns:
        HTTP status and the parsed JSON body.
    """
    headers = _auth_headers(api_key)
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    return _open_request(api_base, path, data, headers, method)


def _open_request(
    api_base: str,
    path: str,
    data: bytes | None,
    headers: dict[str, str],
    method: str,
) -> tuple[int, dict[str, Any]]:
    """Perform one HTTP call and turn failures into ``ApiError``.

    Args:
        api_base: Flask origin, without a trailing path.
        path: Absolute path.
        data: Optional request body.
        headers: Request headers.
        method: HTTP method.

    Returns:
        HTTP status and the parsed JSON body.

    Raises:
        ApiError: The API returned an error or could not be reached.
    """
    url = api_base.rstrip("/") + path
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, _decode_json(response.read())
    except error.HTTPError as exc:
        message = _error_message(exc.read())
        raise ApiError(message) from exc
    except error.URLError as exc:
        raise ApiError(f"Could not reach the API at {api_base}.") from exc
    except TimeoutError as exc:
        raise ApiError("The API took too long to respond.") from exc


def _auth_headers(api_key: str) -> dict[str, str]:
    """Return the API key header when one was entered."""
    if api_key.strip():
        return {"X-API-Key": api_key.strip()}
    return {}


def _decode_json(raw: bytes) -> dict[str, Any]:
    """Parse a response body as a JSON object."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError("The API returned a response that was not JSON.") from exc
    if not isinstance(parsed, dict):
        raise ApiError("The API returned an unexpected response.")
    return parsed


def _error_message(raw: bytes) -> str:
    """Read the Flask error message, falling back to the raw body."""
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "The API request failed."
    if isinstance(parsed, dict):
        error_body = parsed.get("error")
        if isinstance(error_body, dict) and error_body.get("message"):
            return str(error_body["message"])
    return "The API request failed."


if __name__ == "__main__":
    main()

# rag.py

"""A small, self-contained RAG (retrieval-augmented generation) module.

Documents in the ``knowledge/`` folder are read, split into chunks, and
indexed with plain TF-IDF.  When a question is routed to the knowledge
base, the most similar chunks are retrieved and given to the LLM as
context.

No vector database or embedding model is required.
"""

import csv
import hashlib
import json
import math
import re
import shutil

from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# RAG CONFIGURATION
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CACHE_DIR = BASE_DIR / ".rag_cache"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 4
MIN_SCORE = 0.10

OCR_ENABLED = True
OCR_DPI = 150

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".htm",
}

# Very common words add no useful meaning to a search.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
}


# The index is built once and reused until a document is added or removed.
_INDEX_CACHE = None


# ---------------------------------------------------------------------------
# READ DOCUMENT
# ---------------------------------------------------------------------------

def extract_text(file_path: str, use_ocr: bool = True) -> str:
    """Read complete text from a supported document."""

    path = Path(file_path).expanduser()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        return _extract_pdf(path, use_ocr=use_ocr)

    if extension == ".docx":
        return _extract_docx(path)

    if extension == ".csv":
        return _extract_csv(path)

    if extension == ".json":
        return _extract_json(path)

    # TXT, MD, HTML
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _extract_pdf(path: Path, use_ocr: bool = True) -> str:
    """Read a PDF, using OCR only for pages that have no text layer."""

    from pypdf import PdfReader

    reader = PdfReader(str(path))

    pages = []

    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append([number, text])

    missing = [number for number, text in pages if not text.strip()]

    if missing and OCR_ENABLED:
        ocr_pages = _ocr_pages(path, missing, run_ocr=use_ocr)

        for page in pages:
            if not page[1].strip() and page[0] in ocr_pages:
                page[1] = ocr_pages[page[0]]

    text_parts = []

    for number, text in pages:
        if text.strip():
            text_parts.append(f"[Page {number}]\n{text.strip()}")

    return "\n\n".join(text_parts)


def _extract_docx(path: Path) -> str:
    from docx import Document

    document = Document(str(path))

    text_parts = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells
            )

            if row_text.strip():
                text_parts.append(row_text)

    return "\n\n".join(text_parts)


def _extract_csv(path: Path) -> str:
    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            rows.append(" | ".join(row))

    return "\n".join(rows)


def _extract_json(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig") as file:
        data = json.load(file)

    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# OCR FOR SCANNED PDF PAGES
# ---------------------------------------------------------------------------

def _ocr_cache_file(path: Path) -> Path:
    """Return a cache file that changes when the PDF changes."""

    stat = path.stat()

    key = hashlib.sha1(
        f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode()
    ).hexdigest()

    return CACHE_DIR / f"{key}.json"


def _ocr_pages(path: Path, page_numbers: list, run_ocr: bool = True) -> dict:
    """Return cached OCR text, optionally running OCR for missing pages."""

    cache_file = _ocr_cache_file(path)

    cached = {}

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached = {
                int(number): text
                for number, text in data.get("pages", {}).items()
            }
        except (OSError, ValueError, json.JSONDecodeError):
            cached = {}

    missing = [number for number in page_numbers if number not in cached]

    if not missing or not run_ocr:
        return cached

    try:
        import pymupdf
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        # OCR packages are optional; text extraction still works without them.
        return cached

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    engine = RapidOCR()

    with pymupdf.open(str(path)) as pdf:
        for number in missing:
            page = pdf[number - 1]
            image = page.get_pixmap(dpi=OCR_DPI).tobytes("png")

            result, _ = engine(image)

            cached[number] = (
                "\n".join(line[1] for line in result)
                if result
                else ""
            )

    cache_file.write_text(
        json.dumps(
            {
                "path": str(path),
                "pages": {str(n): text for n, text in cached.items()},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return cached


# ---------------------------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list:
    """Split a document into overlapping chunks."""

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------------------------

def _singular(word: str) -> str:
    """A tiny plural normaliser, e.g. "laws" -> "law"."""

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


def tokenize(text: str) -> list:
    """Turn text into simple lowercase keywords."""

    words = re.findall(r"[a-z0-9]+", text.lower())

    return [
        _singular(word)
        for word in words
        if len(word) > 1 and word not in STOPWORDS
    ]


# ---------------------------------------------------------------------------
# BUILD INDEX
# ---------------------------------------------------------------------------

def build_index(force: bool = False) -> dict:
    """Build the TF-IDF index for every file in knowledge/."""

    global _INDEX_CACHE

    if _INDEX_CACHE is None or force:
        _INDEX_CACHE = _create_index()

    return _INDEX_CACHE


def refresh_index() -> dict:
    """Rebuild the index after documents have been added or removed."""

    return build_index(force=True)


def _create_index() -> dict:
    documents = []
    chunks = []

    if KNOWLEDGE_DIR.exists():
        paths = sorted(
            path
            for path in KNOWLEDGE_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        paths = []

    for path in paths:
        try:
            text = extract_text(str(path), use_ocr=False)
        except Exception as error:
            print(f"Could not read '{path.name}': {error}")
            continue

        pieces = chunk_text(text)

        documents.append({
            "filename": path.name,
            "source": str(path),
            "total_characters": len(text),
            "total_chunks": len(pieces),
        })

        for number, piece in enumerate(pieces, start=1):
            tokens = tokenize(piece)

            if not tokens:
                continue

            chunks.append({
                "content": piece,
                "filename": path.name,
                "source": str(path),
                "chunk_number": number,
                "total_chunks": len(pieces),
                "tokens": tokens,
            })

    # Document frequency: how many chunks contain each word.
    document_frequency = Counter()

    for chunk in chunks:
        document_frequency.update(set(chunk["tokens"]))

    total_chunks = len(chunks)

    idf = {
        token: math.log((1 + total_chunks) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }

    # Pre-compute a normalised TF-IDF vector for every chunk.
    for chunk in chunks:
        counts = Counter(chunk["tokens"])
        length = len(chunk["tokens"])

        weights = {
            token: (count / length) * idf[token]
            for token, count in counts.items()
        }

        chunk["weights"] = weights
        chunk["norm"] = math.sqrt(
            sum(weight * weight for weight in weights.values())
        )

        del chunk["tokens"]

    sources = []

    for document in documents:
        if document["filename"] not in sources:
            sources.append(document["filename"])

    return {
        "documents": documents,
        "chunks": chunks,
        "idf": idf,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# RETRIEVE
# ---------------------------------------------------------------------------

def _make_result(chunk: dict, score: float = 0.0, coverage: float = 0.0) -> dict:
    """Turn an indexed chunk into a retrieval result."""

    return {
        "content": chunk["content"],
        "metadata": {
            "filename": chunk["filename"],
            "source": chunk["source"],
            "chunk_number": chunk["chunk_number"],
            "total_chunks": chunk["total_chunks"],
        },
        "score": score,
        "coverage": coverage,
    }


def retrieve(
    question: str,
    index: dict = None,
    top_k: int = TOP_K,
    sources: list = None,
    min_score: float = None,
) -> list:
    """Return the chunks most similar to the question."""

    if index is None:
        index = build_index()

    if min_score is None:
        min_score = MIN_SCORE

    tokens = tokenize(question)

    if not tokens or not index["chunks"]:
        return []

    counts = Counter(tokens)
    length = len(tokens)
    idf = index["idf"]

    query_terms = set(tokens)
    query_term_count = len(query_terms)

    question_weights = {
        token: (count / length) * idf[token]
        for token, count in counts.items()
        if token in idf
    }

    question_norm = math.sqrt(
        sum(weight * weight for weight in question_weights.values())
    )

    allowed = (
        {source.casefold() for source in sources}
        if sources
        else None
    )

    if question_norm == 0:
        # The question has no searchable words (for example "summarize
        # scanned.pdf"). Return the first chunks of a named file so it can
        # still be summarised.
        if not allowed:
            return []

        return [
            _make_result(chunk)
            for chunk in index["chunks"]
            if chunk["filename"].casefold() in allowed
        ][:top_k]

    results = []

    for chunk in index["chunks"]:
        if allowed and chunk["filename"].casefold() not in allowed:
            continue

        dot = sum(
            weight * chunk["weights"].get(token, 0.0)
            for token, weight in question_weights.items()
        )
 
        score = (
            dot / (question_norm * chunk["norm"])
            if chunk["norm"]
            else 0.0
        )

        if score < min_score:
            continue

        matched_terms = sum(
            1
            for token in query_terms
            if token in chunk["weights"]
        )

        results.append(
            _make_result(
                chunk,
                score=score,
                coverage=matched_terms / query_term_count,
            )
        )

    results.sort(key=lambda result: result["score"], reverse=True)

    return results[:top_k]


def retrieve_context(question: str, top_k: int = TOP_K) -> str:
    """Return retrieved chunks as one readable context string."""

    return format_context(retrieve(question, top_k=top_k))


def format_context(results: list) -> str:
    """Convert retrieved chunks into readable context."""

    context_parts = []

    for result in results:
        metadata = result.get("metadata", {})

        context_parts.append(
            f"Source: {metadata.get('filename', 'Unknown')}\n"
            f"Chunk: {metadata.get('chunk_number', 'Unknown')}\n"
            f"Content:\n{result.get('content', '')}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_sources(results: list) -> list:
    """Get the unique file names from retrieved chunks."""

    sources = []

    for result in results:
        filename = result.get("metadata", {}).get("filename")

        if filename and filename not in sources:
            sources.append(filename)

    return sources


def find_mentioned_sources(
    question: str,
    sources: list = None,
) -> list:
    """Find indexed file names that appear in the question."""

    if sources is None:
        sources = build_index()["sources"]

    lowered = question.casefold()

    return [
        source
        for source in sources
        if source.casefold() in lowered
    ]


# ---------------------------------------------------------------------------
# MANAGE DOCUMENTS
# ---------------------------------------------------------------------------

def ingest_document(file_path: str) -> dict:
    """Copy a document into knowledge/ and add it to the index."""

    source = Path(file_path).expanduser()

    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {source.suffix.lower()}"
        )

    text = extract_text(str(source))

    if not text.strip():
        raise ValueError("No readable text found in document.")

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    resolved_source = source.resolve()

    try:
        resolved_source.relative_to(KNOWLEDGE_DIR.resolve())
        inside_knowledge_dir = True
    except ValueError:
        inside_knowledge_dir = False

    if inside_knowledge_dir:
        destination = source
        same_file = True
    else:
        destination = KNOWLEDGE_DIR / source.name
        same_file = resolved_source == destination.resolve()

    replaced = destination.exists() and not same_file

    if not same_file:
        shutil.copy2(source, destination)

    refresh_index()

    chunks = chunk_text(text)

    message = f"Successfully indexed {destination.name}."

    if replaced:
        message += " It replaced the existing copy."

    return {
        "success": True,
        "filename": destination.name,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "message": message,
    }


def get_indexed_documents() -> list:
    """List the documents in the knowledge folder."""

    index = build_index()

    return [
        {
            "filename": document["filename"],
            "total_chunks": document["total_chunks"],
            "source": document["source"],
        }
        for document in index["documents"]
    ]


def remove_document(filename: str) -> dict:
    """Remove a document from the knowledge folder."""

    name = Path(filename).name

    matches = (
        [
            path
            for path in KNOWLEDGE_DIR.rglob("*")
            if path.is_file() and path.name == name
        ]
        if KNOWLEDGE_DIR.exists()
        else []
    )

    if not matches:
        return {
            "success": False,
            "message": f"Document '{name}' was not found.",
        }

    for path in matches:
        path.unlink()

    refresh_index()

    return {
        "success": True,
        "message": f"Removed '{name}' successfully.",
    }


def ensure_ocr(file_path: str) -> str:
    """Return the text of a document, with OCR for scanned PDF pages."""

    return extract_text(file_path)

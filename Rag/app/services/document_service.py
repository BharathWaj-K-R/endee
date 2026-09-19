"""Document service for file validation and ingestion."""

import os
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, UploadFile

from rag.ingest import ingest_document
from utils.config import get_settings


def _validate_file_type(filename: str) -> str:
    """Validate the uploaded file extension and return it."""
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported.")
    return extension


def _ingest_and_cleanup(temp_path: Path) -> None:
    """Run ingestion and remove the temporary file afterward."""
    try:
        ingest_document(temp_path)
    finally:
        if temp_path.exists():
            os.remove(temp_path)


async def process_uploaded_file_background(
    background_tasks: BackgroundTasks, file: UploadFile
) -> dict:
    """Save an uploaded file and ingest it in the background."""
    if not file.filename:
        raise ValueError("Uploaded file must have a filename.")

    _validate_file_type(file.filename)
    settings = get_settings()
    temp_name = f"{uuid4().hex}_{file.filename}"
    temp_path = Path(settings.upload_dir) / temp_name

    content = await file.read()
    if not content:
        raise ValueError("Uploaded file is empty.")

    temp_path.write_bytes(content)
    background_tasks.add_task(_ingest_and_cleanup, temp_path)

    return {
        "message": "Document uploaded. Indexing in background.",
        "filename": file.filename,
        "queued": True,
    }

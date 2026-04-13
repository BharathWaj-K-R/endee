"""Upload routes for document ingestion."""

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.services.document_service import process_uploaded_file_background

router = APIRouter(tags=["documents"])


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> dict:
    """Accept a document upload, ingest it into the vector store, and return a summary."""
    try:
        return await process_uploaded_file_background(background_tasks, file)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process upload: {error}",
        ) from error

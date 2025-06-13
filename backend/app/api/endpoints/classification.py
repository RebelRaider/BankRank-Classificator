from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional, List
from ...core.database import get_db
from ...services.classification_service import ClassificationService
from ...models.schemas import ClassificationResponse, FileUploadResponse
from ...utils.exceptions import ClassificationError, FileProcessingError

router = APIRouter()
classification_service = ClassificationService()

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload file for classification"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process upload
        document = classification_service.process_file_upload(
            file_content=file_content,
            filename=file.filename,
            db=db
        )
        
        return FileUploadResponse(
            document_id=document.id,
            filename=document.filename,
            file_size=document.file_size,
            status="uploaded"
        )
        
    except (FileProcessingError, ClassificationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/classify/{document_id}", response_model=ClassificationResponse)
async def classify_document(
    document_id: int,
    include_toxicity: bool = True,
    include_rating: bool = True,
    db: Session = Depends(get_db)
):
    """Classify uploaded document"""
    try:
        result = classification_service.classify_document(
            document_id=document_id,
            db=db,
            include_toxicity=include_toxicity,
            include_rating=include_rating
        )
        return result
        
    except ClassificationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/classify-text")
async def classify_text_direct(
    text: str = Form(...),
    include_toxicity: bool = Form(True),
    include_rating: bool = Form(True)
):
    """Classify text directly without uploading file"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = classification_service.classify_text_direct(
            text=text,
            include_toxicity=include_toxicity,
            include_rating=include_rating
        )
        return result
        
    except ClassificationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/history")
async def get_classification_history(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get classification history"""
    try:
        if limit > 1000:
            limit = 1000  # Prevent excessive queries
        
        history = classification_service.get_classification_history(
            db=db,
            limit=limit,
            offset=offset
        )
        return history
        
    except ClassificationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Check system health"""
    try:
        health_status = classification_service.inference_service.health_check()
        
        if health_status["status"] == "healthy":
            return health_status
        else:
            raise HTTPException(status_code=503, detail=health_status)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

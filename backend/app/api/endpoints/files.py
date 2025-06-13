from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from ...core.database import get_db
from ...models.database import Document
from ...models.schemas import DocumentBase
from ...services.file_handler import FileHandler
from ...utils.exceptions import FileProcessingError

router = APIRouter()
file_handler = FileHandler()

@router.get("/", response_model=List[DocumentBase])
async def get_all_documents(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000)
):
    """Получить список всех загруженных документов"""
    try:
        documents = db.query(Document).offset(skip).limit(limit).all()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}", response_model=DocumentBase)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Получить метаданные документа по ID"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Удалить документ и связанные данные"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        db.delete(document)
        db.commit()
        return {"message": "Document deleted successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

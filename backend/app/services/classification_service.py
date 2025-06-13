import time
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from ..models.database import Document, ClassificationResult
from ..models.schemas import DocumentCreate, ClassificationResponse
from ..services.file_handler import FileHandler
from ..services.model_inference import ModelInferenceService
from ..core.database import get_db
from ..utils.exceptions import ClassificationError

class ClassificationService:
    def __init__(self):
        self.file_handler = FileHandler()
        self.inference_service = ModelInferenceService()
    
    def process_file_upload(self, file_content: bytes, filename: str, 
                           db: Session) -> Document:
        """Process uploaded file and save to database"""
        try:
            # Validate file
            file_size = len(file_content)
            self.file_handler.validate_file(filename, file_size)
            
            # Extract text content
            text_content = self.file_handler.extract_text_from_file(file_content, filename)
            
            if not text_content.strip():
                raise ClassificationError("File contains no readable text")
            
            # Get file info
            file_info = self.file_handler.get_file_info(file_content, filename)
            
            # Create document in database
            document = Document(
                filename=filename,
                file_type=file_info['file_type'],
                file_size=file_size,
                content=text_content
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            return document
            
        except Exception as e:
            db.rollback()
            raise ClassificationError(f"Error processing file upload: {str(e)}")
    
    def classify_document(self, document_id: int, db: Session, 
                         include_toxicity: bool = True, 
                         include_rating: bool = True) -> ClassificationResponse:
        """Classify document and save results"""
        start_time = time.time()
        
        try:
            # Get document
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ClassificationError(f"Document with id {document_id} not found")
            
            # Perform classification
            classification_results = self.inference_service.classify_text_comprehensive(
                document.content,
                include_toxicity=include_toxicity,
                include_rating=include_rating
            )
            
            # Extract results
            toxicity_result = classification_results.get('toxicity', {})
            rating_result = classification_results.get('rating', {})
            
            # Create classification result record
            result_record = ClassificationResult(
                document_id=document_id,
                toxicity_score=toxicity_result.get('toxicity_score'),
                is_toxic=toxicity_result.get('is_toxic'),
                toxicity_confidence=toxicity_result.get('confidence'),
                rating_category=rating_result.get('category'),
                rating_confidence=rating_result.get('confidence'),
                rating_probabilities=rating_result.get('probabilities'),
                processing_time=time.time() - start_time,
                model_versions={
                    'toxicity_model': toxicity_result.get('model_name'),
                    'rating_model': rating_result.get('model_name')
                }
            )
            
            db.add(result_record)
            db.commit()
            db.refresh(result_record)
            
            # Create response
            response = ClassificationResponse(
                document_id=document_id,
                toxicity_result={
                    "toxicity_score": toxicity_result.get('toxicity_score', 0.0),
                    "is_toxic": toxicity_result.get('is_toxic', False),
                    "confidence": toxicity_result.get('confidence', 0.0)
                },
                rating_result={
                    "category": rating_result.get('category', 'Unknown'),
                    "confidence": rating_result.get('confidence', 0.0),
                    "probabilities": rating_result.get('probabilities', {})
                },
                processing_time=result_record.processing_time
            )
            
            return response
            
        except Exception as e:
            db.rollback()
            raise ClassificationError(f"Error classifying document: {str(e)}")
    
    def classify_text_direct(self, text: str, include_toxicity: bool = True, 
                            include_rating: bool = True) -> Dict:
        """Classify text directly without saving to database"""
        try:
            return self.inference_service.classify_text_comprehensive(
                text,
                include_toxicity=include_toxicity,
                include_rating=include_rating
            )
        except Exception as e:
            raise ClassificationError(f"Error classifying text: {str(e)}")
    
    def get_classification_history(self, db: Session, limit: int = 100, 
                                  offset: int = 0) -> List[Dict]:
        """Get classification history"""
        try:
            results = (
                db.query(ClassificationResult, Document)
                .join(Document, ClassificationResult.document_id == Document.id)
                .order_by(ClassificationResult.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            history = []
            for result, document in results:
                history.append({
                    "id": result.id,
                    "document_id": document.id,
                    "filename": document.filename,
                    "toxicity_score": result.toxicity_score,
                    "is_toxic": result.is_toxic,
                    "rating_category": result.rating_category,
                    "rating_confidence": result.rating_confidence,
                    "processing_time": result.processing_time,
                    "created_at": result.created_at
                })
            
            return history
            
        except Exception as e:
            raise ClassificationError(f"Error getting classification history: {str(e)}")
    
    def get_analytics_data(self, db: Session) -> Dict:
        """Get analytics data for dashboard"""
        try:
            # Total documents
            total_docs = db.query(Document).count()
            
            # Total classifications
            total_classifications = db.query(ClassificationResult).count()
            
            # Average processing time
            avg_processing_time = (
                db.query(ClassificationResult.processing_time)
                .filter(ClassificationResult.processing_time.is_not(None))
                .all()
            )
            avg_time = sum(t[0] for t in avg_processing_time) / len(avg_processing_time) if avg_processing_time else 0
            
            # Toxicity distribution
            toxicity_counts = (
                db.query(ClassificationResult.is_toxic)
                .filter(ClassificationResult.is_toxic.is_not(None))
                .all()
            )
            toxic_count = sum(1 for t in toxicity_counts if t[0])
            non_toxic_count = len(toxicity_counts) - toxic_count
            
            # Rating distribution
            rating_counts = (
                db.query(ClassificationResult.rating_category)
                .filter(ClassificationResult.rating_category.is_not(None))
                .all()
            )
            rating_distribution = {}
            for rating in rating_counts:
                category = rating[0]
                rating_distribution[category] = rating_distribution.get(category, 0) + 1
            
            return {
                "total_documents": total_docs,
                "total_classifications": total_classifications,
                "avg_processing_time": avg_time,
                "toxicity_distribution": {
                    "toxic": toxic_count,
                    "non_toxic": non_toxic_count
                },
                "rating_distribution": rating_distribution
            }
            
        except Exception as e:
            raise ClassificationError(f"Error getting analytics data: {str(e)}")

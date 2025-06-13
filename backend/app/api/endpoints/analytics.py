from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ...core.database import get_db
from ...services.classification_service import ClassificationService
from ...utils.exceptions import ClassificationError

router = APIRouter()
classification_service = ClassificationService()

@router.get("/dashboard")
async def get_dashboard_data(db: Session = Depends(get_db)):
    """Get analytics data for dashboard"""
    try:
        analytics_data = classification_service.get_analytics_data(db)
        return analytics_data
        
    except ClassificationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/statistics")
async def get_detailed_statistics(db: Session = Depends(get_db)):
    """Get detailed statistics"""
    try:
        # This could be expanded with more detailed analytics
        # For now, return the same data as dashboard
        analytics_data = classification_service.get_analytics_data(db)
        
        # Add some computed statistics
        analytics_data["statistics"] = {
            "toxicity_rate": analytics_data["toxicity_distribution"]["toxic"] / max(1, sum(analytics_data["toxicity_distribution"].values())),
            "most_common_rating": max(analytics_data["rating_distribution"], key=analytics_data["rating_distribution"].get) if analytics_data["rating_distribution"] else None,
            "avg_processing_time_ms": analytics_data["avg_processing_time"] * 1000
        }
        
        return analytics_data
        
    except ClassificationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

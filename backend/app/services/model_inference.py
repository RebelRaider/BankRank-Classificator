import time
import json
from typing import Dict, List, Optional
from ..ml.toxicity_classifier import ToxicityClassifier
from ..ml.rating_classifier import RatingClassifier
from ..ml.data_processor import TextProcessor
from ..core.database import get_redis
from ..core.config import settings

class ModelInferenceService:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.toxicity_classifier = None
        self.rating_classifier = None
        self.redis_client = get_redis()
        self._load_models()
    
    def _load_models(self):
        """Load ML models"""
        try:
            print("Loading toxicity classifier...")
            self.toxicity_classifier = ToxicityClassifier()
            
            print("Loading rating classifier...")
            self.rating_classifier = RatingClassifier()
            # Try to load trained model, fallback to pretrained if not available
            try:
                self.rating_classifier.load_trained_model("./trained_rating_model")
            except:
                print("Trained rating model not found, loading pretrained BERT...")
                self.rating_classifier.load_pretrained_model()
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise e
    
    def _get_cache_key(self, text: str, model_type: str) -> str:
        """Generate cache key for text and model type"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"prediction:{model_type}:{text_hash}"
    
    def _cache_prediction(self, cache_key: str, prediction: Dict, ttl: int = 3600):
        """Cache prediction result"""
        try:
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(prediction, default=str)
            )
        except Exception as e:
            print(f"Error caching prediction: {e}")
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[Dict]:
        """Get cached prediction"""
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Error getting cached prediction: {e}")
        return None
    
    def classify_toxicity(self, text: str, use_cache: bool = True) -> Dict:
        """Classify text toxicity"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(text, "toxicity")
        if use_cache:
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                cached_result['processing_time'] = time.time() - start_time
                cached_result['from_cache'] = True
                return cached_result
        
        # Preprocess text
        processed_text = self.text_processor.prepare_for_toxicity_model(text)
        
        # Make prediction
        prediction = self.toxicity_classifier.predict(processed_text)
        
        # Add metadata
        prediction['processing_time'] = time.time() - start_time
        prediction['from_cache'] = False
        prediction['model_name'] = settings.TOXICITY_MODEL_NAME
        
        # Cache result
        if use_cache:
            self._cache_prediction(cache_key, prediction)
        
        return prediction
    
    def classify_rating(self, text: str, use_cache: bool = True) -> Dict:
        """Classify bank rating from text"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(text, "rating")
        if use_cache:
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                cached_result['processing_time'] = time.time() - start_time
                cached_result['from_cache'] = True
                return cached_result
        
        # Preprocess text
        processed_text = self.text_processor.prepare_for_rating_model(text)
        
        # Make prediction
        prediction = self.rating_classifier.predict(processed_text)
        
        # Add metadata
        prediction['processing_time'] = time.time() - start_time
        prediction['from_cache'] = False
        prediction['model_name'] = settings.RATING_MODEL_NAME
        
        # Cache result
        if use_cache:
            self._cache_prediction(cache_key, prediction)
        
        return prediction
    
    def classify_text_comprehensive(self, text: str, include_toxicity: bool = True, 
                                   include_rating: bool = True) -> Dict:
        """Perform comprehensive text classification"""
        start_time = time.time()
        results = {
            "original_text_length": len(text),
            "word_count": len(text.split())
        }
        
        # Toxicity classification
        if include_toxicity:
            try:
                toxicity_result = self.classify_toxicity(text)
                results['toxicity'] = toxicity_result
            except Exception as e:
                results['toxicity'] = {
                    "error": str(e),
                    "toxicity_score": None,
                    "is_toxic": None,
                    "confidence": None
                }
        
        # Rating classification
        if include_rating:
            try:
                rating_result = self.classify_rating(text)
                results['rating'] = rating_result
            except Exception as e:
                results['rating'] = {
                    "error": str(e),
                    "category": None,
                    "confidence": None,
                    "probabilities": None
                }
        
        results['total_processing_time'] = time.time() - start_time
        return results
    
    def batch_classify(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """Batch classification for multiple texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.classify_text_comprehensive(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def health_check(self) -> Dict:
        """Check if models are loaded and working"""
        try:
            test_text = "Это тестовый текст для проверки работоспособности моделей."
            
            # Test toxicity model
            toxicity_result = self.classify_toxicity(test_text, use_cache=False)
            toxicity_working = 'toxicity_score' in toxicity_result
            
            # Test rating model
            rating_result = self.classify_rating(test_text, use_cache=False)
            rating_working = 'category' in rating_result
            
            return {
                "status": "healthy" if toxicity_working and rating_working else "unhealthy",
                "toxicity_model": "working" if toxicity_working else "failed",
                "rating_model": "working" if rating_working else "failed",
                "cache_connection": self._test_cache_connection()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "toxicity_model": "unknown",
                "rating_model": "unknown",
                "cache_connection": False
            }
    
    def _test_cache_connection(self) -> bool:
        """Test Redis cache connection"""
        try:
            self.redis_client.ping()
            return True
        except:
            return False

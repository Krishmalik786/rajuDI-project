from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import BreastCancerPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="ML model API for breast cancer classification (Malignant vs Benign)",
    version="1.0.0"
)

# Load model at startup (efficient - load once, use many times)
predictor = None

@app.on_event("startup")
async def load_model():
    """Load model when API starts"""
    global predictor
    predictor = BreastCancerPredictor()
    print("✅ Model loaded and ready for predictions!")

# Define input data structure
class FeatureInput(BaseModel):
    """Expected input features for prediction"""
    mean_radius: float = Field(..., description="Mean of distances from center to points on perimeter")
    mean_texture: float = Field(..., description="Standard deviation of gray-scale values")
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189
            }
        }

# Root endpoint
@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Breast Cancer Prediction API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if API and model are working"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True
    }

# Prediction endpoint
@app.post("/predict")
async def predict(features: FeatureInput):
    """
    Make prediction on input features
    
    Returns prediction class and probabilities
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dict with correct column names (spaces instead of underscores)
        features_dict = {
            'mean radius': features.mean_radius,
            'mean texture': features.mean_texture,
            'mean perimeter': features.mean_perimeter,
            'mean area': features.mean_area,
            'mean smoothness': features.mean_smoothness,
            'mean compactness': features.mean_compactness,
            'mean concavity': features.mean_concavity,
            'mean concave points': features.mean_concave_points,
            'mean symmetry': features.mean_symmetry,
            'mean fractal dimension': features.mean_fractal_dimension,
            'radius error': features.radius_error,
            'texture error': features.texture_error,
            'perimeter error': features.perimeter_error,
            'area error': features.area_error,
            'smoothness error': features.smoothness_error,
            'compactness error': features.compactness_error,
            'concavity error': features.concavity_error,
            'concave points error': features.concave_points_error,
            'symmetry error': features.symmetry_error,
            'fractal dimension error': features.fractal_dimension_error,
            'worst radius': features.worst_radius,
            'worst texture': features.worst_texture,
            'worst perimeter': features.worst_perimeter,
            'worst area': features.worst_area,
            'worst smoothness': features.worst_smoothness,
            'worst compactness': features.worst_compactness,
            'worst concavity': features.worst_concavity,
            'worst concave points': features.worst_concave_points,
            'worst symmetry': features.worst_symmetry,
            'worst fractal dimension': features.worst_fractal_dimension
        }
        
        # Make prediction
        result = predictor.predict(features_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run with: uvicorn deployment.app:app --reload

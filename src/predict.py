import joblib
import pandas as pd
import numpy as np

class BreastCancerPredictor:
    """Model predictor for breast cancer classification"""
    
    def __init__(self, model_path='models/best_model.pkl', scaler_path='data/processed/scaler.pkl'):
        """Load model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Scaler loaded from {scaler_path}")
    
    def predict(self, features):
        """
        Make prediction on input features
        
        Args:
            features: dict or DataFrame with feature values
        
        Returns:
            dict with prediction and probability
        """
        # Convert to DataFrame if dict
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Format result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Benign' if prediction == 1 else 'Malignant',
            'probability_malignant': float(probability[0]),
            'probability_benign': float(probability[1]),
            'confidence': float(max(probability))
        }
        
        return result

# Test if run directly
if __name__ == "__main__":
    # Load predictor
    predictor = BreastCancerPredictor()
    
    # Example features (first row from dataset)
    example_features = {
        'mean radius': 17.99,
        'mean texture': 10.38,
        'mean perimeter': 122.8,
        'mean area': 1001.0,
        'mean smoothness': 0.1184,
        'mean compactness': 0.2776,
        'mean concavity': 0.3001,
        'mean concave points': 0.1471,
        'mean symmetry': 0.2419,
        'mean fractal dimension': 0.07871,
        'radius error': 1.095,
        'texture error': 0.9053,
        'perimeter error': 8.589,
        'area error': 153.4,
        'smoothness error': 0.006399,
        'compactness error': 0.04904,
        'concavity error': 0.05373,
        'concave points error': 0.01587,
        'symmetry error': 0.03003,
        'fractal dimension error': 0.006193,
        'worst radius': 25.38,
        'worst texture': 17.33,
        'worst perimeter': 184.6,
        'worst area': 2019.0,
        'worst smoothness': 0.1622,
        'worst compactness': 0.6656,
        'worst concavity': 0.7119,
        'worst concave points': 0.2654,
        'worst symmetry': 0.4601,
        'worst fractal dimension': 0.1189
    }
    
    # Test prediction
    result = predictor.predict(example_features)
    print(f"\n🔮 Prediction Result:")
    print(f"   Class: {result['prediction_label']}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")

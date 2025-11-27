import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os

def load_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Calculate all evaluation metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with MLflow tracking"""
    
    # Start MLflow run
    with mlflow.start_run(run_name="Logistic_Regression"):
        
        print("\n🔵 Training Logistic Regression...")
        
        # Define model
        model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Log parameters
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print(f"✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return model, metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with MLflow tracking"""
    
    with mlflow.start_run(run_name="Random_Forest"):
        
        print("\n🟢 Training Random Forest...")
        
        # Define model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Log parameters
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print(f"✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return model, metrics

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with MLflow tracking"""
    
    with mlflow.start_run(run_name="XGBoost"):
        
        print("\n🟡 Training XGBoost...")
        
        # Define model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("random_state", 42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print(f"✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return model, metrics

def save_best_model(models_results):
    """Save the best performing model"""
    
    # Find best model by accuracy
    best_model_name = max(models_results, key=lambda x: models_results[x]['metrics']['accuracy'])
    best_model = models_results[best_model_name]['model']
    best_metrics = models_results[best_model_name]['metrics']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Save model info
    with open('models/model_info.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {best_metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC: {best_metrics['roc_auc']:.4f}\n")
    
    print(f"✅ Best model saved to models/best_model.pkl")
    
    return best_model_name, best_metrics

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("🚀 STARTING MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Set MLflow experiment
    mlflow.set_experiment("breast_cancer_classification")
    
    # Load data
    print("\n📊 Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train all models
    models_results = {}
    
    # 1. Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
    models_results['Logistic Regression'] = {'model': lr_model, 'metrics': lr_metrics}
    
    # 2. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    models_results['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # 3. XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    models_results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    # Save best model
    best_model_name, best_metrics = save_best_model(models_results)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n🎯 View experiments at: http://localhost:5000")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def preprocess_data():
    """
    Preprocess raw data:
    1. Load raw data
    2. Split features and target
    3. Train-test split
    4. Scale features (standardization)
    5. Save processed data and scaler
    """
    
    print("🔄 Starting preprocessing...")
    
    # 1. Load raw data
    df = pd.read_csv('data/raw/breast_cancer.csv')
    print(f"✅ Loaded data: {df.shape}")
    
    # 2. Separate features and target
    X = df.drop('target', axis=1)  # All columns except target
    y = df['target']  # Only target column
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # 3. Split into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 20% for testing
        random_state=42,    # For reproducibility
        stratify=y          # Keep same class ratio in train/test
    )
    
    print(f"✅ Train-test split:")
    print(f"   Train size: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"   Test size: {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.1f}%)")
    
    # 4. Scale features (StandardScaler: mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train, transform train
    X_test_scaled = scaler.transform(X_test)        # Only transform test (no fit!)
    
    print(f"✅ Features scaled (mean=0, std=1)")
    
    # Convert back to DataFrames (keep column names)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 5. Save processed data
    os.makedirs('data/processed', exist_ok=True)
    
    X_train_scaled.to_csv('data/processed/X_train.csv', index=False)
    X_test_scaled.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save scaler (needed for future predictions)
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    print(f"✅ Saved processed data to data/processed/")
    print(f"   - X_train.csv, X_test.csv")
    print(f"   - y_train.csv, y_test.csv")
    print(f"   - scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    preprocess_data()
    print("\n✅ Preprocessing complete!")

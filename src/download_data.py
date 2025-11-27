import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def download_dataset():
    # Load dataset
    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data.data, data.target], axis=1)
    
    # Save
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/breast_cancer.csv', index=False)
    
    print(f"✅ Dataset downloaded: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")

if __name__ == "__main__":
    download_dataset()

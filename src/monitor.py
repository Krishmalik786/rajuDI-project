import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def simple_monitoring_report():
    """Generate simple monitoring report without Evidently"""
    
    print("📊 Generating simple monitoring report...")
    
    # Load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Calculate statistics
    train_stats = X_train.describe()
    test_stats = X_test.describe()
    
    # Compare distributions
    drift_detected = []
    for col in X_train.columns:
        train_mean = X_train[col].mean()
        test_mean = X_test[col].mean()
        train_std = X_train[col].std()
        
        # Simple drift detection: if test mean differs by >1 std
        if abs(test_mean - train_mean) > train_std:
            drift_detected.append(col)
    
    # Generate HTML report
    os.makedirs('reports', exist_ok=True)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Monitoring Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .good {{ color: #27ae60; }}
            .warning {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <h1>📊 Model Monitoring Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="metric">
            <h2>Dataset Summary</h2>
            <p>Training samples: {len(X_train)}</p>
            <p>Test samples: {len(X_test)}</p>
            <p>Features: {X_train.shape[1]}</p>
        </div>
        
        <div class="metric">
            <h2>Data Drift Analysis</h2>
            <p>Features with drift: <strong class="{'warning' if drift_detected else 'good'}">{len(drift_detected)}</strong> / {X_train.shape[1]}</p>
            {'<p class="good">✅ No significant drift detected</p>' if not drift_detected else f'<p class="warning">⚠️ Drift detected in: {", ".join(drift_detected[:5])}</p>'}
        </div>
        
        <div class="metric">
            <h2>Target Distribution</h2>
            <p>Train - Class 0: {(y_train['target']==0).sum()} | Class 1: {(y_train['target']==1).sum()}</p>
            <p>Test - Class 0: {(y_test['target']==0).sum()} | Class 1: {(y_test['target']==1).sum()}</p>
        </div>
        
        <div class="metric">
            <h2>Recommendations</h2>
            {'<p class="good">✅ Model performance should be stable</p>' if len(drift_detected) < 3 else '<p class="warning">⚠️ Consider retraining model due to drift</p>'}
        </div>
    </body>
    </html>
    """
    
    report_path = 'reports/data_monitoring_report.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Report saved to {report_path}")
    print(f"\n📈 Monitoring Summary:")
    print(f"   Features analyzed: {X_train.shape[1]}")
    print(f"   Drift detected: {len(drift_detected)} features")
    if drift_detected:
        print(f"   Drifted features: {', '.join(drift_detected[:5])}")
    else:
        print(f"   ✅ No significant drift")
    
    return report_path

if __name__ == "__main__":
    report_path = simple_monitoring_report()
    print(f"\n🌐 Open: file://{os.path.abspath(report_path)}")

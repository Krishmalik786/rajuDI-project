"""
Simple ML Pipeline Orchestration
Runs all steps in sequence: download → preprocess → train
"""

import subprocess
import sys
import time

def run_step(step_name, script_path):
    """Run a pipeline step and handle errors"""
    print(f"\n{'='*60}")
    print(f"▶️  STEP: {step_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ {step_name} FAILED!")
        print(f"Error: {result.stderr}")
        return False
    
    print(result.stdout)
    print(f"✅ {step_name} completed in {elapsed:.2f}s")
    
    return True

def main():
    """Main pipeline orchestration"""
    
    print("\n" + "="*60)
    print("🚀 ML PIPELINE ORCHESTRATION")
    print("="*60)
    print("\nPipeline Steps:")
    print("  1. Download Data")
    print("  2. Preprocess Data")
    print("  3. Train Models with MLflow")
    print("="*60)
    
    pipeline_start = time.time()
    
    # Step 1: Download
    if not run_step("Download Data", "src/download_data.py"):
        print("\n❌ Pipeline failed at Download step")
        return False
    
    # Step 2: Preprocess
    if not run_step("Preprocess Data", "src/preprocess.py"):
        print("\n❌ Pipeline failed at Preprocessing step")
        return False
    
    # Step 3: Train
    if not run_step("Train Models", "src/train.py"):
        print("\n❌ Pipeline failed at Training step")
        return False
    
    # Success!
    total_time = time.time() - pipeline_start
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
    print("\n📊 Next steps:")
    print("   • View MLflow experiments: http://localhost:5000")
    print("   • Check best model: models/best_model.pkl")
    print("   • Review processed data: data/processed/")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

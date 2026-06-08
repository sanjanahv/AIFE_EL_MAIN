import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_script(script_path):
    print(f"\n{'='*50}\nRunning: {script_path}\n{'='*50}")
    try:
        # Use sys.executable to ensure we use the same Python environment
        result = subprocess.run([sys.executable, script_path], check=True, cwd=PROJECT_ROOT)
        print(f"\n[SUCCESS] {script_path} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script_path} failed with exit code {e.returncode}.\n")
        sys.exit(1)

def main():
    print("Starting setup pipeline...")
    
    preprocess_script = os.path.join(PROJECT_ROOT, "data", "preprocess.py")
    train_script = os.path.join(PROJECT_ROOT, "models", "train_lgbm.py")
    
    run_script(preprocess_script)
    run_script(train_script)
    
    print("\nSetup pipeline completed successfully!")
    print("You can now start the dashboard by running: python app.py")

if __name__ == "__main__":
    main()

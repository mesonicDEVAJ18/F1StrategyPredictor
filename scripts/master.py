import subprocess
import sys
import time

def run_step(script_name):
    """
    Runs a python script and checks for errors.
    stops the master script if a subprocess fails.
    """
    print(f"--- [START] Running {script_name} ---")
    start_time = time.time()
    
    try:
        # Run the script using the current python executable
        result = subprocess.run([sys.executable, script_name], check=True)
        
        elapsed = time.time() - start_time
        print(f"--- [DONE] {script_name} finished in {elapsed:.2f}s ---\n")
        
    except subprocess.CalledProcessError as e:
        print(f"!!! [ERROR] {script_name} failed with exit code {e.returncode}. Stopping pipeline.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"!!! [ERROR] Could not find file: {script_name}. Check your directory.")
        sys.exit(1)

def main():
    print("Starting Master Pipeline...\n")

    # 1. Run all processing scripts
    # (Order usually doesn't matter here, but they are run sequentially)
    process_scripts = [
        "process_laps.py",
        "process_stints.py",
        "process_weather.py"
    ]

    for script in process_scripts:
        run_step(script)

    # 2. Run the main pipeline in specific order
    pipeline_sequence = [
        "feature_engineer.py",      # Feature Engineering
        "train_model.py",           # Training
        "strategy_simulator.py",    # Simulation
        "validate_strategy.py"      # Validation
    ]

    for script in pipeline_sequence:
        run_step(script)

    print("=== Pipeline Complete Successfully ===")

if __name__ == "__main__":
    main()
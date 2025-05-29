import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run protein classification benchmarks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-subfamily", action="store_true", help="Run subfamily-level classification (neural_network_subfamily.py)")
    group.add_argument("-family", action="store_true", help="Run family-level classification (neural_network_family.py)")

    args = parser.parse_args()

    # Get the directory of the current script (run.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.subfamily:
        script_to_run = os.path.join(script_dir, "neural_network_subfamily.py")
        print("Running subfamily-level classification (neural_network_subfamily.py)...")
    elif args.family:
        script_to_run = os.path.join(script_dir, "neural_network_family.py")
        print("Running family-level classification (neural_network_family.py)...")
    else:
        # This case should not be reached due to the mutually exclusive group being required
        parser.print_help()
        sys.exit(1)

    try:
        # Ensure the script to run exists
        if not os.path.exists(script_to_run):
            print(f"Error: Script not found at {script_to_run}")
            sys.exit(1)
            
        # Run the selected script
        # The CWD will be the 'scripts' directory itself.
        # The benchmark scripts use relative paths like '../data_source', so they expect to be run from the 'scripts' directory.
        process = subprocess.Popen([sys.executable, script_to_run], cwd=script_dir)
        process.wait() # Wait for the script to complete

        if process.returncode == 0:
            print(f"Successfully executed {os.path.basename(script_to_run)}.")
        else:
            print(f"Error executing {os.path.basename(script_to_run)}. Return code: {process.returncode}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

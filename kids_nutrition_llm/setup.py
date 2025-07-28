import argparse
import os
import subprocess
import sys

def install_requirements():
    """Installs the required packages from requirements.txt."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import os

def run_full_mode():
    """Runs the full application pipeline."""
    print("Setting up in full mode...")
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Generate the dataset
    print("Generating dataset...")
    subprocess.run([sys.executable, os.path.join(script_dir, "data/create_nutrition_dataset.py")], check=True)
    # Fine-tune the model
    print("Fine-tuning the model...")
    subprocess.run([sys.executable, os.path.join(script_dir, "src/fine_tune_model.py")], check=True)
    # Run the Gradio app
    print("Starting the Gradio application...")
    subprocess.run([sys.executable, os.path.join(script_dir, "deployment/app.py")], check=True)

def run_demo_mode():
    """Runs the application in demo mode."""
    print("Setting up in demo mode...")
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Run the Gradio app with a pre-trained model
    print("Starting the Gradio application with a pre-trained model...")
    subprocess.run([sys.executable, os.path.join(script_dir, "run_demo.py")], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and run the Kids Nutrition LLM.")
    parser.add_argument(
        "--mode",
        choices=["full", "demo"],
        required=True,
        help="The mode to run the application in.",
    )
    args = parser.parse_args()

    install_requirements()

    if args.mode == "full":
        run_full_mode()
    elif args.mode == "demo":
        run_demo_mode()

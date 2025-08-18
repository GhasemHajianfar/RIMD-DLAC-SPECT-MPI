#!/usr/bin/env python3
"""
Usage script to run the inference test on NM raw files

This script demonstrates how to use the inference_test.py file to:
1. Take NM raw files as input
2. Reconstruct them to NAC images in 3 OSEM settings
3. Generate ATM predictions using ensemble of trained models from NAC images
4. Create ensemble ATM predictions by averaging model outputs
5. Create MAC (AC) reconstructions using the ensemble ATM predictions

Usage:
    python run_inference_test.py
"""

import subprocess
import sys
import os

def run_inference_test():
    """
    Inferemce function showing how to run the inference test
    """
    
    # Configuration - modify these paths according to your setup
    config = {
        "nm_raw_dir": "/media/ghasem/Extreme SSD/deep_project/Cardiac/PsudoCT/external_ge/NIfTI/NM2",  # Directory containing NM raw files
        "output_dir": "/media/ghasem/Extreme SSD/deep_project/Cardiac/PsudoCT/infer_test",   # Output directory for results
        "exp_name": "test_ge",               # Experiment name
        "model_dir": "/media/ghasem/Extreme SSD/deep_project/Cardiac/PsudoCT/test_model/train/ATM_AC_loss/OSEM_3",        # Directory containing trained model files
        "colimator": "G8-LEHR",                      # Collimator type
        "num_workers": 4                             # Number of workers for data loading
    }
    
    # Build the command
    cmd = [
        "python", "inference_test.py",
        "--nm_raw_dir", config["nm_raw_dir"],
        "--output_dir", config["output_dir"],
        "--exp_name", config["exp_name"],
        "--model_dir", config["model_dir"],
        "--colimator", config["colimator"],
        "--num_workers", str(config["num_workers"])
    ]
    
    print("Running inference test with the following configuration:")
    print(f"NM Raw Directory: {config['nm_raw_dir']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Experiment Name: {config['exp_name']}")
    print(f"Model Directory: {config['model_dir']}")
    print(f"Collimator: {config['colimator']}")
    print(f"Number of Workers: {config['num_workers']}")
    print("\nCommand:")
    print(" ".join(cmd))
    
    # Check if paths exist
    if not os.path.exists(config["nm_raw_dir"]):
        print(f"ERROR: NM raw directory does not exist: {config['nm_raw_dir']}")
        return False
    
    if not os.path.exists(config["model_dir"]):
        print(f"ERROR: Model directory does not exist: {config['model_dir']}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    try:
        # Run the command
        print("\nStarting inference test...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Inference test completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Inference test failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ERROR: inference_test.py not found. Make sure you're in the correct directory.")
        return False

def print_usage():
    """
    Print usage for different scenarios
    """
    print("\n" + "="*60)
    print("USAGE")
    print("="*60)
    
    print("\n1. Basic Ensemble Mode:")
    print("python inference_test.py \\")
    print("    --nm_raw_dir /path/to/nm/files \\")
    print("    --output_dir /path/to/output \\")
    print("    --exp_name ensemble_test \\")
    print("    --model_dir /path/to/models/")
    
    print("\n2. Custom Collimator:")
    print("python inference_test.py \\")
    print("    --nm_raw_dir /path/to/nm/files \\")
    print("    --output_dir /path/to/output \\")
    print("    --exp_name custom_collimator \\")
    print("    --model_dir /path/to/models/ \\")
    print("    --colimator G8-LEHR")
    
    print("\n3. Custom Workers:")
    print("python inference_test.py \\")
    print("    --nm_raw_dir /path/to/nm/files \\")
    print("    --output_dir /path/to/output \\")
    print("    --exp_name custom_workers \\")
    print("    --model_dir /path/to/models/ \\")
    print("    --num_workers 8")
    
    print("\n" + "="*60)
    print("OUTPUT STRUCTURE")
    print("="*60)
    print("The script will create the following directory structure:")
    print("output_dir/exp_name/")
    print("├── NM/                    # Copied NM raw files")
    print("├── NAC/                   # NAC reconstructions")
    print("│   ├── OSEM_4I4S/        # 4 iterations, 4 subsets")
    print("│   ├── OSEM_6I6S/        # 6 iterations, 6 subsets")
    print("│   └── OSEM_8I8S/        # 8 iterations, 8 subsets")
    print("├── ATM_predictions/       # Individual model ATM predictions")
    print("│   ├── model1.pt/        # Predictions from model 1")
    print("│   ├── model2.pt/        # Predictions from model 2")
    print("│   └── ...               # Predictions from other models")
    print("├── ATM_ensemble/         # Ensemble ATM predictions (averaged)")
    print("└── MAC/                  # AC reconstructions")
    print("    ├── OSEM_4I4S/        # AC with 4 iterations, 4 subsets")
    print("    ├── OSEM_6I6S/        # AC with 6 iterations, 6 subsets")
    print("    └── OSEM_8I8S/        # AC with 8 iterations, 8 subsets")

if __name__ == "__main__":
    print("Inference Test Usage")
    print("="*30)
    
    # Check if inference_test.py exists
    if not os.path.exists("inference_test.py"):
        print("ERROR: inference_test.py not found in current directory")
        print("Please make sure you're running this script from the project directory")
        sys.exit(1)
    
    # Show usage
    print_usage()
    
    # Ask user if they want to run the inference test
    print("\n" + "="*60)
    response = input("Do you want to run the inference test? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\nNOTE: You need to modify the paths in the script before running!")
        print("Please edit the 'config' dictionary in run_inference_test.py")
        print("to point to your actual NM raw files and model.")
        
        response2 = input("Have you updated the paths? (y/n): ").lower().strip()
        if response2 in ['y', 'yes']:
            success = run_inference_test()
            if success:
                print("\nInfernece completed successfully!")
            else:
                print("\nInfernece failed. Please check the error messages above.")
        else:
            print("Please update the paths and run again.")
    else:
        print("Infernece test not run. You can run inference_test.py directly with your own parameters.")

#!/usr/bin/env python
"""
Project Setup Verification
==========================
Verifies that all required files are in place and paths are correct.
"""

import sys
from pathlib import Path

def check_file_exists(path, description):
    """Check if a file exists and print status."""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def check_dir_exists(path, description):
    """Check if a directory exists and print status."""
    exists = Path(path).is_dir()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("="*80)
    print("PROJECT SETUP VERIFICATION")
    print("="*80 + "\n")
    
    base = Path("c:/Users/Jashnoor/Desktop/sem 4/IQF/CFM4_final/project")
    
    print("1. DIRECTORY STRUCTURE")
    print("-" * 80)
    
    dirs_ok = True
    dirs_to_check = [
        (base / "project_code", "Project Code Directory"),
        (base / "data", "Data Directory"),
        (base / "plots", "Plots Directory"),
        (base / "website", "Website Directory"),
        (base / "results", "Results Directory"),
    ]
    
    for path, desc in dirs_to_check:
        if not check_dir_exists(path, desc):
            dirs_ok = False
    
    print("\n2. CORE PYTHON FILES")
    print("-" * 80)
    
    files_ok = True
    files_to_check = [
        (base / "project_code" / "Quant project.py", "Main Quant Project"),
        (base / "project_code" / "run_wilcoxon_test.py", "Wilcoxon Test Runner"),
        (base / "project_code" / "complete_project_code.py", "Core Algorithms"),
        (base / "project_code" / "webapp.py", "Web Application"),
    ]
    
    for path, desc in files_to_check:
        if not check_file_exists(path, desc):
            files_ok = False
    
    print("\n3. DATA FILES")
    print("-" * 80)
    
    data_files = [
        (base / "data" / "daily_returns.xlsx", "Daily Returns Data"),
    ]
    
    for path, desc in data_files:
        check_file_exists(path, desc)
    
    print("\n4. DOCUMENTATION")
    print("-" * 80)
    
    check_file_exists(base / "README.md", "Project README")
    
    print("\n5. EXECUTION TEST")
    print("-" * 80)
    
    # Try to import the Quant project module
    try:
        sys.path.insert(0, str(base / "project_code"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "quant_project",
            base / "project_code" / "Quant project.py"
        )
        quant_module = importlib.util.module_from_spec(spec)
        print("✓ Quant project module imports successfully")
        
        # Check if key functions exist
        spec.loader.exec_module(quant_module)
        functions_to_check = ["run_dtlz_suite", "ranksum_summary", "save_table", "save_pickle"]
        for func_name in functions_to_check:
            if hasattr(quant_module, func_name):
                print(f"  ✓ Function available: {func_name}")
            else:
                print(f"  ✗ Function missing: {func_name}")
                
    except Exception as e:
        print(f"✗ Error importing Quant project: {e}")
    
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    
    if dirs_ok and files_ok:
        print("✓ All directories and files are in place!")
        print("\nTO RUN THE WILCOXON TEST (n=30):")
        print("  cd project/project_code")
        print("  python run_wilcoxon_test.py")
        print("\nExpected runtime: 20-60 minutes")
        print("Output location: project/results/QuantProjectResults/")
        return 0
    else:
        print("✗ Some required files or directories are missing")
        print("Please check the setup")
        return 1

if __name__ == "__main__":
    sys.exit(main())

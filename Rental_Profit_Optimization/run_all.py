"""
===============================================================================
RENTAL PROFIT OPTIMIZATION — MASTER RUNNER
Runs all pipeline scripts in order with proper UTF-8 encoding.
Usage: python run_all.py
===============================================================================
"""
import sys
import os
import importlib.util
import io

# Force UTF-8 for all print output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")


def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"\n{'='*70}")
    print(f"  RUNNING: {script_name}")
    print(f"{'='*70}")
    spec = importlib.util.spec_from_file_location("module", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    scripts = [
        "01_generate_data.py",
        "02_preprocess_data.py",
        "03_business_analytics.py",
        "04_ml_models.py",
        "05_recommendation_engine.py",
        "06_powerbi_exports.py",
    ]

    start_from = 0
    if len(sys.argv) > 1:
        # Allow resuming from a specific step: python run_all.py 3
        try:
            start_from = int(sys.argv[1]) - 1
        except ValueError:
            pass

    for i, script in enumerate(scripts[start_from:], start=start_from + 1):
        run_script(script)

    print("\n" + "="*70)
    print("  ALL PIPELINE SCRIPTS COMPLETE!")
    print("  Launch dashboard: streamlit run dashboard/app.py")
    print("="*70)

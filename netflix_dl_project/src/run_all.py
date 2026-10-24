"""Run the whole pipeline end to end:  python src/run_all.py"""
import subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
for script in ["01_preprocess.py", "02_split_features.py", "03_visualize.py", "04_train_mlp.py", "05_evaluate.py", "06_build_report.py"]:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, script], cwd=here, check=True)
print("\nDone. Report: outputs/Netflix_Maturity_DL_Report.pdf  | figures: figures/")

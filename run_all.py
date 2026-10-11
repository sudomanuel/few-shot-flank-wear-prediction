"""Run the full analysis in order and report what was produced.

    python run_all.py

Scripts 01-07 need the laboratory data in data/ (see data/README.md). Without it, only the external
check on the public PHM 2010 data (script 08) is run.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(ROOT, "analysis")
LAB = ["01_build_ledger.py", "02_sensor_regression.py", "03_fewshot_accuracy.py", "04_robustness.py",
       "05_breakage_hazard.py", "06_hazard_bands.py", "07_figures.py"]
PUBLIC = ["08_phm2010_check.py"]
NEEDS = {"02_sensor_regression.py": os.path.join("data", "vibration_features.csv")}


def run(script):
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(ANALYSIS, script)], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    status = "ok" if r.returncode == 0 else "FAILED"
    print("  %-26s %-7s %5.1f s" % (script, status, time.time() - t0))
    if r.returncode != 0:
        print(r.stderr[-2000:])
    return r.returncode == 0


def main():
    have_doe = os.path.exists(os.path.join(ROOT, "data", "DOE.xlsx"))
    scripts = (LAB if have_doe else []) + PUBLIC
    if not have_doe:
        print("data/DOE.xlsx not found: running only the public PHM 2010 check (see data/README.md).")
    ok = True
    for s in scripts:
        need = NEEDS.get(s)
        if need and not os.path.exists(os.path.join(ROOT, need)):
            print("  %-26s skipped (%s not found)" % (s, need))
            continue
        ok = run(s) and ok
    print("results in out/results/, figures in out/figures/" if ok else "some steps failed; see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Generate the two-hospital Synthea population.

Two independent Synthea runs model two hospital systems with genuinely
different populations (different US states, different seeds). Ages are
capped to the adult range the readmission model targets; skewing older
raises inpatient-admission yield.

Requires tools/synthea-with-dependencies.jar and a Java runtime
(tools/jdk-*/bin/java.exe is used if java is not on PATH).

Usage:
    python scripts/run_synthea.py [--population 2500]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
JAR = TOOLS / "synthea-with-dependencies.jar"
OUT_BASE = ROOT / "data_sources" / "synthea"

RUNS = [
    # (hospital, state, seed) — different states = different case mixes
    ("hospital_a", "Massachusetts", 4242),
    ("hospital_b", "Texas", 4343),
]


def find_java() -> str:
    if shutil.which("java"):
        return "java"
    for candidate in sorted(TOOLS.glob("jdk-*/bin/java.exe")):
        return str(candidate)
    raise SystemExit(
        "No Java runtime found. Install Java 11+ or unzip a Temurin JRE "
        f"under {TOOLS} (see README)."
    )


def run_one(java: str, hospital: str, state: str, seed: int,
            population: int) -> None:
    out_dir = OUT_BASE / hospital
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        java, "-jar", str(JAR),
        "-p", str(population),
        "-s", str(seed),
        "-a", "30-90",                      # adult cohort, admission-heavy
        "--exporter.csv.export", "true",
        "--exporter.fhir.export", "false",  # CSVs feed the trainer; the
                                            # app's FHIR path has its own layer
        "--exporter.hospital.fhir.export", "false",
        "--exporter.practitioner.fhir.export", "false",
        "--exporter.years_of_history", "0", # keep full history for labels
        "--exporter.baseDirectory", str(out_dir),
        state,
    ]
    print(f"[{hospital}] Synthea: {population} patients, {state}, seed {seed}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join(result.stdout.splitlines()[-5:])
    print(tail)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise SystemExit(f"Synthea failed for {hospital}")
    csv_dir = out_dir / "csv"
    if not csv_dir.exists():
        raise SystemExit(f"No CSV output at {csv_dir}")
    print(f"[{hospital}] CSVs -> {csv_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Synthea for both hospitals")
    parser.add_argument("--population", type=int, default=2500,
                        help="living patients per hospital run")
    args = parser.parse_args()

    if not JAR.exists():
        raise SystemExit(
            f"{JAR} not found. Download it from "
            "https://github.com/synthetichealth/synthea/releases"
        )

    java = find_java()
    for hospital, state, seed in RUNS:
        run_one(java, hospital, state, seed, args.population)
    print("Done. Next: python scripts/prepare_synthea.py")


if __name__ == "__main__":
    main()

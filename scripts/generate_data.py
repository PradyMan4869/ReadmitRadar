"""Generate the synthetic hospital datasets.

Run:
    python scripts/generate_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, TRAIN
from data.synthetic import generate_hospital, generate_test_set
from ml.features import TARGET


def main() -> None:
    a = generate_hospital("A", TRAIN.n_hospital_a, TRAIN.random_state)
    b = generate_hospital("B", TRAIN.n_hospital_b, TRAIN.random_state + 1)
    test = generate_test_set(TRAIN.n_test, TRAIN.random_state + 2)

    for name, df in [("hospital_a", a), ("hospital_b", b), ("test", test)]:
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{path.name}: {len(df)} records, "
              f"{df[TARGET].mean():.1%} readmission rate")


if __name__ == "__main__":
    main()

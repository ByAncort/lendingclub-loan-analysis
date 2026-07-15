import pandas as pd
import numpy as np
import os
import argparse
import gzip
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"

ACCEPTED_RAW = "accepted_2007_to_2018Q4.csv"
REJECTED_RAW = "rejected_2007_to_2018Q4.csv"
ACCEPTED_GZ = ACCEPTED_RAW + ".gz"
REJECTED_GZ = REJECTED_RAW + ".gz"


def count_lines(filepath):
    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rb") as f:
            for i, _ in enumerate(f):
                pass
    else:
        with open(filepath, "rb") as f:
            for i, _ in enumerate(f):
                pass
    return i + 1


def sample_csv(input_path, output_path, n, seed=42):
    rng = np.random.default_rng(seed)

    total = count_lines(input_path)
    n_data = total - 1
    logger.info(f"  Total data rows (est.): {n_data}")

    if n_data <= n:
        logger.warning(f"  File has fewer rows ({n_data}) than sample size ({n}), using all")
        df = pd.read_csv(input_path, low_memory=False)
    else:
        skip_data = sorted(rng.choice(n_data, n_data - n, replace=False))
        skip = [i + 1 for i in skip_data]
        df = pd.read_csv(input_path, skiprows=skip, low_memory=False)
        logger.info(f"  Sampled {len(df)} rows (target: {n})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved: {output_path} ({df.shape})")
    return df


def main(n=5000, seed=42, use_gz=False):
    interim = os.path.join(os.getcwd(), INTERIM_DIR)
    raw = os.path.join(os.getcwd(), RAW_DIR)

    accepted_src = os.path.join(raw, ACCEPTED_GZ if use_gz else ACCEPTED_RAW)
    rejected_src = os.path.join(raw, REJECTED_GZ if use_gz else REJECTED_RAW)

    sfx = f"{n // 1000}k" if n % 1000 == 0 else str(n)
    accepted_dst = os.path.join(interim, f"accepted_sample_{sfx}.csv")
    rejected_dst = os.path.join(interim, f"rejected_sample_{sfx}.csv")

    logger.info("=" * 60)
    logger.info("GENERATING SAMPLES")
    logger.info("=" * 60)
    logger.info(f"Sample size: {n}, Seed: {seed}, Use gz: {use_gz}")

    logger.info("Sampling accepted loans...")
    sample_csv(accepted_src, accepted_dst, n, seed)

    logger.info("Sampling rejected loans...")
    sample_csv(rejected_src, rejected_dst, n, seed + 1)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random samples from raw CSVs")
    parser.add_argument("-n", type=int, default=5000, help="Sample size per file (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--gz", action="store_true", help="Use .csv.gz compressed files")
    args = parser.parse_args()

    main(n=args.n, seed=args.seed, use_gz=args.gz)

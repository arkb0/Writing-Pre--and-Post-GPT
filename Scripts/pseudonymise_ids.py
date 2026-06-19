"""
Replace student_id column in one or more CSV files with deterministic pseudonyms.
Mutates each file in place.
"""

import pandas as pd
import hashlib

# List of CSVs to process
CSV_PATHS = [
    "output_1_em/corpus_summary.csv",
    "output_2_ml/corpus_summary.csv",
]
COLUMN = "student_id"

def pseudonymize(value: str, prefix: str = "SID") -> str:
    """
    Deterministically map a student_id string to a pseudonym.
    Uses SHA256 hash truncated to 8 hex chars.
    """
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}"

def process_file(path: str):
    df = pd.read_csv(path)

    if COLUMN not in df.columns:
        raise ValueError(f"Column '{COLUMN}' not found in {path}")

    # Pseudonymize student_id
    df[COLUMN] = df[COLUMN].astype(str).apply(pseudonymize)

    # Also update uid field if present
    if "uid" in df.columns:
        def update_uid(uid, sid_map):
            # uid format: canvas_id__assignment__student_id
            parts = uid.split("__")
            if len(parts) == 3:
                parts[-1] = sid_map
                return "__".join(parts)
            return uid

        df["uid"] = [
            update_uid(u, sid)
            for u, sid in zip(df["uid"], df[COLUMN])
        ]

    df.to_csv(path, index=False)
    print(f"Updated {path} with pseudonymous IDs.")

def main():
    for path in CSV_PATHS:
        try:
            process_file(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    main()

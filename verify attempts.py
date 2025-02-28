import pandas as pd
from datetime import datetime

DEFAULT_DATE = "2025-01-01"
MIN_VALID_YEAR = 1800
MAX_VALID_YEAR = 2024

def is_valid_in_range(date_val) -> bool:
    """
    Check if date_val is not NaN or the default,
    then parse as '%Y-%m-%d' and confirm year is within our valid range.
    """
    if pd.isna(date_val):
        return False
    date_str = str(date_val).strip()
    if date_str == DEFAULT_DATE:
        return False

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return MIN_VALID_YEAR <= dt.year <= MAX_VALID_YEAR
    except ValueError:
        return False

def is_non_default(date_str: str) -> bool:
    return date_str.strip() != DEFAULT_DATE

if __name__ == "__main__":
    final_file = "haunted_places_final.tsv"
    df = pd.read_csv(final_file, sep="\t")

    # 1) Fill any NaN in "Final Date" with DEFAULT_DATE, then convert to string.
    df["Final Date"] = df["Final Date"].fillna(DEFAULT_DATE).astype(str)

    total_rows = len(df)

    # 2) Identify recognized (non-default) dates
    recognized_mask = df["Final Date"].apply(is_non_default)
    recognized_count = recognized_mask.sum()

    # 3) Identify recognized dates that are valid within the year range
    valid_mask = df["Final Date"].apply(is_valid_in_range)
    valid_count = valid_mask.sum()

    print(f"\n--- Verification Results for {final_file} ---")
    print(f"Total rows: {total_rows}")
    print(f"Recognized (non-default) date rows: {recognized_count} "
          f"({recognized_count / total_rows * 100:.2f}%)")
    print(f"Valid recognized rows (year {MIN_VALID_YEAR}-{MAX_VALID_YEAR}): {valid_count} "
          f"({valid_count / total_rows * 100:.2f}%)")

    # 4) Optional sample of recognized rows
    recognized_df = df[recognized_mask]
    sample_size = min(5, len(recognized_df))
    if sample_size > 0:
        sample_df = recognized_df.sample(n=sample_size, random_state=42)
        print("\n--- Random Sample of Recognized Dates ---")
        for idx, row in sample_df.iterrows():
            print(f"\nLocation: {row['location']}")
            desc_snip = str(row['description'])[:120]
            print(f"Description snippet: {desc_snip}...")
            print(f"Extracted Date: {row['Final Date']}")
    else:
        print("\nNo recognized dates found to sample.")

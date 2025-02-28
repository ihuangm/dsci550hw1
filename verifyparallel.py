import pandas as pd
import re
import datefinder
import dateparser
import time
import wikipedia
import logging
from tqdm import tqdm
from datetime import datetime
import concurrent.futures
import numpy as np

# Allows tqdm progress bar for .apply
tqdm.pandas()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------- CONFIG ----------------
DEFAULT_DATE = "2025-01-01"
MIN_VALID_YEAR = 1800
MAX_VALID_YEAR = 2024

# If you only need ~50% coverage, you can skip or limit the broad approach
USE_BROADER_APPROACH = True
SAMPLE_MISSING_FRAC = 0.5      # Only look up 50% of missing locations
MAX_WORKERS_NARROW = 10        # concurrency for narrow lookups
MAX_WORKERS_BROAD = 8          # concurrency for broad lookups
SKIP_SHORT_LOCATIONS = True    # skip single-word or short locations
MIN_LOCATION_LENGTH = 10       # skip if < 10 characters

# Simple cache for Wikipedia results across calls
wiki_cache = {}

# -------------- HELPER FUNCTIONS --------------
def validate_year_in_range(date_str: str) -> bool:
    """
    Return True if date_str is in the valid year range.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return MIN_VALID_YEAR <= dt.year <= MAX_VALID_YEAR
    except (ValueError, TypeError):
        return False

def extract_date_from_text(text: str) -> str:
    """
    Local text extraction using datefinder + dateparser.
    Returns the earliest valid date or DEFAULT_DATE if none found.
    """
    if not isinstance(text, str) or not text.strip():
        return DEFAULT_DATE

    # datefinder pass
    df_dates = list(datefinder.find_dates(text))
    if df_dates:
        # Convert all to naive datetimes to avoid offset comparison issues
        df_dates_naive = [d.replace(tzinfo=None) for d in df_dates]
        df_dates_sorted = sorted(df_dates_naive)
        for d in df_dates_sorted:
            iso_str = d.strftime("%Y-%m-%d")
            if validate_year_in_range(iso_str):
                return iso_str

    # fallback: dateparser
    parsed = dateparser.parse(text)
    if parsed:
        parsed_naive = parsed.replace(tzinfo=None)
        iso_str = parsed_naive.strftime("%Y-%m-%d")
        if validate_year_in_range(iso_str):
            return iso_str

    return DEFAULT_DATE

def generate_narrow_search_terms(location):
    """
    Generate narrower search queries for Wikipedia.
    """
    return [
        f"{location} history",
        f"history of {location}",
        f"{location} founding",
        f"{location} historical",
        f"{location} establishment",
    ]

def narrow_wikipedia_lookup_one(location) -> str:
    """
    Narrow approach for a single location.
    Returns a single date or DEFAULT_DATE.
    """
    if location in wiki_cache:
        return wiki_cache[location]

    possible_dates = []
    for query_str in generate_narrow_search_terms(location):
        try:
            results = wikipedia.search(query_str)
            if not results:
                continue

            # limit to top 2 results to reduce overhead
            for title in results[:2]:
                try:
                    page = wikipedia.page(title)
                    # skip disambiguation
                    if "may refer to:" in page.summary.lower() or "disambiguation" in page.title.lower():
                        continue
                    found = extract_date_from_text(page.summary)
                    if found != DEFAULT_DATE:
                        possible_dates.append(found)
                except:
                    pass

            if possible_dates:
                # break early if we found something
                break
        except:
            pass

    if possible_dates:
        best_date = min(possible_dates)
        wiki_cache[location] = best_date
        return best_date
    else:
        wiki_cache[location] = DEFAULT_DATE
        return DEFAULT_DATE

def broad_wikipedia_lookup_one(location) -> str:
    """
    Broader approach for a single location.
    Returns a date or DEFAULT_DATE.
    """
    if location in wiki_cache:
        return wiki_cache[location]

    possible_dates = []
    try:
        results = wikipedia.search(location)
        if results:
            for title in results[:2]:
                try:
                    page = wikipedia.page(title)
                    if "disambiguation" in page.title.lower() or "may refer to:" in page.summary.lower():
                        continue
                    found = extract_date_from_text(page.summary)
                    if found != DEFAULT_DATE:
                        possible_dates.append(found)
                except:
                    pass
    except:
        pass

    if possible_dates:
        best_date = min(possible_dates)
        wiki_cache[location] = best_date
        return best_date
    else:
        wiki_cache[location] = DEFAULT_DATE
        return DEFAULT_DATE

def parallel_lookup(locations, lookup_func, max_workers=10, desc="Wikipedia Lookup"):
    """
    Generic parallel lookup using ThreadPoolExecutor.
    locations: list of location strings
    lookup_func: a function that takes 'location' -> 'date_str'
    """
    import concurrent.futures
    results_map = {}

    def run_one(loc):
        return loc, lookup_func(loc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_loc = {
            executor.submit(run_one, loc): loc for loc in locations
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_loc),
                           total=len(future_to_loc),
                           desc=desc):
            loc = future_to_loc[future]
            try:
                location, date_str = future.result()
                results_map[location] = date_str
            except Exception:
                results_map[loc] = DEFAULT_DATE

    return results_map

def should_skip_location(loc: str) -> bool:
    """
    Decide if we want to skip a location because it's too short,
    single-word, etc.
    """
    if not isinstance(loc, str):
        return True
    if len(loc.strip().split()) < 2:
        return True
    if len(loc.strip()) < MIN_LOCATION_LENGTH:
        return True
    return False

# -------------- PIPELINE --------------
def main():
    # CHANGE: your new input file that has the extra headers
    input_file = "haunted_places_final_parallel.tsv"  # or whatever your new file is called
    df = pd.read_csv(input_file, sep="\t")

    # The code expects 'description' and 'location' columns in df
    logging.info("Extracting local text dates from 'description'...")

    df["LocalDate"] = df["description"].progress_apply(extract_date_from_text)

    # Identify which rows are still missing a date
    logging.info("Determining which rows need Wikipedia lookup (default date).")
    mask_missing = (df["LocalDate"] == DEFAULT_DATE)
    missing_locs = df.loc[mask_missing, "location"].dropna().unique()

    if SKIP_SHORT_LOCATIONS:
        missing_locs = [loc for loc in missing_locs if not should_skip_location(loc)]

    # Sample a fraction of missing locations for narrow approach
    if SAMPLE_MISSING_FRAC < 1.0:
        missing_locs_series = pd.Series(missing_locs)
        sample_size = int(len(missing_locs_series) * SAMPLE_MISSING_FRAC)
        missing_locs_sampled = missing_locs_series.sample(sample_size, random_state=42)
    else:
        missing_locs_sampled = pd.Series(missing_locs)

    logging.info(
        f"Out of {len(missing_locs)} missing locations, "
        f"we'll do narrow Wikipedia lookups for {len(missing_locs_sampled)} of them."
    )

    if len(missing_locs_sampled) > 0:
        narrow_results = parallel_lookup(
            missing_locs_sampled,
            lookup_func=narrow_wikipedia_lookup_one,
            max_workers=MAX_WORKERS_NARROW,
            desc="Narrow Wiki Parallel"
        )
    else:
        narrow_results = {}

    # Merge narrow results
    df["NarrowWikiDate"] = df.apply(
        lambda row: narrow_results.get(row["location"], DEFAULT_DATE)
        if row["LocalDate"] == DEFAULT_DATE else row["LocalDate"],
        axis=1
    )

    if USE_BROADER_APPROACH:
        logging.info("Broad Wikipedia approach for rows still missing after narrow approach...")
        mask_still_missing = (df["NarrowWikiDate"] == DEFAULT_DATE)
        missing_locs2 = df.loc[mask_still_missing, "location"].dropna().unique()

        if SKIP_SHORT_LOCATIONS:
            missing_locs2 = [loc for loc in missing_locs2 if not should_skip_location(loc)]

        if SAMPLE_MISSING_FRAC < 1.0:
            missing_locs2_series = pd.Series(missing_locs2)
            sample_size2 = int(len(missing_locs2_series) * SAMPLE_MISSING_FRAC)
            missing_locs2_sampled = missing_locs2_series.sample(sample_size2, random_state=999)
        else:
            missing_locs2_sampled = pd.Series(missing_locs2)

        logging.info(
            f"Now {len(missing_locs2)} remain missing. "
            f"Will look up {len(missing_locs2_sampled)} in broad Wikipedia."
        )

        if len(missing_locs2_sampled) > 0:
            broad_results = parallel_lookup(
                missing_locs2_sampled,
                lookup_func=broad_wikipedia_lookup_one,
                max_workers=MAX_WORKERS_BROAD,
                desc="Broad Wiki Parallel"
            )
        else:
            broad_results = {}

        df["BroadWikiDate"] = df.apply(
            lambda row: broad_results.get(row["location"], DEFAULT_DATE)
            if row["NarrowWikiDate"] == DEFAULT_DATE else row["NarrowWikiDate"],
            axis=1
        )
        df["FinalDate"] = df["BroadWikiDate"]
    else:
        df["FinalDate"] = df["NarrowWikiDate"]

    # (Optional) Quick stats
    total = len(df)
    recognized = df[df["FinalDate"] != DEFAULT_DATE]
    recognized_count = len(recognized)
    valid_recognized = recognized[recognized["FinalDate"].apply(validate_year_in_range)]
    valid_recognized_count = len(valid_recognized)

    logging.info("========== SUMMARY STATS ==========")
    logging.info(f"Total rows: {total}")
    logging.info(f"Recognized date rows: {recognized_count} ({recognized_count / total * 100:.2f}%)")
    logging.info(f"Valid recognized rows (within {MIN_VALID_YEAR}-{MAX_VALID_YEAR}): "
                 f"{valid_recognized_count} ({valid_recognized_count / total * 100:.2f}%)")

    # CHANGE: new output filename
    output_file = "haunted_places_final_parallel_4000.tsv"
    df.to_csv(output_file, sep="\t", index=False)
    logging.info(f"Saved final results to {output_file}")


if __name__ == "__main__":
    main()

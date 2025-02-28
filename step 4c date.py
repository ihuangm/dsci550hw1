import pandas as pd
import re
import datefinder
import dateparser
import time
import wikipedia
from tqdm import tqdm
from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON

# ----- CONFIGURATION -----
DEFAULT_DATE = "2025-01-01"  # Fallback date when no date is found
MIN_VALID_YEAR = 1800         # Adjust as appropriate
MAX_VALID_YEAR = 2024

# ----- HELPER FUNCTIONS FOR DATE EXTRACTION -----
def extract_date_from_text(text: str) -> str:
    """
    Tries to extract a date from text using regex patterns and datefinder.
    Returns a date string in 'YYYY-MM-DD' format or the default date if nothing is found.
    """
    if pd.isna(text) or not text.strip():
        return DEFAULT_DATE

    text_lower = text.lower()

    # Pattern 1: Ordinal dates (e.g., "October 3rd, 1995")
    ordinal_match = re.search(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        text_lower)
    if ordinal_match:
        try:
            dt = pd.to_datetime(ordinal_match.group(0), errors="coerce")
            if pd.notnull(dt):
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Pattern 2: "Month YYYY" (e.g., "March 2020")
    month_year_match = re.search(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
        text_lower)
    if month_year_match:
        try:
            dt = pd.to_datetime(month_year_match.group(0), format="%B %Y", errors="coerce")
            if pd.notnull(dt):
                # Default to first day of month
                return dt.strftime("%Y-%m-01")
        except Exception:
            pass

    # Pattern 3: Explicit phrases like "in 1995", "since 1995"
    explicit_year_match = re.search(r"\b(?:in|since|from)\s+(\d{4})\b", text_lower)
    if explicit_year_match:
        return f"{explicit_year_match.group(1)}-01-01"

    # Pattern 4: Standalone 4-digit year
    year_match = re.search(r"\b(1[8-9]\d{2}|20\d{2})\b", text_lower)
    if year_match:
        return f"{year_match.group(1)}-01-01"

    # Fallback: Use datefinder
    matches = list(datefinder.find_dates(text_lower))
    if matches:
        return matches[0].strftime("%Y-%m-%d")

    return DEFAULT_DATE

# ----- FUNCTIONS FOR EXTERNAL LOOKUPS -----
def run_wikipedia_approach(df):
    """
    For rows where the initial extraction yielded the default date,
    query Wikipedia (per unique location) and try to extract a date from the summary.
    """
    missing_mask = df["Haunted Places Date"] == DEFAULT_DATE
    missing_locations = df.loc[missing_mask, "location"].dropna().unique()
    print(f"[Wikipedia] {len(missing_locations)} unique locations need lookup.")

    wiki_rows = []
    for loc in tqdm(missing_locations, desc="Wikipedia Searching"):
        try:
            time.sleep(0.2)
            results = wikipedia.search(loc)
            if not results:
                wiki_rows.append({"location": loc, "wiki_date": DEFAULT_DATE})
            else:
                page = wikipedia.page(results[0])
                summary = page.summary
                wiki_date = extract_date_from_text(summary)
                wiki_rows.append({"location": loc, "wiki_date": wiki_date})
        except Exception:
            wiki_rows.append({"location": loc, "wiki_date": DEFAULT_DATE})

    df_wiki = pd.DataFrame(wiki_rows)
    df = df.merge(df_wiki, on="location", how="left")
    # Use Wikipedia date if initial extraction was default
    df["Final Date"] = df.apply(
        lambda row: row["wiki_date"] if row["Haunted Places Date"] == DEFAULT_DATE else row["Haunted Places Date"],
        axis=1
    )
    print("[Wikipedia] Completed lookups.")
    return df

def query_dbpedia(place_name: str) -> str:
    """
    Query DBpedia for a founding date (or similar historical date) of a location.
    """
    endpoint = "https://dbpedia.org/sparql"
    sparql = SPARQLWrapper(endpoint)
    query_str = f"""
    SELECT ?date WHERE {{
      ?place rdfs:label "{place_name}"@en .
      ?place dbo:foundingDate ?date .
    }}
    LIMIT 1
    """
    sparql.setQuery(query_str)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if bindings:
            return extract_date_from_text(bindings[0]["date"]["value"])
        return DEFAULT_DATE
    except Exception:
        return DEFAULT_DATE

def run_dbpedia_approach(df):
    """
    For any remaining rows with the default date, query DBpedia for a date.
    """
    missing_mask = df["Final Date"] == DEFAULT_DATE
    missing_locations = df.loc[missing_mask, "location"].dropna().unique()
    print(f"[DBpedia] {len(missing_locations)} unique locations need DBpedia lookup.")

    dbpedia_rows = []
    for loc in tqdm(missing_locations, desc="DBpedia SPARQL"):
        db_date = query_dbpedia(loc)
        dbpedia_rows.append({"location": loc, "dbpedia_date": db_date})
    df_dbpedia = pd.DataFrame(dbpedia_rows)
    df = df.merge(df_dbpedia, on="location", how="left")
    df["Final Date"] = df.apply(
        lambda row: row["dbpedia_date"] if row["Final Date"] == DEFAULT_DATE else row["Final Date"],
        axis=1
    )
    print("[DBpedia] Completed DBpedia lookups.")
    return df

# ----- FINAL VALIDATION FUNCTION -----
def validate_dates(df):
    """
    Print summary statistics on date extraction and flag suspicious years.
    """
    total = len(df)
    recognized = df[df["Final Date"] != DEFAULT_DATE]
    recognized_count = len(recognized)
    try:
        # Compute valid recognized rows (year within our range)
        valid_recognized = recognized[recognized["Final Date"].apply(lambda d: MIN_VALID_YEAR <= datetime.strptime(d, "%Y-%m-%d").year <= MAX_VALID_YEAR)]
        valid_recognized_count = len(valid_recognized)
    except Exception as e:
        valid_recognized_count = recognized_count

    print("\n--- FINAL DATE EXTRACTION STATS ---")
    print(f"Total rows: {total}")
    print(f"Rows with recognized dates (not default): {recognized_count} ({(recognized_count/total)*100:.2f}%)")
    print(f"Rows with valid recognized dates (within {MIN_VALID_YEAR}-{MAX_VALID_YEAR}): {valid_recognized_count} ({(valid_recognized_count/total)*100:.2f}%)")

    # Optionally, show a random sample for manual review
    sample = recognized.sample(n=min(5, recognized_count), random_state=42)
    print("\n--- Sample Recognized Dates ---")
    for idx, row in sample.iterrows():
        print(f"Location: {row['location']}")
        print(f"Description snippet: {row['description'][:150]}...")
        print(f"Extracted Date: {row['Final Date']}\n")

# ----- MAIN PIPELINE -----
def main():
    # Step 1: Convert CSV to TSV (if not already done)
    csv_input = "../data/haunted_places.csv"  # adjust as needed
    tsv_initial = "haunted_places_initial.tsv"
    df = pd.read_csv(csv_input)
    df.to_csv(tsv_initial, sep="\t", index=False)
    print(f"✅ Converted CSV to TSV: {tsv_initial}")

    # Step 2: Add basic date extraction using regex/datefinder/dateparser
    print("\n📆 Extracting dates from descriptions...")
    df["Haunted Places Date"] = df["description"].apply(extract_date_from_text)
    print(f"✅ Basic extraction complete. Recognized dates: {(df['Haunted Places Date'] != DEFAULT_DATE).sum()}")

    # Step 3: For rows with default date, try Wikipedia lookup
    df = run_wikipedia_approach(df)

    # Step 4: For any remaining defaults, try DBpedia lookup
    df = run_dbpedia_approach(df)

    # Save final dataset
    final_output = "haunted_places_final.tsv"
    df.to_csv(final_output, sep="\t", index=False)
    print(f"\n✅ Final dataset saved as {final_output}")

    # Final validation and stats
    validate_dates(df)

if __name__ == "__main__":
    main()

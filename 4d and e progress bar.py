import spacy
import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.load("en_core_web_sm")

# Expanded synonyms for each time of day
MORNING_SYNONYMS = {
    "morning", "am", "sunrise", "dawn", "early", "daybreak", "noon", "afternoon"
}

EVENING_SYNONYMS = {
    "evening", "night", "pm", "nightfall", "late", "midnight", "wee", "overnight", "mid-night"
}

DUSK_SYNONYMS = {
    "dusk", "twilight", "sunset"
}

def parse_time_of_day(text: str) -> str:
    """
    Return "Morning", "Evening", or "Dusk" if we find synonyms.
    Else "Unknown".
    """
    if not text or not text.strip():
        return "Unknown"
    text_lower = text.lower()
    tokens = text_lower.split()

    # Priority: check MORNING first, then DUSK, then EVENING
    # because the text might say "late morning" or "morning and night".
    # Adjust as desired if you want a different priority.

    # 1) Check MORNING
    if any(word in tokens for word in MORNING_SYNONYMS):
        return "Morning"

    # 2) Check DUSK
    if any(word in tokens for word in DUSK_SYNONYMS):
        return "Dusk"

    # 3) Check EVENING
    if any(word in tokens for word in EVENING_SYNONYMS):
        return "Evening"

    return "Unknown"


# -------------- Example advanced witness function (unchanged) --------------

FUZZY_NUMBERS = {
    "some": 3,
    "several": 4,
    "few": 2,
    "couple": 2,
    "dozen": 12,
    "handful": 5,
    "many": 10,
    "multiple": 4,
    "numerous": 10
}

WITNESS_KEYWORDS = {
    "people", "witness", "witnesses", "persons", "locals", "onlookers",
    "visitors", "guests", "employees", "kids", "children", "folks", "crowd"
}

WITNESS_VERBS = {
    "see", "saw", "spotted", "witnessed", "reported", "claimed", "encountered",
    "viewed", "observed"
}

IGNORE_CONTEXT = {"kill", "killed", "died", "dying", "dead", "death", "accident", "burned"}

def advanced_witness_count(text: str) -> int:
    if not text:
        return 0
    doc = nlp(text)

    numeric_val = check_cardinals(doc)
    if numeric_val > 0:
        return numeric_val

    guess_val = fuzzy_witness_synonyms(doc)
    if guess_val > 0:
        return guess_val

    if has_witness_mention(doc):
        return 1

    return 0

def check_cardinals(doc) -> int:
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            try:
                val = int(ent.text)
            except ValueError:
                continue
            ent_i = ent.end - 1

            if death_context_nearby(doc, ent_i):
                continue

            # check up to 2 tokens ahead
            for forward_steps in [1, 2]:
                if ent_i + forward_steps < len(doc):
                    nxt_lemma = doc[ent_i + forward_steps].lemma_.lower()
                    if nxt_lemma in WITNESS_KEYWORDS:
                        return val

            # check up to 2 tokens behind
            for backward_steps in [1, 2]:
                if ent_i - backward_steps >= 0:
                    prev_lemma = doc[ent_i - backward_steps].lemma_.lower()
                    if prev_lemma in WITNESS_KEYWORDS:
                        return val
    return 0

def death_context_nearby(doc, idx, window=5) -> bool:
    end_idx = min(len(doc), idx + window + 1)
    for i in range(idx + 1, end_idx):
        if doc[i].lemma_.lower() in IGNORE_CONTEXT:
            return True
    return False

def fuzzy_witness_synonyms(doc) -> int:
    lemmas = [t.lemma_.lower() for t in doc]
    length = len(lemmas)
    for i, lemma in enumerate(lemmas):
        if lemma in FUZZY_NUMBERS:
            if i + 1 < length and lemmas[i + 1] in WITNESS_KEYWORDS:
                return FUZZY_NUMBERS[lemma]
            if i - 1 >= 0 and lemmas[i - 1] in WITNESS_KEYWORDS:
                return FUZZY_NUMBERS[lemma]
    return 0

def has_witness_mention(doc) -> bool:
    for token in doc:
        if token.lemma_.lower() in WITNESS_KEYWORDS:
            return True
        if token.lemma_.lower() in WITNESS_VERBS:
            return True
    return False


# -------------- MAIN PIPELINE --------------
def main():
    df = pd.read_csv("haunted_places_with_evidence.tsv", sep="\t")

    print("Extracting witness counts with advanced hybrid approach...")
    df["Haunted Places Witness Count"] = df["description"].progress_apply(advanced_witness_count)

    print("Extracting time of day (Evening, Morning, Dusk) from text...")
    df["Time of Day"] = df["description"].progress_apply(parse_time_of_day)

    total = len(df)
    nonzero = (df["Haunted Places Witness Count"] > 0).sum()
    print(f"\nWITNESS COUNT > 0: {nonzero} out of {total} ({nonzero / total * 100:.2f}%)")

    time_counts = df["Time of Day"].value_counts()
    print("\n--- TIME OF DAY DISTRIBUTION ---")
    print(time_counts)

    df.to_csv("haunted_places_with_evidence.tsv", sep="\t", index=False)
    print("\nFinished! Updated 'haunted_places_with_evidence.tsv' saved.")


if __name__ == "__main__":
    main()

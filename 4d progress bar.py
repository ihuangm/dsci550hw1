import spacy
import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.load("en_core_web_sm")

# Words that indicate some group of people, but no exact number:
# choose your own numeric guess
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

# People synonyms to match if we see fuzzy expressions or cardinal
WITNESS_KEYWORDS = {
    "people", "witness", "witnesses", "persons", "locals", "onlookers",
    "visitors", "guests", "employees", "kids", "children", "folks", "crowd"
}

# Verbs that imply a “witnessing” action
WITNESS_VERBS = {
    "see", "saw", "spotted", "witnessed", "reported", "claimed", "encountered",
    "viewed", "observed"
}

# Words to skip if found near the cardinal => it’s not a witness count
IGNORE_CONTEXT = {"kill", "killed", "died", "dying", "dead", "death", "accident", "burned"}

def advanced_witness_count(text: str) -> int:
    if not text:
        return 0
    doc = nlp(text)

    # 1) Attempt: “CARDINAL + WITNESS_KEYWORDS” within 2 tokens, skipping death context.
    numeric_val = check_cardinals(doc)
    if numeric_val > 0:
        return numeric_val

    # 2) Attempt: Fuzzy synonyms (“some people,” “a few guests,” etc.)
    guess_val = fuzzy_witness_synonyms(doc)
    if guess_val > 0:
        return guess_val

    # 3) If we see a mention of any WITNESS_KEYWORDS or WITNESS_VERBS, default to 1
    if has_witness_mention(doc):
        return 1

    return 0

def check_cardinals(doc) -> int:
    """
    Look for numeric entities (CARDINAL).
    If found near a WITNESS_KEYWORD in +/-2 tokens, parse it.
    If death-related words are near the cardinal, skip it.
    Return the first match or 0 if none found.
    """
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            # parse e.g. "10"
            try:
                val = int(ent.text)
            except ValueError:
                continue

            # ent.end - 1 is the token index
            ent_i = ent.end - 1

            # skip if “kill/died” words appear near ent
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
    """
    Look up to ‘window’ tokens ahead of idx.
    If we find words like "killed", "dead", "died", skip it (it’s not a witness).
    """
    end_idx = min(len(doc), idx + window + 1)
    for i in range(idx + 1, end_idx):
        if doc[i].lemma_.lower() in IGNORE_CONTEXT:
            return True
    return False

def fuzzy_witness_synonyms(doc) -> int:
    """
    E.g. "some people", "few locals", "several visitors", "a dozen onlookers" => parse a guessed integer.
    Return the largest match if multiple occur, or the first?
    """
    # Turn doc into list of lemmas to parse more easily
    lemmas = [t.lemma_.lower() for t in doc]
    length = len(lemmas)

    for i, lemma in enumerate(lemmas):
        if lemma in FUZZY_NUMBERS:
            # next or prev token is in WITNESS_KEYWORDS?
            # e.g. “some + people”
            # "some" might be doc[i], next token is doc[i+1] => "people"
            # or reversed: "people some"? probably not used, but possible
            # If  i+1 < length => check next token
            if i + 1 < length and lemmas[i + 1] in WITNESS_KEYWORDS:
                return FUZZY_NUMBERS[lemma]
            if i - 1 >= 0 and lemmas[i - 1] in WITNESS_KEYWORDS:
                return FUZZY_NUMBERS[lemma]

    return 0

def has_witness_mention(doc) -> bool:
    """
    Check if the text has any witness-related nouns or verbs at all.
    If so, we’ll default to 1 if no numeric or fuzzy expression was found.
    """
    for token in doc:
        # e.g. "people", "witnesses" in WITNESS_KEYWORDS
        if token.lemma_.lower() in WITNESS_KEYWORDS:
            return True
        # or a verb in WITNESS_VERBS
        if token.lemma_.lower() in WITNESS_VERBS:
            return True
    return False

def main():
    df = pd.read_csv("haunted_places_with_evidence.tsv", sep="\t")
    print("Extracting witness counts with advanced hybrid approach...")

    df["Haunted Places Witness Count"] = df["description"].progress_apply(advanced_witness_count)

    total = len(df)
    nonzero = (df["Haunted Places Witness Count"] > 0).sum()
    print(f"\nWITNESS COUNT > 0: {nonzero} out of {total} ({nonzero / total * 100:.2f}%)")

    # Show top 5 highest witness counts
    print("\n--- TOP 5 HIGHEST WITNESS COUNTS ---")
    top_5 = df.nlargest(5, "Haunted Places Witness Count")
    print(top_5[["city", "location", "description", "Haunted Places Witness Count"]])

    # Show top 5 lowest witness counts
    print("\n--- TOP 5 LOWEST WITNESS COUNTS ---")
    bottom_5 = df.nsmallest(5, "Haunted Places Witness Count")
    print(bottom_5[["city", "location", "description", "Haunted Places Witness Count"]])

    df.to_csv("haunted_places_with_evidence.tsv", sep="\t", index=False)
    print("\nFinished! Updated 'haunted_places_with_evidence.tsv' saved.")

if __name__ == "__main__":
    main()

import pandas as pd
import re
import nltk
import requests
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download NLTK datasets (for Synonyms and Tenses)
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the dataset (Convert the original CSV file to TSV at first)
df = pd.read_csv("haunted_places.csv")
df.to_csv("haunted_places.tsv", sep="\t", index=False)
df = pd.read_csv("haunted_places.tsv", sep="\t")


# Function to generate synonyms
def get_synonyms(word_list):
    synonyms = set()
    for word in word_list:
        synonyms.add(lemmatizer.lemmatize(word))
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemmatizer.lemmatize(lemma.name().lower().replace('_', ' ')))
    return synonyms


# Keywords for Q4(f)
ghost_keywords = get_synonyms(["ghost", "spirit", "phantom", "specter", "poltergeist", "haunt", "apparition", "figure",
                               "wraith", "shade", "entity", "presence", "shadow", "soul"])
orb_keywords = get_synonyms(["orb", "light ball", "floating light", "glowing sphere", "luminescence", "halo",
                             "radiance", "aura"])
ufo_keywords = get_synonyms(["ufo", "unidentified flying object", "flying saucer", "extraterrestrial craft",
                             "alien ship", "spaceship", "disk", "craft"])
uap_keywords = get_synonyms(["uap", "unidentified aerial phenomenon", "mystery aircraft", "unknown object",
                             "anomaly", "unexplained light"])
male_keywords = get_synonyms(["man", "male", "gentleman", "boy", "father", "husband", "monk", "lord", "duke"])
female_keywords = get_synonyms(["woman", "female", "lady", "girl", "mother", "wife", "nun", "queen", "princess"])
child_keywords = get_synonyms(["child", "boy", "girl", "kid", "baby", "toddler", "infant", "orphan"])
group_keywords = get_synonyms(["several", "many", "group", "crowd", "multiple", "host", "gathering", "horde"])

# Keywords for Q4(g)
base_keywords = {
    "murder": ["murder", "kill", "homicide", "slay", "stab", "shoot", "crime", "assassinate", "massacre", "butcher"],
    "death": ["die", "death", "corpse", "dead", "pass away", "suicide", "grave", "bury", "lifeless", "perish",
              "deceased"],
    "supernatural": ["ghost", "haunt", "apparition", "spirit", "phantom", "poltergeist", "witch", "curse", "demon",
                     "supernatural", "possession", "shadow figure", "specter", "mystery", "unknown", "disappearance",
                     "unexplained", "legend", "myth", "folklore"],
}
murder_keywords = get_synonyms(base_keywords["murder"])
death_keywords = get_synonyms(base_keywords["death"])
supernatural_keywords = get_synonyms(base_keywords["supernatural"])


# Function to categorize apparition type
def categorize_apparition(description):
    description = str(description).lower()
    words = [lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b', description)]

    type_I_matches = []
    type_II = "N/A"
    type_III = "N/A"

    if any(word in orb_keywords for word in words):
        type_I_matches.append("Orb")
    if any(word in ufo_keywords for word in words):
        type_I_matches.append("UFO")
    if any(word in uap_keywords for word in words):
        type_I_matches.append("UAP")
    if any(word in ghost_keywords for word in words):
        type_I_matches.append("Ghost")

    type_I = ", ".join(type_I_matches) if type_I_matches else "Unknown"

    if "Ghost" in type_I_matches:
        has_multiple = any(word in group_keywords for word in words)
        type_II = "Several Ghosts" if has_multiple else "Single Ghost"

        ghost_positions = [i for i, word in enumerate(words) if word in ghost_keywords]
        male_positions = [i for i, word in enumerate(words) if word in male_keywords]
        female_positions = [i for i, word in enumerate(words) if word in female_keywords]
        child_positions = [i for i, word in enumerate(words) if word in child_keywords]

        has_male = any(abs(m - g) <= 4 for g in ghost_positions for m in male_positions)
        has_female = any(abs(f - g) <= 4 for g in ghost_positions for f in female_positions)
        has_child = any(abs(c - g) <= 4 for g in ghost_positions for c in child_positions)

        type_III = []
        if has_male:
            type_III.append("Male Ghost")
        if has_female:
            type_III.append("Female Ghost")
        if has_child:
            type_III.append("Child Ghost")

        type_III = ", ".join(type_III) if type_III else "Unknown"

    return type_I, type_II, type_III


# Function to categorize event type
def categorize_event(description):
    description_lower = str(description).lower()
    description_tokens = [lemmatizer.lemmatize(word) for word in description_lower.split()]

    murder_match = any(word in description_tokens for word in murder_keywords)
    death_match = any(word in description_tokens for word in death_keywords)
    supernatural_match = any(word in description_tokens for word in supernatural_keywords)

    if murder_match and death_match:
        return "Murder, Death"
    elif murder_match and supernatural_match:
        return "Murder, Supernatural"
    elif death_match and supernatural_match:
        return "Death, Supernatural"
    elif murder_match:
        return "Murder"
    elif death_match:
        return "Death"
    elif supernatural_match:
        return "Supernatural Phenomenon"

    return "Unknown"


def WebParse():
    # Q4(h) Access to the website
    url = 'https://drugabusestatistics.org/alcohol-abuse-statistics/'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.text
    except Exception:
        raise Exception("Error: Not able to access to the website")

    # Extract Relevant Data
    abuse_percentage_data = {}

    if content:
        pattern = r'(\d+\.\d+)% of ([A-Za-z\s]+) adults over 18 binge drink at least once per month\.'
        matches = re.findall(pattern, content)
        abuse_percentage_data = {state.strip(): float(percentage) for percentage, state in matches}

    if "state" not in df.columns:
        raise Exception("Error: Not able to find State Column")
    else:
        df["state"] = df["state"].astype(str).str.strip()
        df["Alcohol Abuse"] = df["state"].apply(
            lambda state: abuse_percentage_data.get(state, "Unknown") if state else "Unknown"
        )


# Save the final dataset as TSV file
df["Apparition Type I"], df["Apparition Type II"], df["Apparition Type III"] = \
    zip(*df["description"].apply(categorize_apparition))
df["Event Type"] = df["description"].apply(categorize_event)
WebParse()
df.to_csv("result_haunted_places.tsv", sep="\t", index=False)
print("Finished")

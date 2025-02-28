import pandas as pd

AUDIO_KEYWORDS = [
     "sobbing", "weeping", "humming", "buzzing", "thumping", "tapping", "creaking", "rustling", "scratching", "growling", "hissing", "roaring", "ringing", "ticking",
    "unexplained noise", "strange sound", "eerie sound", "faint noise", "disembodied voice", "phantom sounds", "EVP",
    "bells", "piano", "static", "echoing", "muffled", "unnatural silence","noises", "sounds", "voices", "whispers", "screams", "cries", "moans", "groans", "laughter", "giggling", "footsteps", "knocking", "banging", "clanking", "rattling", "slamming", "splashing", "music", "singing", "applause", "gunshots",
    "chanting", "yelling", "shouting"]
IMAGE_KEYWORDS = [
     "observed", "glimpse", "shape", "form", "outline", "silhouette", "transparent figure", "translucent figure",
    "dematerializing", "disappearing", "vanishing", "floating figure", "hovering figure",
    "photograph", "video", "film", "image", "picture","apparition", "figure", "shadowy figure", "shadow", "ghost", "specter", "spectre", "phantom", "wraith", "shade", "spirit", "entity", "ectoplasm",
    "orb", "mist", "glowing", "glowing eyes", "red eyes", "unexplained light", "strange light", "flickering light", "flashing light",
    "seen", "sighting", "witnessed"]



def has_audio_evidence(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in AUDIO_KEYWORDS)


def has_image_evidence(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in IMAGE_KEYWORDS)


def add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Create the boolean columns by scanning "description"
    df["Audio Evidence"] = df["description"].apply(has_audio_evidence)
    df["Image/Video/Visual Evidence"] = df["description"].apply(has_image_evidence)

    # Find the position (index) of the "Haunted Places Date" column
    date_col_index = df.columns.get_loc("Haunted Places Date")

    # Move these two new columns so they appear
    # immediately to the LEFT of "Haunted Places Date"
    # 1) Save the newly created Series
    audio_series = df.pop("Audio Evidence")
    image_series = df.pop("Image/Video/Visual Evidence")

    # 2) Insert them back in, in the right spot
    df.insert(date_col_index, "Audio Evidence", audio_series)
    df.insert(date_col_index + 1, "Image/Video/Visual Evidence", image_series)

    return df


# ---------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Suppose you already have a DataFrame "df" with columns
    # including "description" and "Haunted Places Date".
    # For example:
    df = pd.read_csv("haunted_places_final_simplified.tsv", sep="\t")

    df = add_evidence_columns(df)

    # Now df has "Audio Evidence" and "Image/Video/Visual Evidence"
    # inserted to the left of the "Haunted Places Date" column.
    df.to_csv("haunted_places_with_evidence.tsv", sep="\t", index=False)

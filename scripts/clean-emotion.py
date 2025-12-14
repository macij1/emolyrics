import pandas as pd
import matplotlib.pyplot as plt
import re

# Configuration
INPUT_FILE = "data/spotify_dataset.csv"
OUTPUT_FILE = "data/spotify_emotion_clean.csv"

TEXT_COL = "lyrics"
CLASS_COL = "emotion"    # class column
ARTIST_COL = "artist"
SONG_COL = "song_title"

# Minimum / maximum text length
MIN_TEXT_LEN = 30
MAX_TEXT_LEN = 10_000

# Minimum number of samples per class to keep
MIN_SAMPLES_PER_CLASS = 5000


def plot_emotion_distribution(df: pd.DataFrame, title: str) -> None:
    """Plot the distribution of the emotion classes."""
    counts = (
        df[CLASS_COL]
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(CLASS_COL)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def load_and_standardize(path: str = INPUT_FILE) -> pd.DataFrame:
    """Read CSV and standardize key column names."""
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip"
    )

    print("Loaded dataset with shape:", df.shape)

    rename_map = {
        "Artist(s)": ARTIST_COL,
        "song": SONG_COL,
        "text": TEXT_COL,
        # we keep "emotion" as-is
    }

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = [TEXT_COL, CLASS_COL, ARTIST_COL, SONG_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after renaming: {missing}")

    return df


def remove_empty_lyrics(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Remove rows with empty or missing lyrics."""
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    mask = df[TEXT_COL].str.strip() != ""
    df = df[mask]
    print("After removing empty lyrics:", df.shape)
    plot_emotion_distribution(df, "After removing empty lyrics")
    return df


def remove_duplicate_artist_song(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Remove duplicate artist+song pairs."""
    df = df.copy()
    before = df.shape[0]
    df = df.drop_duplicates(subset=[ARTIST_COL, SONG_COL])
    after = df.shape[0]
    print(f"After removing duplicate artist+song pairs: {df.shape} "
          f"(removed {before - after} rows)")
    plot_emotion_distribution(df, "After removing duplicate artist+song pairs")
    return df

def strip_section_tags(df: pd.DataFrame) -> pd.DataFrame:
    SECTION_TAG_RE = re.compile(r"\[[^\]]+\]")
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].str.replace(SECTION_TAG_RE, " ", regex=True)
    df[TEXT_COL] = df[TEXT_COL].str.replace(r"\s+", " ", regex=True).str.strip()
    print("After stripping [Intro]/[Verse]/[Chorus] tags:", df.shape)
    plot_emotion_distribution(df, "After stripping section tags")
    return df

def filter_by_text_length(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Remove very short (<MIN_TEXT_LEN) or very long (>MAX_TEXT_LEN) lyrics."""
    df = df.copy()
    lengths = df[TEXT_COL].str.len()
    mask = (lengths >= MIN_TEXT_LEN) & (lengths <= MAX_TEXT_LEN)
    df = df[mask]
    print(f"After filtering by text length [{MIN_TEXT_LEN}, {MAX_TEXT_LEN}]:", df.shape)
    plot_emotion_distribution(df, "After filtering by text length")
    return df


def remove_rare_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Remove classes with very few samples.

    You define what “very few” means via MIN_SAMPLES_PER_CLASS.
    """
    df = df.copy()
    counts = df[CLASS_COL].value_counts()
    keep_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    discarded_classes = counts[counts < MIN_SAMPLES_PER_CLASS].index
    print(f"Discarded classes: {discarded_classes}")
    df = df[df[CLASS_COL].isin(keep_classes)]
    print("After removing rare classes:", df.shape)
    plot_emotion_distribution(df, "After removing rare classes")
    return df


def normalize_emotion_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize specific emotion label variants to standardized forms.

    - "Love"  -> "love"
    - "angry" -> "anger"
    """
    df = df.copy()

    mapping = {
        "Love": "love",
        "angry": "anger",
    }

    df[CLASS_COL] = df[CLASS_COL].replace(mapping)
    print("After normalizing emotion labels:")
    print(df[CLASS_COL].value_counts())
    plot_emotion_distribution(df, "After normalizing emotion labels")

    return df

def remove_specific_artist_songs(df: pd.DataFrame, artist_name: str) -> pd.DataFrame:
    df = df.copy()
    df = df[df[ARTIST_COL] != artist_name]
    print("After removing specific artist's songs:", df.shape)
    plot_emotion_distribution(df, "After removing specific artist's songs")
    return df

def main():
    # Load + standardize
    df = load_and_standardize(INPUT_FILE)
    plot_emotion_distribution(df, "Original distribution")

    # Ensure emotion is not missing
    df = df.dropna(subset=[CLASS_COL])
    print("After dropping rows with missing emotion:", df.shape)
    plot_emotion_distribution(df, "After dropping missing emotion")

    # Normalize specific emotion labels
    df = normalize_emotion_labels(df)

    # 1. Remove empty lyrics
    df = remove_empty_lyrics(df)

    # 2. Remove duplicate artist+song pairs
    df = remove_duplicate_artist_song(df)

    # 3. Remove specific artist's songs
    df = remove_specific_artist_songs(df, "L.A.B.")

    # 4. Strip section tags
    df = strip_section_tags(df)

    # 5. Remove very short/very long texts
    df = filter_by_text_length(df)

    # 6. Remove classes with very few samples
    df = remove_rare_classes(df)

    # Save cleaned dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )
    print("Saved cleaned dataset to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
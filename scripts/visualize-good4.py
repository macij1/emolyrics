import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_FILE = "spotify_labeled_clean.csv"
TEXT_COL = "text"
ARTIST_COL = "Artist(s)"
SONG_COL = "song"
GENRE_COL = "Genre"   # <-- change if your dataset uses a different column name

# Label columns used in your project
LABEL_COLS = [
    "Good for Party",
    "Good for Work/Study",
    "Good for Relaxation/Meditation",
    "Good for Exercise",
    "Good for Running",
    "Good for Yoga/Stretching",
    "Good for Driving",
    "Good for Social Gatherings",
    "Good for Morning Routine"
]


def main():

    # === LOAD DATASET ===
    df = pd.read_csv(CSV_FILE)
    print("Loaded CSV with shape:", df.shape)

    # ======================================================
    # 1) DISTRIBUTION OF GOOD-FOR LABELS
    # ======================================================
    label_counts = df[LABEL_COLS].sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    label_counts.plot(kind="bar")
    plt.title("Distribution of 'Good-For' Labels")
    plt.ylabel("Number of Songs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ======================================================
    # 2) DISTRIBUTION OF ARTISTS (TOP 30)
    # ======================================================
    artist_counts = df[ARTIST_COL].value_counts().head(50)

    plt.figure(figsize=(10, 8))
    artist_counts.plot(kind="bar")
    plt.title("Top 50 Artists by Number of Songs")
    plt.ylabel("Number of Songs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ======================================================
    # 3) DISTRIBUTION OF GENRES
    # ======================================================
    if GENRE_COL in df.columns:
        genre_counts = (
            df[GENRE_COL]
            .astype(str)
            .str.split(",")              # handle "Hip-Hop, Rap" lists
            .explode()                   # one genre per row
            .str.strip()
            .value_counts()
        )

        plt.figure(figsize=(12, 6))
        genre_counts.head(30).plot(kind="bar")
        plt.title("Top 30 Genres")
        plt.ylabel("Number of Songs")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ WARNING: Column '{GENRE_COL}' not found. Skipping genre plot.")


if __name__ == "__main__":
    main()

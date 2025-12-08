import pandas as pd

df = pd.read_csv(
    '../archive/spotify_dataset.csv',
    engine='python',
    on_bad_lines='skip'
)

print("Loaded dataset with shape:", df.shape)

''' 39 columns:
"Artist(s)":string"!!!"
"song":string"Even When the Waters Cold"
"text":string"Friends told her she was better off at the bottom of a river Than in a bed with him He said "Until you try both, you won't know what you like better Why don't we go for a swim?" Well, friends told her this and friends told her that But friends don't choose what echoes in your head When she got bored with all the idle chit-and-chat Kept thinking 'bout what he said I'll swim even when the water's cold That's the one thing that I know Even when the water's cold She remembers it fondly, she doesn't remember it all But what she does, she sees clearly She lost his number, and he never called But all she really lost was an earring The other's in a box with others she has lost I wonder if she still hears me I'll swim even when the water's cold That's the one thing that I know Even when the water's cold If you believe in love You know that sometimes it isn't Do you believe in love? Then save the bullshit questions Sometimes it is and sometimes it isn't Sometimes it's just how the light hits their eyes Do you believe in love?"
"Length":string"03:47"
"emotion":string"sadness"
"Genre":string"hip hop"
"Album":string"Thr!!!er"
"Release Date":string"2013-04-29"
"Key":string"D min"
"Tempo":float0.4378698225
"Loudness (db)":float0.785065407
"Time signature":string"4/4"
"Explicit":string"No"
"Popularity":string"40"
"Energy":string"83"
"Danceability":string"71"
"Positiveness":string"87"
"Speechiness":string"4"
"Liveness":string"16"
"Acousticness":string"11"
"Instrumentalness":string"0"
"Good for Party":int0
"Good for Work/Study":int0
"Good for Relaxation/Meditation":int0
"Good for Exercise":int0
"Good for Running":int0
"Good for Yoga/Stretching":int0
"Good for Driving":int0
"Good for Social Gatherings":int0
"Good for Morning Routine":int0
"Similar Songs":
'''

label_cols = [
    "joy"
    "sadness"
    "anger"
    "fear"
    "surprise"
]

# Clean rows where any label is NaN: convert NaN → 0
df[label_cols] = df[label_cols].fillna(0).astype(int)

# Filter rows where sum of labels > 0
mask = df[label_cols].sum(axis=1) > 0
df_filtered = df[mask].copy()

print("Filtered dataset shape:", df_filtered.shape)

TEXT_COLUMN = "text" 

# Make sure text is fully string and safe
df_filtered[TEXT_COLUMN] = (
    df_filtered[TEXT_COLUMN]
    .astype(str)
    .str.replace('\r', ' ', regex=False)
    .str.replace('\n', ' ', regex=False)
)

# Normalize artist + song fields
df_filtered["Artist(s)"] = df_filtered["Artist(s)"].astype(str).str.strip()
df_filtered["song"] = df_filtered["song"].astype(str).str.strip()

# Remove rows with missing / empty text or empty song name
df_filtered = df_filtered[
    df_filtered["text"].astype(str).str.strip() != ""
]
df_filtered = df_filtered[
    df_filtered["song"].astype(str).str.strip() != ""
]

# Remove very short texts (< 100 characters) or very large
df_filtered = df_filtered[
    df_filtered["text"].str.len() >= 100
]
df_filtered = df_filtered[df_filtered["text"].str.len() < 10000]

# Drop duplicates
df_filtered = df_filtered.drop_duplicates(subset=["Artist(s)", "song"])
df_filtered = df_filtered.drop_duplicates(subset=["text"])
df_filtered = df_filtered[df_filtered["text"].str.len() < 10000]


output_file = "data/spotify_labeled_clean.csv"

df_filtered.to_csv(
    output_file,
    index=False,
    encoding="utf-8",
    quoting=1   # csv.QUOTE_ALL → safest option
)

print("Filtered dataset shape:", df_filtered.shape)
print("Saved cleaned dataset to:", output_file)
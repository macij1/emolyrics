(iao) juan.macias@feverup.com@Juan-Macias-Romeros-MacBook-Pro iao-spotify-good4 % python clean-emotion.py
Loaded dataset with shape: (551443, 39)
2025-12-06 12:43:01.633 python[7418:20453204] +[IMKClient subclass]: chose IMKClient_Modern
2025-12-06 12:43:01.633 python[7418:20453204] +[IMKInputSession subclass]: chose IMKInputSession_Modern
After dropping rows with missing emotion: (551443, 39)
After normalizing emotion labels:
emotion
joy          209009
sadness      171078
anger        109679
fear          28097
love          27966
surprise       5592
True             17
pink              2
thirst            1
confusion         1
interest          1
Name: count, dtype: int64
After removing empty lyrics: (551443, 39)
After removing duplicate artist+song pairs: (498052, 39) (removed 53391 rows)
After filtering by text length [30, 10000]: (497534, 39)
Discarded classes: Index(['True', 'pink', 'thirst', 'confusion', 'interest'], dtype='object', name='emotion')
After removing rare classes: (497513, 39)
Saved cleaned dataset to: data/spotify_emotion_clean.csv
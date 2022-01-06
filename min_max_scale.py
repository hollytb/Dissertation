import pandas as pd
from sklearn.preprocessing import MinMaxScaler

features_csv = "data/features.csv"
df = pd.read_csv(features_csv, index_col=0)
print(df.shape)

numb_cols = ["comment_count",
             "articleWordCount"
             ]

scaler = MinMaxScaler()
df[numb_cols] = scaler.fit_transform(df[numb_cols])

df.to_csv(path_or_buf="data/normalised.csv")

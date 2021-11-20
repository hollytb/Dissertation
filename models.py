import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math


def identity_tokenizer(text):
    return text

normalised_csv = "data/normalised.csv"
df = pd.read_csv(normalised_csv, index_col=0)
df["text"] = df["text"].apply(eval)
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 1), lowercase=False)
tfidf_text = tfidf.fit_transform(df['text']).toarray()
tfidf_tokens = tfidf.get_feature_names_out()

K = range(5, 50, 5)
SSE_mean = []
SSE_std = []
for k in K:
    gmm = KMeans(n_clusters=k)
    kf = KFold(n_splits=5)
    m = 0
    v = 0
    for train, test in kf.split(tfidf_text):
        gmm.fit(train.reshape(-1, 1))
        cost = -gmm.score(test.reshape(-1, 1))
        m = m + cost
        v = v + cost * cost
    SSE_mean.append(m / 5)
    SSE_std.append(math.sqrt(v / 5 - (m / 5) * (m / 5)))
plt.errorbar(K, SSE_mean, yerr=SSE_std, xerr=None, fmt='bx-')
plt.ylabel("cost")
plt.xlabel("number of clusters")
plt.show()

k = 10
gmm = KMeans(n_clusters=k).fit(tfidf_text)
centers = gmm.cluster_centers_.argsort()[:, ::-1]
for i in range(0, k):
    word_list = []
    for j in centers[i, :25]:
        word_list.append(tfidf_tokens[j])
    print("cluster%d:" % i)
    print(word_list)

labels = gmm.predict(tfidf_text)
count = 0
print("\nsimiliar comments:")
for j in range(0, labels.shape[0]):
    if labels[j] == 0:
        print("\n" + df["comment"].iloc[j])
        count = count + 1
        if count >= 5:
            break

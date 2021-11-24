import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap
from bokeh.io import show
from bokeh.plotting import figure


def identity_tokenizer(text):
    return text


def k_fold_cross_val(k, model, input):
    print(f"=== KFOLD k={k} ===")
    k_fold = KFold(n_splits=k, shuffle=True)
    m = 0
    v = 0
    for train, test in k_fold.split(input):
        model.fit(train.reshape(-1, 1))
        cost = -gmm.score(test.reshape(-1, 1))
        print("Model:" + type(model).__name__)
        m = m + cost
        v = v + cost * cost
    mean = (m / k)
    std = (math.sqrt(v / k - (m / k) * (m / k)))
    print(f"mean={mean},variance={std}")
    return mean, std


def error_plot(x, means, yerr, title, x_label):
    plt.errorbar(x, means, yerr=yerr, fmt='.', capsize=5)
    plt.plot(x, means, linestyle=':', label='mean', linewidth=2, color='orange')
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('cost')
    plt.tight_layout()
    plt.show()


normalised_csv = "data/normalised.csv"

df = pd.read_csv(normalised_csv, index_col=0)
df["text"] = df["text"].apply(eval)
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 2), lowercase=False, max_df=0.2)
tfidf_text = tfidf.fit_transform(df['text']).toarray()
tfidf_tokens = tfidf.get_feature_names_out()

# Model 1 - K Means
print("\n=== K MEANS ===")

# Cross Validation for hyper-parameter n_clusters:
K_range = range(5, 50, 5)
means = []
std_devs = []
for K in K_range:
    gmm = KMeans(n_clusters=K)
    res = k_fold_cross_val(5, gmm, tfidf_text)
    means.append(res[0])
    std_devs.append(res[1])
log_k_vals = np.log10(K_range)
error_plot(log_k_vals, means, std_devs, 'KMeans: varying clusters', 'log10(number of clusters)')

# K Means model with chosen hyper-parameter:
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

# Visualize clusters
tsne = TSNE(n_components=3, perplexity=30)  # , verbose=1, perplexity=140, n_iter=5000, learning_rate=100
tsne_scale_results = tsne.fit_transform(tfidf_text)
tsne_df_scale = pd.DataFrame(tsne_scale_results, columns=['tsne1', 'tsne2', 'tsne3'])
kmeans_tsne_scale = KMeans(n_clusters=k).fit(
    tsne_df_scale)  # n_init=100, max_iter=400, init='k-means++', random_state=42

plt.figure(figsize=(15, 15))
labels_tsne_scale = kmeans_tsne_scale.labels_
sns.scatterplot(tsne_df_scale.iloc[:, 0], tsne_df_scale.iloc[:, 1], hue=labels_tsne_scale, palette='Set1', s=100,
                alpha=0.6).set_title('Cluster Vis tSNE Scaled Data', fontsize=15)
plt.tight_layout();
plt.legend()
plt.show()

# Cross Validation for hyper-parameter n_clusters:
means = []
std_devs = []
for K in K_range:
    gmm = KMeans(n_clusters=K)
    res = k_fold_cross_val(5, gmm, tsne_df_scale)
    means.append(res[0])
    std_devs.append(res[1])
log_k_vals = np.log10(K_range)
error_plot(log_k_vals, means, std_devs, 'KMeans: varying clusters with tSNE_Scaled Data', 'log10(number of clusters)')

# Visualize clusters with 3d graph
merged_frame_rolled_up = df  # .groupby(['ID','comment']).mean().reset_index()
merged_frame_rolled_up['tsne-2d-one'] = tsne_df_scale.iloc[:, 0]
merged_frame_rolled_up['tsne-2d-two'] = tsne_df_scale.iloc[:, 1]
merged_frame_rolled_up['labels'] = labels_tsne_scale

# data sources
source = ColumnDataSource(data=dict(
    x=merged_frame_rolled_up['tsne-2d-one'].values,
    y=merged_frame_rolled_up['tsne-2d-two'].values,
    desc=merged_frame_rolled_up['labels'].values,
    titles=merged_frame_rolled_up['comment'].values
))

# hover over information
hover = HoverTool(tooltips=[
    ("Title", "@titles")
])
# map colors
mapper = linear_cmap(field_name='desc',
                     palette=Category20[len(merged_frame_rolled_up['labels'].unique())],
                     low=min(merged_frame_rolled_up['labels'].values),
                     high=max(merged_frame_rolled_up['labels'].values))
# prepare the figure

p = figure(plot_width=1000, plot_height=1000,
           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
           title="Clustering Comments Based on K Means, TfidfVectorizer and",
           toolbar_location="right")
# plot
p.scatter('x', 'y', size=5,
          source=source,
          fill_color=mapper,
          line_alpha=0.3,
          line_color="black")

show(p)

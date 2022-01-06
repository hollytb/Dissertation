import nltk
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math
from ast import literal_eval
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap
from bokeh.io import show
from bokeh.plotting import figure
from sklearn.metrics import silhouette_samples, silhouette_score
import time
from scipy import sparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import collections

nltk.download("punkt")


def identity_tokenizer(text):
    return text


# for k means
def k_fold_cross_val2(k, model, input):
    print(f"=== KFOLD k={k} ===")
    k_fold = KFold(n_splits=k, shuffle=True)
    m = 0
    v = 0
    for train, test in k_fold.split(input):
        model.fit(train.reshape(-1, 1))
        cost = -model.score(test.reshape(-1, 1))
        print("Model:" + type(model).__name__)
        m = m + cost
        v = v + cost * cost
    mean = (m / k)
    std = (math.sqrt(v / k - (m / k) * (m / k)))
    print(f"mean={mean},variance={std}")
    return mean, std


def k_fold_cross_val(k, model, input, y):
    print(f"=== KFOLD k={k} ===")
    k_fold = KFold(n_splits=k, shuffle=True)
    sq_errs = []
    for train, test in k_fold.split(input):
        model.fit(input[train], y[train])
        print("Model:" + type(model).__name__)
        ypred = model.predict(input[test])
        sq_errs.append(mean_squared_error(y[test], ypred))
        print("Train: " + str(model.score(input[train], y[train])))
        print("Test: " + str(model.score(input[test], y[test])))
    mean = np.mean(sq_errs)
    std = np.std(sq_errs)
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


def print_results(preds, y_vals, title):
    print("===" + title + "===")
    # print(f"Confusion Matrix:\n{confusion_matrix(y_vals, preds)}")
    # f"\nAccuracy:{accuracy_score(y_vals, preds)}"
    # f"\nRecall:{recall_score(y_vals, preds)}"
    # f"\nF1:{f1_score(y_vals, preds)}"
    # f"\nPrecision:{precision_score(y_vals, preds)}")
    print(confusion_matrix(y_vals, preds))


def plot_top_features(classifier, coef_dict):
    model_coefs = pd.DataFrame(classifier.coef_)  # changed from dataframe to series
    coefs_df = model_coefs.T
    feature_name_list = list(X_ef.columns)

    # list w/ eng'd features & tf-idf n-grams
    all_feat_names = []
    for i in feature_name_list:
        all_feat_names.append(i)

    for i in tfidf.get_feature_names():
        all_feat_names.append(i)

    # creating column for feat names
    coefs_df['feats'] = pd.Series(all_feat_names)
    coefs_df.set_index('feats', inplace=True)
    coefs_df['feats'] = pd.Series(all_feat_names)
    coefs_df.set_index('feats', inplace=True)

    # plot non-cb
    coefs_df[0].sort_values(ascending=True).head(20).plot(kind='bar')
    plt.title("SVM: Top 20 Non-Clickbait Coefs")
    # plt.title("LogReg: Top 20 Non-Clickbait Coefs")
    plt.xlabel("features")
    plt.ylabel("coef value")
    plt.xticks(rotation=55)
    plt.show()

    # plot CB classification
    # coefs_df[0].sort_values(ascending=False).head(20).plot(kind='bar', color='orange')
    # plt.title("SVM: Top 20 Clickbait Coefs")
    # plt.title("LogReg: Top 20 Clickbait Coefs")
    # plt.xlabel("features")
    # plt.ylabel("coef value")
    # plt.xticks(rotation=55)
    # plt.show()


def plot_frequent_keywords(tokens):
    # count frequency of words and store in dict
    cv = CountVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    words_cv = cv.fit_transform(tokens)
    word_list = cv.get_feature_names()
    count_list = words_cv.toarray().sum(axis=0)
    word_dict = dict(zip(word_list, count_list))
    word_counter = collections.Counter(word_dict)

    # list words and counts
    top_words = []
    for word, count in word_counter.most_common(25):
        print("Top 10 Article Keywords:")
        print(word, ": ", count)
        str(word).replace("'", '')
        top_words.append(word)

    # plot words and counts
    lst = word_counter.most_common(25)
    df = pd.DataFrame(lst, columns=['Word', 'Count'])
    df.plot.barh(x='Word', y='Count', rot=5)
    plt.title("Top 25 Article Keywords ")
    plt.show()

    # Create dataframe of top keywords and the number of comments they elicit
    df_boxplot = pd.read_csv("data/features.csv")
    df_boxplot.sort_values('articleID', inplace=True)  # this is duplicate code which should be refactored
    df_boxplot.drop_duplicates(subset="articleID", keep="first", inplace=True)

    df_boxplot = pd.concat([Series(row['comment_count'], row['keywords'].split("', '"))
                            for _, row in df_boxplot.iterrows()]).reset_index()
    # Preprocess keywords
    df_boxplot['index'] = df_boxplot['index'].str.replace("[", '')
    df_boxplot['index'] = df_boxplot['index'].str.replace("]", '')
    df_boxplot['index'] = df_boxplot['index'].str.replace("'", '')
    # Remove keywords not in the top 25
    df_boxplot = df_boxplot[df_boxplot['index'].isin(top_words)]

    # plot box plot
    sns.boxplot(x=0, y='index', data=df_boxplot, orient="h")
    plt.title('Top Keywords & No.Comments')
    plt.xlabel('number of comments');
    plt.ylabel('keyword')
    plt.show()


s_time = time.time()
normalised_csv = "data/normalised.csv"

# df = pd.read_csv(normalised_csv, index_col=0)
# df["text"] = df["text"].apply(eval)
# tfidf = TfidfVectorizer(stop_words = "english", tokenizer=identity_tokenizer, ngram_range=(1, 1), lowercase=False, max_df=0.2)
# tfidf_text = tfidf.fit_transform(df['text']).toarray()
# tfidf_tokens = tfidf.get_feature_names_out()

# print("Number of comments:",len(tfidf_text))
# print(f"Number of unique words in {len(tfidf_text)} comments:", len(tfidf_tokens))
## for i in tfidf_tokens:
#    print(i)

# Model 1 - MiniBatchKMeans
# print("\n=== K MEANS ===")

# Cross Validation for hyper-parameter n_clusters:
# K_range = range(5, 50, 5)
# means = []
# std_devs = []
# for K in K_range:
#    gmm = MiniBatchKMeans(n_clusters=K)
#    res = k_fold_cross_val(5, gmm, tfidf_text)
#    means.append(res[0])
#    std_devs.append(res[1])
# log_k_vals = np.log10(K_range)
# error_plot(log_k_vals, means, std_devs, 'MiniBatchKMeans: varying clusters', 'log10(number of clusters)')

# MiniBatchKMeans model with chosen hyper-parameter:
# k = 10
# t0 = time.time()

# gmm = MiniBatchKMeans(n_clusters=k, verbose=1).fit(tfidf_text)
# t_batch = time.time() - t0
# print("MiniBatchKMeans train time: %.2fs\ninertia: %f" % (t_batch, gmm.inertia_))


# centers = gmm.cluster_centers_.argsort()[:, ::-1]
# for i in range(0, k):
#    word_list = []
#    for j in centers[i, :25]:
#        word_list.append(tfidf_tokens[j])
#    print("cluster%d:" % i)
#    print(word_list)

# labels = gmm.predict(tfidf_text)
# count = 0
# print("\nsimiliar comments:")
# for j in range(0, labels.shape[0]):
#    if labels[j] == 0:
#        print("\n" + df["commentBody"].iloc[j])
#        count = count + 1
#        if count >= 5:
#            break


# silhouette_avg = silhouette_score(tfidf_text, labels)
# print("For n_clusters =",k,"The average silhouette_score is :",silhouette_avg,)

# REVISIT VISUALIZING CLUSTERS
# Visualize clusters
# tsne = TSNE(n_components=3, perplexity=80)  # , verbose=1, perplexity=140, n_iter=5000, learning_rate=100
# tsne_scale_results = tsne.fit_transform(tfidf_text)
# tsne_df_scale = pd.DataFrame(tsne_scale_results, columns=['tsne1', 'tsne2', 'tsne3'])
# kmeans_tsne_scale = KMeans(n_clusters=k).fit(
#    tsne_df_scale)  # n_init=100, max_iter=400, init='k-means++', random_state=42

# plt.figure(figsize=(15, 15))
# labels_tsne_scale = kmeans_tsne_scale.labels_
# sns.scatterplot(tsne_df_scale.iloc[:, 0], tsne_df_scale.iloc[:, 1], hue=labels_tsne_scale, palette='Set1', s=100,
#                alpha=0.6).set_title('Cluster Vis tSNE Scaled Data', fontsize=15)
# plt.tight_layout();
# plt.legend()
# plt.show()

# Cross Validation for hyper-parameter n_clusters:
# means = []
# std_devs = []
# for K in K_range:
#    gmm = KMeans(n_clusters=K)
#    res = k_fold_cross_val(5, gmm, tsne_df_scale)
#    means.append(res[0])
#    std_devs.append(res[1])
# log_k_vals = np.log10(K_range)
# error_plot(log_k_vals, means, std_devs, 'KMeans: varying clusters with tSNE_Scaled Data', 'log10(number of clusters)')
# error_plot(K_range, means, std_devs, 'KMeans: varying clusters with tSNE_Scaled Data', 'number of clusters')


# Visualize clusters with 3d graph
# merged_frame_rolled_up = df  # .groupby(['ID','comment']).mean().reset_index()
# merged_frame_rolled_up['tsne-2d-one'] = tsne_df_scale.iloc[:, 0]
# merged_frame_rolled_up['tsne-2d-two'] = tsne_df_scale.iloc[:, 1]
# merged_frame_rolled_up['labels'] = labels_tsne_scale

# data sources
# source = ColumnDataSource(data=dict(
#    x=merged_frame_rolled_up['tsne-2d-one'].values,
#    y=merged_frame_rolled_up['tsne-2d-two'].values,
#    desc=merged_frame_rolled_up['labels'].values,
#    titles=merged_frame_rolled_up['commentBody'].values
# ))

# hover over information
# hover = HoverTool(tooltips=[
#    ("Title", "@titles")
# ])
# map colors
# mapper = linear_cmap(field_name='desc',
#                     palette=Category20[len(merged_frame_rolled_up['labels'].unique())],
#                     low=min(merged_frame_rolled_up['labels'].values),
#                     high=max(merged_frame_rolled_up['labels'].values))
# prepare the figure

# p = figure(plot_width=1000, plot_height=1000,
#           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
#           title="Clustering Comments Based on K Means, TfidfVectorizer and",
#           toolbar_location="right")
# plot
# p.scatter('x', 'y', size=5,
#          source=source,
#          fill_color=mapper,
#          line_alpha=0.3,
#          line_color="black")

# show(p)
# e_time = time.time()
# print("Time: ", (e_time - s_time), "seconds")

# Predicting the number of comments
# df2 = pd.read_csv(normalised_csv)
# features2 = df2.drop(columns=['commentBody'])
# features2.to_csv(path_or_buf="data/features1.csv")
# features2.sort_values('articleID', inplace=True)
# features2.drop_duplicates(subset="articleID", keep="first", inplace=True)
# features2.to_csv(path_or_buf="data/features2.csv")
from string import punctuation

df2 = pd.read_csv("data/features2.csv", index_col=0)
my_punctuation = '€£' + punctuation
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth',
                       None):  # more options can be specified also
    print(df2.head())
# df2['keywords'] = df2['keywords'].apply(lambda y: [token.lower() for token in y if token.isalpha()])
# df2['keywords'] = [' '.join(map(str, l)) for l in df2['keywords']]


df2['keywords'] = df2['keywords'].apply(eval)

# df2['sectionName'] = df2['sectionName'].apply(eval)
features = df2.drop(columns=['comment_count', 'articleID', 'sectionName'])
y = df2['comment_count'].to_numpy()
# print(np.array(np.unique(y, return_counts=True)).T)
print(y.shape)

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 1), lowercase=False,
                        max_features=100)

tfidf_text = tfidf.fit_transform(features['keywords'])
print("YUP")
print(tfidf.vocabulary_)
# tfidf2 = TfidfVectorizer(stop_words='english', tokenizer=identity_tokenizer, ngram_range=(1, 1), lowercase=False,
#                         )
# tfidf_text2 = tfidf2.fit_transform(features['sectionName'])

X_ef = features.drop(columns='keywords')
X = sparse.hstack([X_ef, tfidf_text]).tocsr()  # tfidf_text2
print(X.shape)
print(y.shape)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):  # more options can be specified also
#    print(tfidf_tokens)
# print(tfidf2.get_feature_names_out())
#################################
# model 1 - Linear Ridge
print("\n=== LINEAR RIDGE | L2 PENALTY ===")
# Cross Validation for hyperparameter alpha:
# a_range = [5, 10, 20, 30, 40]
# means = []
# std_devs = []
# for A in a_range:
#    lin_ridge_clf = Ridge(alpha=A)
#    res = k_fold_cross_val(5, lin_ridge_clf, X, y)
#    means.append(res[0])
#    std_devs.append(res[1])
# log_a_vals = np.log10(a_range)
# error_plot(log_a_vals, means, std_devs, 'LinRidge: varying Alpha', 'log10(A)')

X_train, X_test, y_train, y_test = train_test_split(X, y)  # , random_state=20, test_size=0.3
print(X_train.shape)
print(X_test.shape)
lin_ridge = Ridge(alpha=20).fit(X_train, y_train)
pred = lin_ridge.predict(X_test)
predictions = pred.reshape(-1, 1)
print(lin_ridge.intercept_)
print("TEST")
print(lin_ridge.coef_)

print('MSE : ', mean_squared_error(y_test, predictions))
print('RMSE : ', np.sqrt(mean_squared_error(y_test, predictions)))

# coefficients = pd.DataFrame({"names":tfidf.get_feature_names(),
# "coef":lin_ridge.coef_})
# coefficients.sort_values("coef", ascending=False).head(10)
coef_dict = {}
for coef, feat in zip(lin_ridge.coef_, tfidf.get_feature_names()):
    coef_dict[feat] = coef

print(coef_dict)
new = coef_dict.keys()
feature_name_list = coef_dict.values()
print(coef_dict)
print("TEST AGAIN:")
print(len(lin_ridge.coef_))
print("AND AGAIN")
print(len(tfidf.get_feature_names()))
df2 = pd.read_csv("data/features.csv")
df_corr = abs(df2.corr().sort_values(by='comment_count', ascending=False))[['comment_count']]
# sns.heatmap(df2.corr())
df_small = df2[df_corr[df_corr['comment_count'] > 0].index.tolist()]
df_small.to_csv(path_or_buf="data/small.csv")
# sns.heatmap(features.corr())
from statsmodels.stats.outliers_influence import variance_inflation_factor

# plot_top_features(lin_ridge, coef_dict)

# coefs_df = pd.DataFrame.from_dict(coef_dict)

# coefs_df.index[0].sort_values(ascending=True).head(20).plot(kind='bar')
# plt.title("SVM: Top 20 Non-Clickbait Coefs")
# plt.title("LogReg: Top 20 Non-Clickbait Coefs")
# plt.xlabel("features")
# plt.ylabel("coef value")
# plt.xticks(rotation=55)
# plt.show()

# vals = sorted(coef_dict.values(), reverse=False)
# top = vals[:5]

# test = vals.index(0,4)

# vals2 = sorted(coef_dict.keys(), reverse=False)
# top2 = vals2[:5]
# print(top2)
# x = top2
sorted_dict = {}
sorted_keys = sorted(coef_dict, reverse=True, key=coef_dict.get)
for w in sorted_keys:
    sorted_dict[w] = coef_dict[w]

print(sorted_dict)
top = [v for v in list(sorted_dict.values())[:3]]

x = [v for v in list(sorted_dict.keys())[:3]]
fig, ax = plt.subplots()

ax.bar(x, top, width=0.1, color='m')

ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_xlabel("This graph shows amount of protocols used")
ax.set_ylabel("Number of times used")
ax.grid('on')

# Most frequent keywords for articles
# Keywords elicit more comments
plot_frequent_keywords(features["keywords"])

# run simple linear regression to see if Trump os a significant
# predictor of how many comments an article will get
# create a new feature
df3 = pd.read_csv("data/features2.csv")
df3["trumpMention"] = df3["keywords"].apply(lambda x: "Trump, Donald J" in x)

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth',
                       None):  # more options can be specified also
    print(df3.head())

import nltk
from dask_ml.cluster import KMeans
from pandas import Series
import math
from string import punctuation
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

plt.rcParams["font.family"] = "serif"
# fig, ax = plt.subplots(figsize=(30, 30))
# fig.tight_layout(pad=5)
#sequential_colors = sns.color_palette("Set2", 5)
#sns.set_palette(sequential_colors)
# nltk.download("punkt")
my_punctuation = '€£' + punctuation


# https://towardsdatascience.com/oversampling-and-undersampling-5e2bbaf56dcf
def balancing_data(X, y):
    # instantiating over and under sampler
    over = RandomOverSampler(sampling_strategy=0.7)
    under = RandomUnderSampler(sampling_strategy=0.9)
    # first performing oversampling to minority class
    X_over, y_over = over.fit_resample(X, y)
    print(f"Oversampled: {Counter(y_over)}")

    # now to comine under sampling
    X_combined_sampling, y_combined_sampling = under.fit_resample(X_over, y_over)
    print(f"Combined Random Sampling: {Counter(y_combined_sampling)}")
    return X_combined_sampling, y_combined_sampling


def identity_tokenizer(text):
    return text


def k_fold_cross_val_unsupervised(k, model, input):
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


def k_fold_cross_val_supervised(k, model, input, y):
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


def error_plot_clusters(x, means, yerr, title):
    plt.errorbar(x, means, yerr=yerr, xerr=None, fmt='bx-')
    plt.ylabel('cost');
    plt.ylabel('number of clusters');
    plt.title(title);
    plt.show()


def error_plot(x, means, yerr, title, x_label):
    plt.errorbar(x, means, yerr=yerr, fmt='.', capsize=5, color='mediumaquamarine')
    plt.plot(x, means, linestyle=':', label='mean', linewidth=2, color='orange')
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('cost')
    plt.tight_layout()
    plt.show()


def plot_coeff(features_col, model, title):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    coeff_parameter = pd.DataFrame(model.coef_.T, features_col, columns=['coefficient'])
    coeff_parameter.sort_values(by='coefficient', ascending=True).head(50).plot(
        kind='barh', figsize=(9, 7), title=('Negative Top 25 Features: ' + title), ylabel='feature')
    coeff_parameter.sort_values(by='coefficient', ascending=False).head(50).plot(
        kind='barh', figsize=(9, 7), title=('Positive Top 25 Features: ' + title), ylabel='feature')
    plt.show()


def print_results_regression(title, model, X, y):
    print("===" + title + "===")
    print(f"R2 Score:\n{model.score(X, y)}"
          f"\nIntercept:{model.intercept_}"
          f"\nCoefficients:{model.coef_}"
          f"\nNumber of Cofeficients:{len(model.coef_)}")


def print_results_classification(preds, y_vals, title):
    print("===" + title + "===")
    print(f"Confusion Matrix:\n{confusion_matrix(y_vals, preds)}"
          f"\nAccuracy:{accuracy_score(y_vals, preds)}"
          f"\nRecall:{recall_score(y_vals, preds)}"
          f"\nF1:{f1_score(y_vals, preds)}"
          f"\nPrecision:{precision_score(y_vals, preds)}")


def plot_words_with_number_comments(df_boxplot, x, y, top_words, title, x_label, y_label):
    # Create dataframe of top keywords and the number of comments they elicit
    df_boxplot = pd.concat([Series(row[x], row[y].split("', '"))
                            for _, row in df_boxplot.iterrows()]).reset_index()

    # Preprocess keywords
    df_boxplot['index'] = df_boxplot['index'].str.replace("[", '')
    df_boxplot['index'] = df_boxplot['index'].str.replace("]", '')
    df_boxplot['index'] = df_boxplot['index'].str.replace("'", '')
    # Remove keywords not in the top 25
    df_boxplot = df_boxplot[df_boxplot['index'].isin(top_words)]

    # plot box plot
    sns.catplot(x=0, y='index', data=df_boxplot, kind='box', orient="h", linewidth=0.5, fliersize=1, aspect=1.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    plt.show()


def plot_frequent_words(tokens, title):
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
        print("Top 10:")
        print(word, ": ", count)
        str(word).replace("'", '')
        top_words.append(word)

    # plot words and counts
    lst = word_counter.most_common(25)
    df = pd.DataFrame(lst, columns=['word', 'count'])
    df.plot.barh(x='word', y='count')
    plt.title(title)
    plt.show()

    return top_words


def data_handling(file, text, drop_cols, y_val):
    df = pd.read_csv(file)
    df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)
    df[text] = df[text].apply(eval)
    print(df.head())
    features = df.drop(columns=drop_cols)
    y = df[y_val].to_numpy()
    print(np.array(np.unique(y, return_counts=True)).T)
    print(y.shape)
    return features, y, df


def vectorization(features, text):
    pd.set_option('display.max_columns', None)
    tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 1), lowercase=False,
                            stop_words='english')  # max_features=500, max_df=0.5,
    X_ef = features.drop(columns=[text])
    print(X_ef.head())
    tfidf_text = tfidf.fit_transform(features[text])
    X = sparse.hstack([X_ef.astype(float), tfidf_text]).tocsr()
    feature_columns = np.concatenate((X_ef.keys().to_numpy(), tfidf.get_feature_names_out()), axis=None)
    return X, feature_columns, tfidf_text, tfidf


def linear_model(X, y, feature_columns, title):  # article_X, article_y
    print("\nLinear Regression for " + title)
    # print(article_X.shape)
    # print(article_y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=20, test_size=0.3)
    model = Ridge(alpha=10).fit(X_train, y_train)

    #curr_df[numb_col_article] = MinMaxScaler().fit_transform(curr_df[numb_col_article])

    pred_train = model.predict(X_train)  # predictions = pred.reshape(-1, 1)
    pred = model.predict(X_test)  # predictions = pred.reshape(-1, 1)
    print('Train MSE : ', mean_squared_error(y_train, pred_train))
    print('Train RMSE : ', np.sqrt(mean_squared_error(y_train, pred_train)))
    print('Test MSE : ', mean_squared_error(y_test, pred))
    print('Test RMSE : ', np.sqrt(mean_squared_error(y_test, pred)))
    print_results_regression('Linear Regression Model', model, X_train, y_train)
    #plot_coeff(feature_columns, model, title)
    return X_train, X_test, y_train, y_test, model

def categorize_sentiment(x):
    if x > 0:
        return 'Positive'
    elif x == 0:
        return 'Netural'
    else:
        return 'Negative'


# def print_sentiment_comments(int_label, text_label):
#     print("\n10 comments with highest " + text_label + " sentiment polarity:")
#     comments = non_norm_comment_df.loc[non_norm_comment_df['sentiment'] == int_label, ['comment']].sample(10).values
#     for comment in comments:
#         print(comment[0])


# ----------------------------------
# Articles File

article_df, article_features, article_y, article_X, article_feat_cols = {}, {}, {}, {}, {}
article_tfidf_text, article_tfidf_text, article_tfidf, coeff_parameter, coeff_parameter_pos, coeff_parameter_neg = \
    {}, {}, {}, {}, {}, {}
all_topics_pos, all_topics_neg, all_topics_dist_plot = [], [], []
topics = ['Foreign', 'National']
i = 0

for topic in topics:
    article_features[i], article_y[i], article_df[i] = data_handling("data/normalised" + topic + "0.csv",
                                                                     'keywords', ['comment_count', 'section_name',
                                                                                  'article_id'],
                                                                     'comment_count')
#     article_X[i], article_feat_cols[i], article_tfidf_text[i], article_tfidf[i] = vectorization(
#         article_features[i], 'keywords')
#
#     # 2. Most frequent keywords for articles and Keywords elicit more comments
#
#     top_words = plot_frequent_words(article_features[i]['keywords'], 'Top 25 Keywords from Articles')
#     non_norm_article_df = pd.read_csv("data/featuresArticlesTest" + topic + ".csv")
#
#     plot_words_with_number_comments(non_norm_article_df, 'comment_count', 'keywords', top_words,
#                                     'Top Keywords & No.Comments', 'comments', 'word')
#
#     # Models:
#     # 1.Linear Regression - Run simple linear regression to see if Trump is a significant
#     _, _, _, _, model = linear_model(article_X[i], article_y[i], article_feat_cols[i],
#                                      "LinReg Articles")
#
#     coeff_parameter[i] = pd.DataFrame(model.coef_.T, article_feat_cols[i], columns=['coefficient'])
#     coeff_parameter[i]['Section'] = topic
#     coeff_parameter[i]['Section'] = topic
#     coeff_parameter_pos[i] = coeff_parameter[i].sort_values(by='coefficient', ascending=False).head(10)
#     coeff_parameter_neg[i] = coeff_parameter[i].sort_values(by='coefficient', ascending=True).head(10)
#
#     all_topics_pos.append(coeff_parameter_pos[i])
#     all_topics_neg.append(coeff_parameter_neg[i])
#     all_topics_dist_plot.append(coeff_parameter[i])
#     i += 1
# # ch2 = SelectKBest(chi2, k=1000)
# # article_X = ch2.fit_transform(article_X, article_y)
#
# # Supervised Regression for Articles
# # Graphs:
#
# # 1. number of comments that mention Trump but are not under Trump articles
# # plt.pie(article_df['trump_article'].value_counts(), labels=('not trumpArticles', 'trumpArticles'), autopct='%.0f%%');
# # plt.title('All articles')
# # plt.show()
#
#
# # -----------
# # Comparison plot
# all_topics_df = pd.concat(all_topics_pos)
# all_topics_df = all_topics_df.sort_values(by='coefficient', ascending=True)
# all_topics_df = all_topics_df.reset_index()  # Convert index of the dataframe to a column for barchart
#
# fig, ax = plt.subplots(figsize=(30, 30))
# fig.tight_layout(pad=5)
# sns.barplot(x='coefficient', y='index', data=all_topics_df, hue='Section', dodge=False)
# plt.title('Top 30 + Features  \n by news sections', fontsize=60)
# plt.xlabel('coefficient', fontsize=60)
# plt.xticks(fontsize=35)
# plt.ylabel('features', fontsize=60)
# plt.yticks(fontsize=45)
# sns.despine(bottom=False, left=True)
# ax.grid(False)
# plt.legend(fontsize=45)
# plt.show()
#
# # Plot histogram
# all_topics_dist_plot_df = pd.concat(all_topics_dist_plot)
# all_topics_dist_plot_df = all_topics_dist_plot_df.reset_index()
# sns.displot(x='coefficient', data=all_topics_dist_plot_df, hue='Section', element='step')
# plt.xlabel('Coefficient')
# plt.show()

topics = ['Sports', 'National', 'Magazine']
#topics = ['Sports']
colour = ['green', 'blue', 'pink']
palette ={"Sports": "skyblue", "National": "slateblue", "Magazine": "dodgerblue"}
#topics = ['National']
# ----------------------------------
# Comments File
comment_features_reg, comment_y_reg, comment_df_reg = {}, {}, {}
comment_features_class, comment_y_class, comment_df_class = {}, {}, {}
non_norm_comment_df, comment_X_reg, comment_feat_cols_reg, comment_tfidf_text_reg, comment_tfidf_reg = {}, {}, {}, {}, {}
comment_X_class, comment_feat_cols_class, comment_tfidf_text_class, comment_tfidf_class = {}, {}, {}, {}
X_train_com, X_test_com, y_train_com, y_test_com, top_words_comments = {}, {}, {}, {}, {}
X_train_com_reg, X_test_com_reg, y_train_com_reg, y_test_com_reg = {}, {}, {}, {}
all_topics_pos_com, all_topics_neg_com, all_topics_dist_plot_com = [], [], []
all_topics_pos_com_reg, all_topics_neg_com_reg, all_topics_dist_plot_com_reg = [], [], []
coeff_parameter_com, coeff_parameter_com_pos, coeff_parameter_com_neg = {}, {}, {}
coeff_parameter_com_reg, coeff_parameter_com_pos_reg, coeff_parameter_com_neg_reg = {}, {}, {}
from sklearn.ensemble import RandomForestClassifier

i = 0
for topic in topics:
    comment_features_reg[i], comment_y_reg[i], comment_df_reg[i] = data_handling("data/normalisedBalanced"+topic+"0.csv", 'comment',
                                                                   ['comment_count',
                                                                    'recommendations', 'keywords',
                                                                    'article_id'],
                                                                   'recommendations')

    comment_features_class[i], comment_y_class[i], comment_df_class[topic] = data_handling(
        "data/normalisedBalanced" + topic + "0.csv", 'comment',
        ['comment_count',
         'gets_reply', 'keywords',
         'article_id'],
        'gets_reply')

    comment_X_reg[i], comment_feat_cols_reg[i], comment_tfidf_text_reg[i], comment_tfidf_reg[i] = vectorization(comment_features_reg[i],
                                                                                               'comment')

    comment_X_class[i], comment_feat_cols_class[i], comment_tfidf_text_class[i], comment_tfidf_class[i] = vectorization(
        comment_features_class[i], 'comment')
    # print('sentiment') # what you wanna do is try print out the sentiments of them texts shown, beside features
    # to tell if there are more postive words having a impact and so on.
    # print(comment_features_class[i]['sentiment'])
    # ch2 = SelectKBest(chi2, k=500)
    # ch2 = SelectKBest(chi2, k=250)
    # comment_X_class[i] = ch2.fit_transform(comment_X_class[i], comment_y_class[i])

    # # Supervised Regression for Comments
    # # Graphs:
    #
    # # 1.
    # non_norm_comment_df[i] = pd.read_csv("data/featuresCommentsTest"+topic+".csv")
    # plt.pie(non_norm_comment_df.loc[non_norm_comment_df['trump_article'] == 0, 'trump_comment'].value_counts(),
    #         labels=('not trumpComment', 'trumpComment'),
    #         autopct='%.0f%%')
    # plt.title('All articles that are not Trump related')
    # plt.show()

    # # 2.
    # plt.pie(non_norm_comment_df.loc[non_norm_comment_df['trump_article'] == 1, 'trump_comment'].value_counts(),
    #         labels=('not trumpComment', 'trumpComment'),
    #         autopct='%.0f%%')
    # plt.title('All articles that are Trump related')
    # plt.show()
    #
    # # 3.
    # sns.barplot(x='trump_article', y='recommendations', hue='trump_comment',
    #             data=non_norm_comment_df)
    # plt.title('Non-Trump/Trump articles with comments and recommendations')
    # plt.show()
    #
    # 4.  Most frequent words in comments and words that elicit more recommendations
    # top_words_comments[i] = plot_frequent_words(comment_features_class[i]['comment'], 'Top 25 Words from Comments')
    # plot_words_with_number_comments(non_norm_comment_df[i], 'recommendations', 'comment', top_words_comments[i],
    #                                'Top Words & No.Recommendations', 'recommendations', 'word')

    # # 5.
    # # non_norm_comment_df['sentiment_label'] = non_norm_comment_df['sentiment'].apply(lambda x: categorize_sentiment(x))
    # # non_norm_comment_df['sentiment_label'].value_counts().T.plot(kind='bar', rot=0, fontsize=30, color=['C1', 'C2', 'C0'])
    # # plt.title('The sentiment label of comments', fontsize=30)
    # # plt.show()
    #
    # # 6. print the highest value positive/negative/neutral comment
    # # print_sentiment_comments(1, 'positive')
    # # print_sentiment_comments(-1, 'negative')
    # # print_sentiment_comments(0, 'neutral')
    #
    # # 7. print the distribution of sentiment polarity of comments
    # sns.distplot(non_norm_comment_df['sentiment'])
    # plt.title("Distribution of sentiment polarity of comments")
    # plt.show()
    #
    # # Models
    # # 1. Linear Regression (run simple linear regression to see if Trump is a significant):





    X_train_com_reg[i], X_test_com_reg[i], y_train_com_reg[i], y_test_com_reg[i], lig_model = linear_model(comment_X_reg[i], comment_y_reg[i], comment_feat_cols_reg[i],
                                                                    "LinReg Comments")

    # Cross Validation for hyperparameter C:
    # c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # means = []
    # std_devs = []
    # for C in c_range:
    #     lin_ridge = Ridge(alpha=C)
    #     res = k_fold_cross_val_supervised(5, lin_ridge, comment_X_reg[i], comment_y_reg[i])
    #     means.append(res[0])
    #     std_devs.append(res[1])
    # log_c_vals = np.log10(c_range)
    # error_plot(log_c_vals, means, std_devs, 'LinReg: L2 penalty, varying C', 'C')

    # Logistic Regression model with chosen hyper-parameters:

    coeff_parameter_com_reg[i] = pd.DataFrame(lig_model.coef_.T, comment_feat_cols_reg[i], columns=['coefficient'])
    coeff_parameter_com_reg[i]['Section'] = topic
    coeff_parameter_com_reg[i]['Section'] = topic
    coeff_parameter_com_reg[i].to_csv(path_or_buf='coefRegTEST'+topic+'.csv')

    coeff_parameter_com_pos_reg[i] = coeff_parameter_com_reg[i].sort_values(by='coefficient', ascending=False).head(15)
    coeff_parameter_com_neg_reg[i] = coeff_parameter_com_reg[i].sort_values(by='coefficient', ascending=True).head(15)

    all_topics_pos_com_reg.append(coeff_parameter_com_pos_reg[i])
    all_topics_neg_com_reg.append(coeff_parameter_com_neg_reg[i])
    all_topics_dist_plot_com_reg.append(coeff_parameter_com_reg[i])

    # # going to have to create a new X_train value's for comments
    # # Supervised Classification for Comments
    # # Graphs:
    #
    # # 1. plot the distribution of target variable to see if class is balanced
    # plot = sns.countplot(comment_df_class['gets_reply'])
    # plot.set_xticklabels(['No Reply', 'Reply'])
    # plt.show()
    #
    # # Models:
    comment_X_class[i], comment_y_class[i] = balancing_data(comment_X_class[i], comment_y_class[i])
    X_train_com[i], X_test_com[i], y_train_com[i], y_test_com[i] = train_test_split(comment_X_class[i],
                                                                                    comment_y_class[i], random_state=20,
                                                                                    test_size=0.3)
    # # 1. Logistic Regression
    print("\n=== LOGISTIC REGRESSION | L2 PENALTY ===")
    # # Cross Validation for hyperparameter C:
    # c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # means = []
    # std_devs = []
    # for C in c_range:
    #     log_clf = LogisticRegression(C=C, penalty="l1", class_weight='balanced', solver='liblinear')
    #     res = k_fold_cross_val_supervised(5, log_clf, comment_X_class[i], comment_y_class[i])
    #     means.append(res[0])
    #     std_devs.append(res[1])
    # log_c_vals = np.log10(c_range)
    # error_plot(log_c_vals, means, std_devs, 'LogReg Sports: L1 penalty, varying C', 'C')

    # # Logistic Regression model with chosen hyper-parameters:
    # C=10,
    log_clf = LogisticRegression(C=10, penalty='l2', solver='liblinear', class_weight='balanced')  # penalty="l2"
    print(X_train_com[i].shape)
    print(X_test_com[i].shape)
    #
    # # ROC curve plotting
    log_clf.fit(X_train_com[i], y_train_com[i])
    prediction = log_clf.predict_proba(X_test_com[i])
    #
    fpr, tpr, _ = roc_curve(y_test_com[i], prediction[:, 1])
    auc_score = roc_auc_score(y_test_com[i], prediction[:, 1])
    print("AUC Score:", auc_score)
    plt.plot(fpr, tpr, color='mediumslateblue', label='Logistic Regression')
    #
    # # predictions
    # log_clf.fit(X_train_com, y_train_com)
    preds_train_com = log_clf.predict(X_train_com[i])
    preds_test_com = log_clf.predict(X_test_com[i])
    #
    print_results_classification(preds_train_com, y_train_com[i], "LogReg train")
    print_results_classification(preds_test_com, y_test_com[i], "LogReg test")
    #plot_coeff(comment_feat_cols_class, log_clf, 'LogReg Comments')
    #

    coeff_parameter_com[i] = pd.DataFrame(log_clf.coef_.T, comment_feat_cols_class[i], columns=['coefficient'])
    coeff_parameter_com[i]['Section'] = topic
    coeff_parameter_com[i]['Section'] = topic
    coeff_parameter_com[i].to_csv(path_or_buf='coefClassTEST'+topic+'.csv')

    coeff_parameter_com_pos[i] = coeff_parameter_com[i].sort_values(by='coefficient', ascending=False).head(15)
    coeff_parameter_com_neg[i] = coeff_parameter_com[i].sort_values(by='coefficient', ascending=True).head(15)

    all_topics_pos_com.append(coeff_parameter_com_pos[i])
    all_topics_neg_com.append(coeff_parameter_com_neg[i])
    all_topics_dist_plot_com.append(coeff_parameter_com[i])

    #print(comment_feat_cols_class[i])
    #plot_coeff(comment_feat_cols_class[i], log_clf, "yup")

    # # # 2. Gaussian kernel SVM (SVC) model
    #print("\n=== SVC, Gaussian Kernel ===")
    # c_range = [0.001, 1, 1000]
    # gammas = [1, 2, 5, 8, 10]
    # for C in c_range:
    #    means = []
    #    std_devs = []
    #    for g in gammas:
    #        rbf_svc = SVC(C=C, kernel='rbf', gamma=g)
    #        results = k_fold_cross_val_supervised(5, rbf_svc, X_train_com[i], y_train_com[i])
    #        means.append(results[0])
    #        std_devs.append(results[1])
    #    plt.errorbar(gammas, means, yerr=std_devs, fmt='.', capsize=5, label=C)
    #    plt.plot(gammas, means, linestyle=':', linewidth=2)
    # plt.ylabel('mean square error')
    # plt.xlabel('gamma')
    # plt.title('MSE: varying C and γ')
    # plt.legend(title='C')
    # plt.show()
    # from sklearn import svm
    #
    #
    # # SVC Model with chosen parameters
    svc = SVC(C=1, kernel='linear', gamma=1, cache_size=1200, probability=True)
    #
    #roc curve
    svc.fit(X_train_com[i], y_train_com[i])
    prediction = svc.predict_proba(X_test_com[i])
    fpr, tpr, _ = roc_curve(y_test_com[i], prediction[:, 1])
    auc_score = roc_auc_score(y_test_com[i], prediction[:, 1])
    print("AUC Score:", auc_score)
    plt.plot(fpr, tpr, color='firebrick', label='SVC')
    #
    # #predictions
    svc.fit(X_train_com[i], y_train_com[i])
    preds_train = svc.predict(X_train_com[i])
    print_results_classification(preds_train, y_train_com[i], "SVC train")
    preds_test = svc.predict(X_test_com[i])
    print_results_classification(preds_test, y_test_com[i], "SVC test")

    #
    # # 3. kNN
    # print("\n=== kNN ===")
    #
    # Cross Validation for hyperparameter n_neighbors:
    # neighbours = [1, 3, 5, 7, 9]
    # means = []
    # stds = []
    # for n in neighbours:
    #     knn_clf = KNeighborsClassifier(n_neighbors=n, weights='uniform')
    #     res = k_fold_cross_val_supervised(5, knn_clf, X_train_com[i], y_train_com[i])
    #     knn_clf.score(X_train_com[i], y_train_com[i])
    #     means.append(res[0])
    #     stds.append(res[1])
    # error_plot(neighbours, means, stds, 'Prediction Error: varying n_neighbors parameters', 'n_neighbors')
    # plt.ylabel('cost')
    # kNN model with chosen hyperparameter:
    knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', n_jobs=2)
    #
    knn.fit(X_train_com[i], y_train_com[i])
    prediction = knn.predict_proba(X_test_com[i])
    fpr, tpr, _ = roc_curve(y_test_com[i], prediction[:, 1])
    auc_score = roc_auc_score(y_test_com[i], prediction[:, 1])
    print("AUC Score:", auc_score)
    plt.plot(fpr, tpr, color='gold', label='K-Neighbours')

    knn.fit(X_train_com[i], y_train_com[i])
    preds_train = knn.predict(X_train_com[i])
    preds_test = knn.predict(X_test_com[i])

    print_results_classification(preds_train, y_train_com[i], "KNN train")
    print_results_classification(preds_test, y_test_com[i], "KNN test")
    #
    # coeff_parameter_com[i] = pd.DataFrame(svc.coef_.T, comment_feat_cols_class[i], columns=['coefficient'])
    # coeff_parameter_com[i]['Section'] = topic
    # coeff_parameter_com[i]['Section'] = topic
    # coeff_parameter_com_pos[i] = coeff_parameter_com[i].sort_values(by='coefficient', ascending=False).head(10)
    # coeff_parameter_com_neg[i] = coeff_parameter_com[i].sort_values(by='coefficient', ascending=True).head(10)
    #
    # all_topics_pos_com.append(coeff_parameter_com_pos[i])
    # all_topics_neg_com.append(coeff_parameter_com_neg[i])
    # all_topics_dist_plot_com.append(coeff_parameter_com[i])
    #
    #
    # #
    # # # 4. baseline classifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_com[i], y_train_com[i])
    preds_train = dummy.predict(X_train_com[i])
    preds_test = dummy.predict(X_test_com[i])

    print_results_classification(preds_train, y_train_com[i], "Dummy train")
    print_results_classification(preds_test, y_test_com[i], "Dummy test")

    # # baseline confusion matrix for plotting point
    matrix = confusion_matrix(y_train_com[i], preds_train)

    most_freq_fpr = matrix[0][1] / (matrix[0][1] + matrix[0][0])  # FP / (FP + TN)
    most_freq_tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])  # TP / (TP + FN)

    plt.plot(most_freq_fpr, most_freq_tpr, label='Most Frequent Clf', marker='o', linestyle='None')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='mediumaquamarine', linestyle='--')
    plt.title('ROC curves for the chosen classifiers')
    plt.legend()
    plt.show()  # ROC plot

    i += 1
#
# # Unsupervised learning
#
# # Model:
# # 1. MiniBatchKMeans
# print("\n=== K MEANS ===")
#
# # Cross Validation for hyper-parameter n_clusters:
# K_range = range(5, 50, 5)
# means = []
# std_devs = []
# for K in K_range:
#     gmm = MiniBatchKMeans(n_clusters=K)
#     res = k_fold_cross_val_unsupervised(5, gmm, comment_tfidf_text_class)
#     means.append(res[0])
#     std_devs.append(res[1])
# error_plot_clusters(K_range, means, std_devs, 'MiniBatchKMeans: varying clusters')
#
# # MiniBatchKMeans model with chosen hyper-parameter:
# k = 15  # selected hyper-parameter
# gmm = MiniBatchKMeans(n_clusters=k, verbose=1).fit(comment_tfidf_text_class)
#
# terms = comment_tfidf_class.get_feature_names_out()
# centers = gmm.cluster_centers_.argsort()[:, ::-1]
# for i in range(0, k):
#     word_list = []
#     for j in centers[i, :25]:
#         word_list.append(terms[j])
#     print("cluster%d:" % i)
#     print(word_list)
# labels = gmm.predict(comment_tfidf_text_class)
# count = 0
# print('\nsimiliar comments:')
# for j in range(0, labels.shape[0]):
#     if labels[j] == 0:
#         print('\n' + str(comment_df_class['comment'].iloc[j]))
#         count = count + 1
#         if count >= 5:
#             break
#
# silhouette_avg = silhouette_score(comment_tfidf_text_class, labels)
# print('For n_clusters =', k, 'The average silhouette_score is :', silhouette_avg)

# -----------
# For Logistic regression Comparison plot
all_topics_df = pd.concat(all_topics_pos_com)
all_topics_df = all_topics_df.sort_values(by='coefficient', ascending=True)
all_topics_df = all_topics_df.reset_index()  # Convert index of the dataframe to a column for barchart

fig, ax = plt.subplots(figsize=(30, 30))
fig.tight_layout(pad=5)
sns.barplot(x='coefficient', y='index', data=all_topics_df, hue='Section', palette=palette, dodge=False)  #
plt.title('Top Positive 30 + Features  \n by news sections', fontsize=60)
plt.xlabel('coefficient', fontsize=60)
plt.xticks(fontsize=35)
plt.ylabel('features', fontsize=60)
plt.yticks(fontsize=45)
sns.despine(bottom=False, left=True)
ax.grid(False)
plt.legend(fontsize=45)
plt.show()

# -----------
# Comparison plot
all_topics_df = pd.concat(all_topics_neg_com)
all_topics_df = all_topics_df.sort_values(by='coefficient', ascending=True)
all_topics_df = all_topics_df.reset_index()  # Convert index of the dataframe to a column for barchart
fig, ax = plt.subplots(figsize=(30, 30))
fig.tight_layout(pad=5)
sns.barplot(x='coefficient', y='index', data=all_topics_df, hue='Section', palette=palette, dodge=False)  #
plt.title('Top Negative 30 + Features  \n by news sections', fontsize=60)
plt.xlabel('coefficient', fontsize=60)
plt.xticks(fontsize=35)
plt.ylabel('features', fontsize=60)
plt.yticks(fontsize=45)
sns.despine(bottom=False, left=True)
ax.grid(False)
plt.legend(fontsize=45)
plt.show()

# Plot histogram
fig.tight_layout(pad=5)
all_topics_dist_plot_df = pd.concat(all_topics_dist_plot_com)
all_topics_dist_plot_df = all_topics_dist_plot_df.reset_index()

g = sns.FacetGrid(all_topics_dist_plot_df, #the dataframe to pull from
                  row="Section", #define the column for each subplot row to be differentiated by
                  hue="Section", #define the column for each subplot color to be differentiated by
                  aspect=4, #aspect * height = width
                  height=1.3, #height of each subplot
                  palette=palette#google colors
                 )

g.map(sns.kdeplot, "coefficient", shade=True, alpha=1, lw=1.5, bw=0.2)
g.map(sns.kdeplot, "coefficient", lw=4, bw=0.2)
plt.show()



# For Linear regression Comparison plot
all_topics_df = pd.concat(all_topics_pos_com_reg)
all_topics_df = all_topics_df.sort_values(by='coefficient', ascending=True)
all_topics_df = all_topics_df.reset_index()  # Convert index of the dataframe to a column for barchart

fig, ax = plt.subplots(figsize=(30, 30))
fig.tight_layout(pad=5)
sns.barplot(x='coefficient', y='index', data=all_topics_df, hue='Section', palette=palette, dodge=False)  #
plt.title('Top Positive 30 + Features Reg \n by news sections', fontsize=60)
plt.xlabel('coefficient', fontsize=60)
plt.xticks(fontsize=35)
plt.ylabel('features', fontsize=60)
plt.yticks(fontsize=45)
sns.despine(bottom=False, left=True)
ax.grid(False)
plt.legend(fontsize=45)
plt.show()

# -----------
# Comparison plot
all_topics_df = pd.concat(all_topics_neg_com_reg)
all_topics_df = all_topics_df.sort_values(by='coefficient', ascending=True)
all_topics_df = all_topics_df.reset_index()  # Convert index of the dataframe to a column for barchart
fig, ax = plt.subplots(figsize=(30, 30))
fig.tight_layout(pad=5)
sns.barplot(x='coefficient', y='index', data=all_topics_df, hue='Section', palette=palette, dodge=False)  #
plt.title('Top Negative 30 + Features Reg \n by news sections', fontsize=60)
plt.xlabel('coefficient', fontsize=60)
plt.xticks(fontsize=35)
plt.ylabel('features', fontsize=60)
plt.yticks(fontsize=45)
sns.despine(bottom=False, left=True)
ax.grid(False)
plt.legend(fontsize=45)
plt.show()

# Plot histogram
fig.tight_layout(pad=5)
all_topics_dist_plot_df = pd.concat(all_topics_dist_plot_com_reg)
all_topics_dist_plot_df = all_topics_dist_plot_df.reset_index()

g = sns.FacetGrid(all_topics_dist_plot_df, #the dataframe to pull from
                  row="Section", #define the column for each subplot row to be differentiated by
                  hue="Section", #define the column for each subplot color to be differentiated by
                  aspect=4, #aspect * height = width
                  height=1.3, #height of each subplot
                  palette=palette#google colors
                 )

g.map(sns.kdeplot, "coefficient", shade=True, alpha=1, lw=1.5, bw=0.2)
g.map(sns.kdeplot, "coefficient", lw=4, bw=0.2)
plt.show()
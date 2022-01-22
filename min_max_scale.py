import pandas as pd
from sklearn.preprocessing import MinMaxScaler

numb_col_article = ['comment_count',
                    'article_word_count',
                    ]

numb_col_comment = ['comment_count',
                    'article_word_count',
                    'recommendations',
                    'comment_word_count',
                    'pub_length'
                    ]


def read_in_df(csv_file):
    curr_df = pd.read_csv(csv_file)
    print(curr_df.shape)
    return curr_df


def normalise(curr_df, csv_counter):
    if csv_counter == 0:
        curr_df[numb_col_article] = MinMaxScaler().fit_transform(curr_df[numb_col_article])
    else:
        curr_df[numb_col_comment] = MinMaxScaler().fit_transform(curr_df[numb_col_comment])


def remove_nan_rows(curr_df):  # removes list of empty keywords and comments
    nan_value = float("NaN")
    curr_df.replace('[]', nan_value, inplace=True)
    curr_df.dropna(inplace=True)


csv_files = ["data/featuresArticlesTest.csv", "data/featuresCommentsTest.csv"]
csv_no = 0
for csv in csv_files:
    df = read_in_df(csv)
    normalise(df, csv_no)
    remove_nan_rows(df)
    df.to_csv(path_or_buf=("data/normalised" + str(csv_no) + ".csv"), index=False)
    csv_no = csv_no + 1

# need to check how the target feature is balanced
# then down sample the size
# will need to refactor this

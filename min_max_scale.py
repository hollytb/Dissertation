import pandas as pd
from sklearn.preprocessing import MinMaxScaler

numb_col_article = ['comment_count',
                    'article_word_count',
                    ]

numb_col_comment = ['comment_count',
                    'article_word_count',
                    'recommendations',
                    'comment_word_count',
                    'pub_length',
                    'sentiment'
                    ]


def read_in_df(csv_file):
    curr_df = pd.read_csv(csv_file, index_col=0)
    return curr_df


def normalise(curr_df, csv_counter):
    if csv_counter == 0:
        curr_df[numb_col_comment] = MinMaxScaler().fit_transform(curr_df[numb_col_comment])
    else:
        curr_df[numb_col_article] = MinMaxScaler().fit_transform(curr_df[numb_col_article])


def remove_nan_rows(curr_df):  # removes list of empty keywords and comments
    nan_value = float("NaN")
    curr_df.replace('[]', nan_value, inplace=True)
    curr_df.dropna(inplace=True)

#
#csv_files = ["data/featuresArticlesTestSPORTS.csv", "data/featuresCommentsTestForeign.csv"]
csv_files = ['data/balancedSports.csv']
csv_no = 0
for csv in csv_files:


    df = read_in_df(csv)
    df = df[df.article_id != '5a665df410f40f00018bd3ad']
    normalise(df, csv_no)
    remove_nan_rows(df)

    df = df.drop(columns=['approve_date', 'pub_date', 'depth', 'comment_id',
                          'text', 'comment_body', 'parent_id', 'new_desk'])
    df.to_csv(path_or_buf=("data/normalisedBalancedSports" + str(csv_no) + ".csv"), index=False)

    csv_no = csv_no + 1
    print('File ' + str(csv_no) + ' shape: ' + str(df.shape))
    print()

# need to check how the target feature is balanced
# then down sample the size
# will need to refactor this

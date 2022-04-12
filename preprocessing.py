import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
import csv
import nltk
import contractions
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from nltk.tokenize import (MWETokenizer, word_tokenize)
import datetime as dt
from bs4 import BeautifulSoup

from textblob import TextBlob

nlp = en_core_web_sm.load()
nltk.download('words')
my_punctuation = '€£' + punctuation
en = spacy.load('en_core_web_sm')  # only load once

def remove_nan_rows(curr_df):  # removes list of empty keywords and comments
    nan_value = float("NaN")
    curr_df.replace('[]', nan_value, inplace=True)
    curr_df.dropna(inplace=True)

def tokenize(comment):
    tokens = [word_tokenize(x) for x in comment]
    return tokens


def remove_punct_tokens(tokens):
    return [token for token in tokens if token.isalnum()]


def word_count(comment):
    return len(comment.split())


def comment_count(article):
    return article.value_counts


def question_mark(comment):
    return int('?' in comment)


def exclamation_mark(comment):
    return int('!' in comment)


def starts_with_digit(comment):
    return int(comment[0].isdigit())


def starts_with_question_word(comment):
    return int(comment.startswith(('What', 'Where', 'When', 'Who', 'Why', 'Whom', 'Whose',
                                   'Which', 'How', 'Will', 'Would', 'Should', 'Could',
                                   'Do', 'Did')))


def remove_punctuation(comment):
    return ''.join(w for w in comment if w not in my_punctuation)


def longest_word_len(tokens):
    token_lengths = set(len(token) for token in tokens)
    return max(token_lengths)


def avg_word_len(tokens):
    return round(sum(len(tokens) for token in tokens) / len(tokens), 4)


def ratio_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    count = 0
    for token in tokens:
        if token in stop_words:
            count += 1
    return round(count / len(tokens), 4)


def lemmatise(tokens_wordnet):
    lemmatiser = WordNetLemmatizer()
    new_tokens = []
    for token, tag in tokens_wordnet:
        if tag is None:
            new_tokens.append(lemmatiser.lemmatize(token))
        else:
            new_tokens.append(lemmatiser.lemmatize(token, tag))
    return new_tokens


def pos_tagging(tokens):
    return pos_tag(tokens)


def pos_tag_convert(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def change_pos_tag(list_tuples):
    result = []
    for token, tag in list_tuples:
        result.append(tuple([token, pos_tag_convert(tag)]))
    return result


def remove_stopwords(tokens):
    """
    :param tokens: tokenized headline
    """
    sw_spacy = en.Defaults.stop_words
    return [token for token in tokens if token not in sw_spacy and len(token) > 2]


def display_most_frequentwords(comments, max):
    cnt = pd.Series(np.concatenate([str(x).split() for x in comments])).value_counts()
    print("Most Frequent Words:" + str(cnt[:max]))
    return cnt


def most_rarewords(cnt, max):
    rarewords = cnt[:-max - 1:-1]
    print('Most Rare Words:' + str(rarewords))
    return rarewords


def remove_rarewords(tokens, rarewords):
    return ' '.join([token for token in str(tokens).split() if token not in rarewords])


def trump_related(text, trump_string, feature):
    df[feature] = df[text].apply(lambda x: trump_string in x)
    df[feature] = df[feature].astype(int)


def has_reply(comment, reply):
    """
    This function checks if a comment has a reply
    :param comment: string
    :param reply: string
    :return: integer 0 or 1
    """
    if comment in reply.values:
        return 1
        print("YES")
    else:
        return 0



# import pandas as pd
# pd.options#.mode.chained_assignment = None  # default='warn'
# if this doesn't work then you will have to run eda on all months first
comments = ["data/CommentsApril2018.csv",
            "data/CommentsFeb2017.csv",
            "data/CommentsFeb2018.csv",
            "data/CommentsJan2017.csv",
            "data/CommentsJan2018.csv",
            "data/CommentsMarch2017.csv",
            "data/CommentsMarch2018.csv",
            "data/CommentsMay2017.csv"]
# leave April 2017 cause the columns are formatted in a different way
articles = ["data/ArticlesApril2018.csv",
            "data/ArticlesFeb2017.csv",
            "data/ArticlesFeb2018.csv",
            "data/ArticlesJan2017.csv",
            "data/ArticlesJan2018.csv",
            "data/ArticlesMarch2017.csv",
            "data/ArticlesMarch2018.csv",
            "data/ArticlesMay2017.csv"]

# df_cM17 = pd.read_csv("data/cleanedCommentsM17.csv",
#                      usecols=['commentBody', 'articleID', 'recommendations', 'parentID',
#                               'commentID', 'depth', 'approveDate'])
# df_aM17 = pd.read_csv("data/cleanedArticlesM17.csv", usecols=['articleID', 'articleWordCount', 'keywords', 'newDesk',
#                                                              'pubDate', 'sectionName', 'typeOfMaterial'], index_col=0)
# df = pd.merge(df_aM17, df_cM17, on='articleID')  # Merge articles with corresponding comments

data_frames = []
for i in range(8):
    pd.set_option('display.max_columns', None)
    df_cM17 = pd.read_csv(comments[i], #dtype='string',
                          usecols=['articleID', 'approveDate', 'commentBody', 'commentID', 'depth', 'parentID',
                                   'recommendations'])
    print(df_cM17.head())
    df_aM17 = pd.read_csv(articles[i], #dtype='string',
                          usecols=['articleID', 'articleWordCount', 'keywords', 'newDesk',
                                   'pubDate'], index_col=0)
    print(df_aM17.head())
    curr_df = pd.merge(df_aM17,df_cM17, on=['articleID'])  # Merge articles with corresponding comments
    curr_df.columns = ['articleID',
                       'articleWordCount', 'keywords', 'newDesk',
                       'pubDate','approveDate', 'commentBody', 'commentID', 'depth', 'parentID', 'recommendations']
    curr_df.rename(
        columns={
            'articleID': 'article_id',
            'articleWordCount': 'article_word_count',
            'commentBody': 'comment_body',
            'newDesk': 'new_desk',
            'pubDate': 'pub_date',
            'approveDate': 'approve_date',
            'commentID': 'comment_id',
            'parentID': 'parent_id'},

        inplace=True)
    curr_df = curr_df.loc[curr_df['new_desk'] == 'National']
    remove_nan_rows(curr_df)

    print("TEST")
    print(curr_df.head())
    data_frames.append(curr_df)

pd.set_option('display.max_columns', None)
for _, df in enumerate(data_frames):
    # df = df.dropna(how='all').dropna(how='all', axis=1)
    if len(df) > 0:
        print(df.head())
        print(df.shape)

        remove_nan_rows(df)
        print(df.head())
        print(df.tail())
        # Tokens for POS tagging and lemmatisation later
        df['text'] = tokenize(df['comment_body'])
        df['text'] = df['text'].apply(remove_punct_tokens)

        # Text processing with strings BEFORE removing punctuation and stopwords
        df['comment_word_count'] = df['comment_body'].apply(word_count)
        print(df['comment_word_count'])
        df['start_question'] = df['comment_body'].apply(starts_with_question_word)
        df['question_mark'] = df['comment_body'].apply(question_mark)
        df['exclamation_mark'] = df['comment_body'].apply(exclamation_mark)

        # Remove urls
        df['comment'] = df['comment_body'].str.replace('<a\s[^>]*.*?<\/a>', '')
        print('COMMENTS:')
        print(df['comment_body'])
        # Remove contractions
        df['comment'] = df['comment'].apply(lambda x: [contractions.fix(token) for token in str(x).split()])
        df['comment'] = [' '.join(map(str, x)) for x in df['comment']]
        df['comment'] = df['comment'].str.replace('manderings', ' ')
        df['comment'] = df['comment'].str.replace('<br/><br/>', ' ')  # Remove carraige returns
        df['comment'] = df['comment'].str.replace('<br/>', ' ')  # Remove spaces
        df['comment'] = df['comment'].str.replace('/', ' ')  # Handle words like traditional/modern
        df['comment'] = df['comment'].str.replace('.', '. ')  # Handle words like inside.and
        df['comment'] = df['comment'].str.replace('-', ' ')  # Handle words like china-basing
        df['comment'] = df['comment'].str.replace(',',
                                                  ', ')  # Change words like perhaps,given to perhaps, given
        df['comment'] = df['comment'].str.replace('?',
                                                  '? ')  # Change words like better?read to better? read
        df['comment'] = df['comment'].apply(remove_punctuation)  # Remove punctuation
        df['comment'] = df['comment'].str.replace('\d+', '')  # Remove numbers

        print("finished removing puncation")
        # sentiment of comments
        df['sentiment'] = df['comment'].map(lambda text: TextBlob(text).sentiment.polarity)
        print("finsihed calculated sentiment")

        # Tokenize
        df['comment'] = tokenize(df['comment'])
        print("finsihed tokenize")
        # Lemmatisation
        df['comment'] = df['comment'].apply(pos_tagging)
        df['comment'] = df['comment'].apply(change_pos_tag)
        df['comment'] = df['comment'].apply(lemmatise)
        print("finsihed lemmatise")
        df['comment'] = df['comment'].apply(lambda x: [token.lower() for token in x])
        df['comment'] = df['comment'].apply(remove_stopwords)
        print(df['comment'])
        print("finsihed removing stopwords")
        # cnt = display_most_frequentwords(df['commentBody'], 100)
        # print(cnt)
        # rarewords = most_rarewords(cnt, 100)
        # df['commentBody'] = df['commentBody'].apply(lambda x: remove_rarewords(x, rarewords))  # remove rare words

        trump_related('keywords', 'Trump, Donald J', 'trump_article')  # Indicates if the article is Trump related
        trump_related('comment', 'trump', 'trump_comment')  # Indicates if the article is Trump related in comment

        df['comment_count'] = df.groupby('article_id')['article_id'].transform(
            'count')  # Count of comments on a article

        df['gets_reply'] = df['comment_id'].apply(lambda x: has_reply(x, df.loc[df['depth'] == 2.0, [
                                                      'parent_id']]))  # Indicates if a comment has received a reply
        print("finsihed getting replies")
        df['pub_length'] = df.apply(lambda x: (dt.datetime.fromtimestamp(int(x['approve_date'])) -
                                               dt.datetime.strptime(x['pub_date'],
                                                                    '%Y-%m-%d %H:%M:%S')).total_seconds(),
                                    axis=1)  # Time between article published and comment posted
        print("finished pub length")
        # Preprocess keywords
        # Preprocess  section Name must group middle east and asia pacific together
        # unbalanced by the looks of it


        #df = df.drop(columns=['approve_date', 'pub_date', 'depth', 'comment_id',
        #                      'text', 'comment_body', 'parent_id'])
        #df.to_csv(mode='a', path_or_buf='data/RE_TEST_AGAIN.csv', index=False)
        print('TESTING BALANCED DATA ')
        print(df['gets_reply'].value_counts())
        print(df.head())
# the indexes are bolloxed
the_motherload = pd.concat(data_frames, ignore_index=True)
#s = the_motherload['new_desk'].value_counts()
#the_motherload = the_motherload[the_motherload.isin(s.index[s > 20000]).values]
#the_motherload = the_motherload.groupby('new_desk', group_keys=False).apply(lambda x: x.sample(20000))
#print(the_motherload['new_desk'].value_counts())
the_motherload.drop(columns=['approve_date', 'pub_date', 'depth', 'comment_id',
                              'text', 'comment_body', 'parent_id'])
print("Length of dataframe:")
print(the_motherload.shape)
print('Gets_Reply Balance')
print(the_motherload['gets_reply'].value_counts())
print('Depth')
print(the_motherload['depth'].value_counts())
print('Parent_id')
print(the_motherload['parent_id'].value_counts())
# so now I need to balance the data
the_motherload.to_csv(path_or_buf='data/balancedNationalBigger.csv')

# something going wrong with replies

# df = df.drop(columns=['approve_date', 'pub_date', 'depth', 'comment_id',
#                      'text', 'comment_body', 'parent_id'])

# df.to_csv(path_or_buf="data/featuresCommentsTestEditorial.csv", index=False)

# df.sort_values('article_id', inplace=True)
# df.drop_duplicates(subset="article_id", keep="first", inplace=True)
# df.drop(columns=['comment', 'gets_reply', 'question_mark', 'exclamation_mark',
#                 'start_question', 'comment_word_count', 'recommendations', 'pub_length', 'sentiment'], inplace=True)
# df.to_csv(path_or_buf="data/featuresArticlesTestEditorial.csv", index=False)

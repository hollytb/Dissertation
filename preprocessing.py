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
    else:
        return 0


df_cM17 = pd.read_csv("data/cleanedCommentsM17.csv",
                      usecols=['commentBody', 'articleID', 'recommendations', 'parentID',
                               'commentID', 'depth', 'approveDate'])
df_aM17 = pd.read_csv("data/cleanedArticlesM17.csv", usecols=['articleID', 'articleWordCount', 'keywords', 'newDesk',
                                                              'pubDate', 'sectionName', 'typeOfMaterial'], index_col=0)
df = pd.merge(df_aM17, df_cM17, on='articleID')  # Merge articles with corresponding comments
#df = df.loc[df['newDesk'] == 'Magazine']

print(df.head())
print(df.shape)

df.rename(
    columns={
        'articleID': 'article_id',
        'articleWordCount': 'article_word_count',
        'commentBody': 'comment_body',
        'sectionName': 'section_name',
        'typeOfMaterial': 'type_of_material',
        'newDesk': 'new_desk',
        'pubDate': 'pub_date',
        'approveDate': 'approve_date',
        'commentID': 'comment_id',
        'parentID': 'parent_id'},
    inplace=True)



s = df['new_desk'].value_counts()
df = df[df.isin(s.index[s > 714]).values]
df = df.groupby('new_desk', group_keys=False).apply(lambda x: x.sample(714))
print(df['new_desk'].value_counts())

print(df.head())
print(df.tail())
# Tokens for POS tagging and lemmatisation later
df['text'] = tokenize(df['comment_body'])
df['text'] = df['text'].apply(remove_punct_tokens)

# Text processing with strings BEFORE removing punctuation and stopwords
df['comment_word_count'] = df['comment_body'].apply(word_count)
df['start_question'] = df['comment_body'].apply(starts_with_question_word)
df['question_mark'] = df['comment_body'].apply(question_mark)
df['exclamation_mark'] = df['comment_body'].apply(exclamation_mark)

# Remove urls
df['comment'] = df['comment_body'].str.replace('<a\s[^>]*.*?<\/a>', '')

# Remove contractions
df['comment'] = df['comment'].apply(lambda x: [contractions.fix(token) for token in str(x).split()])
df['comment'] = [' '.join(map(str, x)) for x in df['comment']]

df['comment'] = df['comment'].str.replace('<br/><br/>', ' ')  # Remove carraige returns
df['comment'] = df['comment'].str.replace('<br/>', ' ')  # Remove spaces
df['comment'] = df['comment'].str.replace('/', ' ')  # Handle words like traditional/modern
df['comment'] = df['comment'].str.replace('.', '. ')  # Handle words like inside.and
df['comment'] = df['comment'].str.replace('-', ' ')  # Handle words like china-basing
df['comment'] = df['comment'].str.replace(',', ', ')  # Change words like perhaps,given to perhaps, given
df['comment'] = df['comment'].str.replace('?', '? ')  # Change words like better?read to better? read
df['comment'] = df['comment'].apply(remove_punctuation)  # Remove punctuation
df['comment'] = df['comment'].str.replace('\d+', '')  # Remove numbers
print("finsihed removing puncation")
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
print("finsihed removing stopwords")
# cnt = display_most_frequentwords(df['commentBody'], 100)
# print(cnt)
# rarewords = most_rarewords(cnt, 100)
# df['commentBody'] = df['commentBody'].apply(lambda x: remove_rarewords(x, rarewords))  # remove rare words


trump_related('keywords', 'Trump, Donald J', 'trump_article')  # Indicates if the article is Trump related
trump_related('comment', 'trump', 'trump_comment')  # Indicates if the article is Trump related in comment

df['comment_count'] = df.groupby('article_id')['article_id'].transform('count')  # Count of comments on a article

df['gets_reply'] = df['comment_id'].apply(lambda x:
                                          has_reply(x, df.loc[df['depth'] == 2.0, [
                                              'parent_id']]))  # Indicates if a comment has received a reply
print("finsihed getting replies")
df['pub_length'] = df.apply(lambda x: (dt.datetime.fromtimestamp(int(x['approve_date'])) -
                                       dt.datetime.strptime(x['pub_date'], '%Y-%m-%d %H:%M:%S')).total_seconds(),
                            axis=1)  # Time between article published and comment posted
print("finsihed pub length")
# Preprocess keywords
# Preprocess  section Name must group middle east and asia pacific together
# unbalanced by the looks of it
##tokenizer = MWETokenizer([('Middle', 'East'), ('Asia', 'Pacific'), ('College', 'Basketball'), ('Lesson', 'Plans'),
##                          ('Art', '&', 'Design'), ('Pro', 'Football'), ('Pro', 'Basketball'),
##                          ("401(k)'s", 'and', 'Similar', 'Plans'), ('Energy', '&', 'Environment'), ('Personal', 'Tech'),
##                          ('College', 'Football'), ('Paying', 'for', 'College'), ('Insider', 'Events')])
##df['sectionName'] = df['sectionName'].apply(lambda x: tokenizer.tokenize(str(x).split()))


df = pd.get_dummies(df, columns=['type_of_material', 'new_desk'])

df.drop(columns=['approve_date', 'pub_date', 'depth', 'comment_id',
                 'text', 'comment_body', 'parent_id'], inplace=True)
df.to_csv(path_or_buf="data/featuresCommentsTest.csv", index=False)

df.sort_values('article_id', inplace=True)
df.drop_duplicates(subset="article_id", keep="first", inplace=True)
df.drop(columns=['comment', 'gets_reply', 'question_mark', 'exclamation_mark',
                 'start_question', 'comment_word_count', 'recommendations', 'pub_length', 'sentiment'], inplace=True)
df.to_csv(path_or_buf="data/featuresArticlesTest.csv", index=False)

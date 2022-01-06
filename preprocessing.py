import pandas as pd
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
import time
import numpy as np
import csv
import nltk
import contractions
import en_core_web_sm
import gensim.downloader as api
from spellchecker import SpellChecker
import re
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from nltk.tokenize import (MWETokenizer, word_tokenize)

nlp = en_core_web_sm.load()

nltk.download('words')

# pd.set_option('display.max_rows', 40)


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
    return (article.value_counts)


def question_mark(comment):
    return int('?' in comment)


def exclamation_mark(comment):
    return int('!' in comment)


def starts_with_digit(comment):
    return int(comment[0].isdigit())


def starts_with_question_word(comment):
    return int(comment.startswith(('what', 'where', 'when', 'who', 'why', 'whom', 'whose',
                                   'which', 'how', 'will', 'would', 'should', 'could',
                                   'do', 'did')))


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


s_time = time.time()
# chunked_data = pd.read_csv("data/CommentsMarch2017.csv", skipinitialspace=True, usecols=['commentBody'], chunksize=10000)
# , dtype='string', quotechar='"', skipinitialspace=True, usecols=['commentBody'],
df_cM17 = pd.read_csv("data/cleanedCommentsM17.csv", usecols=['commentBody',
                                                              'articleID'], index_col=0)  # , quoting=csv.QUOTE_NONE, error_bad_lines= False, low_memory=False
df_aM17 = pd.read_csv(
    "data/cleanedArticlesM17.csv", index_col=0)  # , quoting=csv.QUOTE_NONE, error_bad_lines= False, low_memory=False

df = pd.merge(df_aM17, df_cM17, on='articleID')
#df = df.loc[df['newDesk'] == 'Foreign']
print(df.head())
print(df.shape)

# tokens for POS tagging and lemmatisation later
df['text'] = tokenize(df['commentBody'])
df['text'] = df['text'].apply(remove_punct_tokens)

# make lowercase
# df['commentBody'] = df['commentBody'].str.lower()

# text processing with strings BEFORE removing punctuation and stopwords
# df['word_count'] = df['comment'].apply(word_count)
# df['question_mark'] = df['comment'].apply(question_mark)
# df['exclamation_mark'] = df['comment'].apply(exclamation_mark)
# df['start_digit'] = df['comment'].apply(starts_with_digit)
# df['start_question'] = df['comment'].apply(starts_with_question_word)

# remove 1. contractions

df['commentBody'] = df['commentBody'].apply(lambda x: [contractions.fix(token) for token in str(x).split()])
df['commentBody'] = [' '.join(map(str, l)) for l in df['commentBody']]

# remove 2. punctuation
df['commentBody'] = df['commentBody'].str.replace('<br/><br/>', ' ')  # remove <br/
df['commentBody'] = df['commentBody'].str.replace('<br/>', '')
df['commentBody'] = df['commentBody'].apply(remove_punctuation)
df['commentBody'] = df['commentBody'].str.replace('\d+', '')  # remove 3. numbers

# tokenize
df['commentBody'] = tokenize(df.commentBody)

# text processing with tokens
# df['longest_word_len'] = df['commentBody'].apply(longest_word_len)
# df['avg_word_len'] = df['commentBody'].apply(avg_word_len)
# df['ratio_stopwords'] = df['commentBody'].apply(ratio_stopwords)

# lemmatisation
df['commentBody'] = df['commentBody'].apply(pos_tagging)
df['commentBody'] = df['commentBody'].apply(change_pos_tag)
df['commentBody'] = df['commentBody'].apply(lemmatise)
df['commentBody'] = df['commentBody'].apply(lambda x: [token.lower() for token in x])
df['commentBody'] = df['commentBody'].apply(remove_stopwords)

cnt = display_most_frequentwords(df['commentBody'], 100)
rarewords = most_rarewords(cnt, 100)
df['commentBody'] = df['commentBody'].apply(lambda x: remove_rarewords(x, rarewords))  # remove rare words

# Other features:
# df['comment_count'] = df.groupby('articleID').count()
df['comment_count'] = df.groupby('articleID')['articleID'].transform('count')


# preprocess keywords
# preprocess  section Name must group middle east and asia pacific together
# unbalanced by the looks of it
tokenizer = MWETokenizer([('Middle', 'East'), ('Asia', 'Pacific'), ('College', 'Basketball'), ('Lesson', 'Plans'),
                          ('Art', '&', 'Design'), ('Pro', 'Football'), ('Pro', 'Basketball'),
                          ("401(k)'s", 'and', 'Similar', 'Plans'), ('Energy', '&', 'Environment'), ('Personal', 'Tech'),
                          ('College', 'Football'), ('Paying', 'for', 'College'), ('Insider', 'Events')])
df['sectionName'] = df['sectionName'].apply(lambda x: tokenizer.tokenize(str(x).split()))
# with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):  # more options can be specified also
#    print(df['comment_count'])


df.drop(columns=['source', 'webURL', 'byline', 'headline', 'snippet', 'text', 'documentType', 'typeOfMaterial',
                 'newDesk', 'pubDate'], inplace=True)
print(df.head())
df.to_csv(path_or_buf="data/features.csv", index=False)  # , na_rep='Unknown'
e_time = time.time()
print('Time:', (e_time - s_time), 'seconds')

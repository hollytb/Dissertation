import pandas as pd
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 40)

my_punctuation = '€£' + punctuation


def tokenize(comment):
    tokens = [word_tokenize(x) for x in comment]
    return tokens


def remove_punct_tokens(tokens):
    return [token for token in tokens if token.isalnum()]


def word_count(comment):
    return len(comment.split())


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
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


df = pd.read_csv("data/comments.csv", quotechar='"', skipinitialspace=True, dtype='string')

# tokens for POS tagging and lemmatisation later
df['text'] = tokenize(df['comment'])

df['text'] = df['text'].apply(remove_punct_tokens)

# make lowercase
df['comment'] = df['comment'].str.lower()

# text processing with strings BEFORE removing punctuation and stopwords
df['word_count'] = df['comment'].apply(word_count)
df['question_mark'] = df['comment'].apply(question_mark)
df['exclamation_mark'] = df['comment'].apply(exclamation_mark)
df['start_digit'] = df['comment'].apply(starts_with_digit)
df['start_question'] = df['comment'].apply(starts_with_question_word)

# remove punctuation
df['comment'] = df['comment'].apply(remove_punctuation)

# tokenize
df['comment'] = tokenize(df.comment)

# text processing with tokens
df['longest_word_len'] = df['comment'].apply(longest_word_len)
df['avg_word_len'] = df['comment'].apply(avg_word_len)
df['ratio_stopwords'] = df['comment'].apply(ratio_stopwords)

# lemmatisation
df['text'] = df['text'].apply(pos_tagging)
df['text'] = df['text'].apply(change_pos_tag)
df['text'] = df['text'].apply(lemmatise)
df['text'] = df['text'].apply(lambda x: [token.lower() for token in x])
df['text'] = df['text'].apply(remove_stopwords)

df.to_csv(path_or_buf="data/features.csv")

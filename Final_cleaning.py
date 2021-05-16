import pandas as pd
import re
import nltk
import argparse
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel


def read_and_trim(file):
    '''
    Description:
        - Read and trim dataset
    Input:
        - File name
    Return:
        - Dataset with proper columns
    '''
    df = pd.read_csv(file)
    df = df[['text','tag']]
    return df

def get_wordnet_pos(word):
    '''
    Description:
        - Map POS tag to first character lemmatize() accepts
    Input:
        - Word
    Return:
        - Word POS tag information
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def clean_sent(sent):
    '''
    Description:
        - (Data is already all lowercased)
        - Remove punctations
        - Lemmatize
        - Remove stopwords
    Input:
        - Uncleaned sentence
    Return:
        - Cleaned tokens joined by whitespace
    '''
    tokens = word_tokenize(sent)
    tokens = [i for i in tokens if not re.match(r'\W',i)]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in tokens]
    stop = set(stopwords.words('english'))
    tokens = [i for i in tokens if i not in stop]
    return ' '.join(tokens)

def clean_and_split(df,mac=False):
    '''
    Description:
        - Apply cleaning
        - Train, test split
    Input:
        - Dataframe
    Return:
        - Train tuple and test tuple
    '''
    if mac:
        pandarallel.initialize()
        df['clean_text'] = df['text'].parallel_apply(clean_sent)
    else:
        df['clean_text'] = df['text'].apply(clean_sent)
    X_train, X_test, Y_train, Y_test = train_test_split(df['clean_text'],df['tag'],
                                                        test_size = 0.2, random_state = 42)
    return (X_train,Y_train),(X_test,Y_test)

def read_and_clean(file, mac=False):
    df = read_and_trim(file)
    train,test = clean_and_split(df,mac)
    return train,test

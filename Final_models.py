from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from nltk.stem.porter import PorterStemmer
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from Final_cleaning import read_and_clean
import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd
from numpy.random import seed
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping

def bag_of_words(dataset):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=800, stop_words='english')
    bow = bow_vectorizer.fit_transform(dataset['reviews'])
    return bow

def tokenizer(text):
    return text.split()

def tokenizer_stemmer(text):
    porter=PorterStemmer()
    return[porter.stem(word) for word in text.split()]

def W2V(dataset):
    token = dataset['reviews'].apply(lambda x: x.split())
    model_w2v = gensim.models.Word2Vec(token,size=100,window=5,
        min_count=2,sg = 1, hs = 0,negative = 10, workers= 8,seed = 34)
    model_w2v.train(token, total_examples= len(dataset['reviews']), epochs=20)
    return model_w2v

#size=no. of features/independent variables
def word2vec(token,size, my_w2vmodel):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for i in token:
        try:
            vec += my_w2vmodel[i].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
             continue
    if count != 0:
        vec /= count
    return vec

def wordtovector(dataset, my_w2vmodel):
    token = dataset['reviews'].apply(lambda x: x.split())
    wordvec_arrays = np.zeros((len(token), 100))
    for i in range(len(token)):
        wordvec_arrays[i,:] = word2vec(token[i], 100, my_w2vmodel)
    df_w2vec = pd.DataFrame(wordvec_arrays)
    return df_w2vec

#size=no of features/independent variables
def word2vec_sum(token,size, my_w2vmodel):
    vec = np.zeros(size).reshape((1, size))
    #count = 0.
    for i in token:
        try:
            vec += my_w2vmodel[i].reshape((1, size))
            #count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
             continue
    # if count != 0:
    #     vec /= count
    return vec

def wordtovector_sum(dataset, my_w2vmodel):
    token = dataset['reviews'].apply(lambda x: x.split())
    wordvec_arrays = np.zeros((len(token), 100))
    for i in range(len(token)):
        wordvec_arrays[i,:] = word2vec_sum(token[i], 100, my_w2vmodel)
    df_w2vec = pd.DataFrame(wordvec_arrays)
    return df_w2vec

def get_max_value(martix):

    res = []
    for i in range(martix.shape[1]):
        max_value = 0
        for j in range(martix.shape[0]):
            if martix[j][i] > max_value:
                max_value = martix[j][i]
        res.append(max_value)
    return np.array(res)

#size=no of features/independent variables
def word2vec_max(token,size, my_w2vmodel):
    vec = np.zeros(size).reshape((1, size))
    #count = 0.
    for i in token:
        try:
            tmp = my_w2vmodel[i].reshape((1, size))
            vec = np.append(vec, tmp, axis=0)
            #count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
             continue
    # if count != 0:
    #     vec /= count
    vec = get_max_value(vec)
    return vec

def wordtovector_max(dataset, my_w2vmodel):
    token = dataset['reviews'].apply(lambda x: x.split())
    wordvec_arrays = np.zeros((len(token), 100))
    for i in range(len(token)):
        wordvec_arrays[i,:] = word2vec_max(token[i], 100, my_w2vmodel)
    df_w2vec = pd.DataFrame(wordvec_arrays)
    return df_w2vec

def mean_vec(x):
  mean = np.zeros((1, 100))
  for i in range(x.shape[0]):
    vec = np.zeros(100).reshape((1, 100))
    for j in range(x.shape[1]):
       vec += x[i,j,:]
    vec /= x.shape[1]
    mean = np.append(mean, vec, axis=0)
  return mean[1:]

def sum_vec(x):
  sum = np.zeros((1, 100))
  for i in range(x.shape[0]):
    vec = np.zeros(100).reshape((1, 100))
    for j in range(x.shape[1]):
       vec += x[i,j,:]
    sum = np.append(sum, vec, axis=0)
  return sum[1:]

def get_max_value(martix):

    res = []
    for i in range(martix.shape[1]):
        max_value = 0
        for j in range(martix.shape[0]):
            if martix[j][i] > max_value:
                max_value = martix[j][i]
        res.append(max_value)
    return np.array(res)

def max_vec(x):
    max = np.zeros((1, 100))
    for i in range(x.shape[0]):
        # print(x[i,:].shape)
        vec = get_max_value(x[i,:])
        vec = np.array(vec).reshape(1, 100)
        max = np.append(max, vec, axis=0)
    return max[1:]

def read_data(train,test):
    (X_train, Y_train) = train
    (X_test, Y_test) = test

    le = LabelEncoder()
    le.fit(Y_train)
    Y_target_train = le.transform(Y_train)

    le = LabelEncoder()
    le.fit(Y_test)
    Y_target_test = le.transform(Y_test)

    data1 = pd.DataFrame(X_train)
    data11 = pd.DataFrame(X_test)
    data2 = pd.DataFrame(Y_target_train)
    data22 = pd.DataFrame(Y_target_test)

    new_data1 = pd.concat([data1,data11])
    new_data1=pd.DataFrame(new_data1)
    new_data2 = pd.concat([data2,data22])
    new_data2=pd.DataFrame(new_data2)

    new_data1.columns=['reviews']
    new_data1.index = range(len(new_data1))
    new_data2.columns=['setiment']
    new_data2.index = range(len(new_data2))
    #gain new_data
    new_data=pd.concat([new_data1,new_data2],join='outer',axis=1)
    return new_data

def BOW(new_data):
    X=bag_of_words(new_data)

    y = new_data.setiment.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2, shuffle=True)

    #LogisticRegression
    clf = LogisticRegressionCV(cv=5,
                                scoring='accuracy',
                                random_state=1,
                                n_jobs=-1,
                                max_iter=300).fit(X_train, y_train)

    print("BOW-LR:", clf.score(X_test,y_test))

    #svm
    clf = svm.SVC(kernel = 'linear')
    clf.fit(X_train, y_train)

    print("BOW-SVM:", clf.score(X_test,y_test))

def tf_idf(new_data):
    tfidf = TfidfVectorizer(strip_accents=None,
                                    lowercase=True,
                                    preprocessor=None, # defined preprocessor in Data Cleaning
                                    tokenizer=tokenizer_stemmer,
                                    use_idf=True,
                                    norm='l2',
                                    smooth_idf=True)

    X = tfidf.fit_transform(new_data.reviews)
    y = new_data.setiment.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2, shuffle=True)

    #LogisticRegression
    clf = LogisticRegressionCV(cv=5,
                                scoring='accuracy',
                                random_state=1,
                                n_jobs=-1,
                                max_iter=300).fit(X_train, y_train)

    print("TFIDF-LR:",clf.score(X_test,y_test))

    #svm
    clf = svm.SVC(kernel = 'linear')
    clf.fit(X_train, y_train)

    print("TFIDF-SVM",clf.score(X_test,y_test))

def cnn(new_data):
    X = new_data['reviews']
    y = new_data.setiment.values

    MAX_NB_WORDS = 30000
    MAX_SEQUENCE_LENGTH = 80
    EMBEDDING_DIM = 100

    tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"$%&()*,-./:;<=>?@[\]^_`{|}~', split = ' ', lower=False)
    tokenizer.fit_on_texts(X)
    word_index = tokenizer.word_index

    X = tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    Y = pd.get_dummies(y).values

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 1)

    x_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(x_input)
    ################### add CNN layer
    kernel_sizes = [1,2,3]
    pooled_outputs = []
    for i in kernel_sizes:
        conv = Conv1D(filters = 512, kernel_size = i, padding = 'valid',
                      activation = 'relu', kernel_initializer = 'he_uniform')(x)
        pool = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - i + 1)(conv)
        pooled_outputs.append(pool)
    ###################
    merged = Concatenate(axis=-1)(pooled_outputs)
    flatten = Flatten()(merged)
    drop = Dropout(rate=0.4)(flatten)
    x_output = Dense(2, kernel_initializer='he_uniform', activation='softmax')(drop)
    model = Model(inputs=x_input, outputs=x_output)
    opt = tfk.optimizers.Adam(learning_rate=0.05)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 40
    batch_size = 512

    history = model.fit(X_train, Y_train, epochs=epochs,
                        batch_size=batch_size,validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_accuracy', patience=10,
                                                 min_delta=0.001, restore_best_weights = True)])
    accr = model.evaluate(X_test,Y_test)
    print('Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.4f}'.format(accr[0],accr[1]))

def word_2_vec_vector(new_data):
    my_w2vmodel=W2V(new_data)

    X=wordtovector(new_data, my_w2vmodel) # - mean
    X_sum=wordtovector_sum(new_data, my_w2vmodel) # - sum
    X_max=wordtovector_max(new_data, my_w2vmodel) #- max

    y = new_data.setiment.values

    for i,j in zip([X,X_sum,X_max], ["MEAN","SUM","MAX"]):
        X_train, X_test, y_train, y_test = train_test_split(i, y, random_state=1, test_size=0.2, shuffle=True)

        #LogisticRegression
        clf = LogisticRegressionCV(cv=5,
                                    scoring='accuracy',
                                    random_state=1,
                                    n_jobs=-1,
                                    max_iter=500).fit(X_train, y_train)

        print("W2V-{}-LR:".format(j),clf.score(X_test,y_test))

        #svm
        clf = svm.SVC(kernel = 'linear')
        clf.fit(X_train, y_train)

        print("W2V-{}-SVM:".format(j),clf.score(X_test,y_test))

def cnn_vector(new_data,path2):
    model = load_model(path2)
    X_train = new_data['reviews']
    MAX_NB_WORDS = 30000
    MAX_SEQUENCE_LENGTH = 80
    EMBEDDING_DIM = 100

    tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"$%&()*,-./:;<=>?@[\]^_`{|}~', split = ' ', lower=False)
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)

    Y_train = new_data['setiment']

    le = LabelEncoder()
    le.fit(Y_train)
    Y_target_train = le.transform(Y_train)
    y = pd.get_dummies(Y_train).values

    input = model.get_layer("input_12")
    embedding = model.get_layer("embedding_11")

    x = input(X_train)
    x = embedding(x)
    x = x.numpy()
    #x = sum_vec(x)
    x_mean = mean_vec(x)
    x_sum = sum_vec(x)
    x_max = max_vec(x)

    for x,j in zip([x_mean,x_sum,x_max],['MEAN','SUM','MAX']):

        X_train, X_test, y_train, y_test = train_test_split(x, Y_target_train, random_state=1, test_size=0.2, shuffle=True)

        #LogisticRegression mean
        clf = LogisticRegressionCV(cv=5,
                                    scoring='accuracy',
                                    random_state=1,
                                    n_jobs=-1,
                                    max_iter=500).fit(X_train, y_train)

        print("CNN-{}-LR:".format(j),clf.score(X_test,y_test))

        #svm mean
        clf = svm.SVC(kernel = 'linear')
        clf.fit(X_train, y_train)

        print("CNN-{}-SVM:".format(j),clf.score(X_test,y_test))


def main(path1, path2):

  #read data
  print('Reading and cleaning data...')
  train,test = read_and_clean(path1)
  new_data = read_data(train,test)

  ###################bag_of_words Logistic Regression and SVM
  print('Fitting BOW...')
  BOW(new_data)

  ####################tf-idf Logistic Regression and SVM
  print('Fitting tf-idf...')
  tf_idf(new_data)

  ####################cnn Logistic Regression and SVM
  print('Fitting CNN...')
  cnn(new_data)

  ######################W2V mean,sum,max with Logistic Regression and SVM
  print('Fitting w2v embeddings...')
  word_2_vec_vector(new_data)

  ####################CNN mean,sum,max with Logistic Regression and SVM
  print('Fitting trained CNN embeddings...')
  cnn_vector(new_data,path2)


import argparse
if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='project')
  parser.add_argument('--path1', type=str, default="dataset.csv",
                      help='path to dataset')
  parser.add_argument('--path2', type=str, default="CNN_weights_8425.h5",
                      help='path to model')

  args = parser.parse_args()
  #main(args.path)

  main(args.path1, args.path2)

import pandas as pd
import re
import nltk
import argparse
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel


import csv
review_dataset_path="movie_reviews"
print(os.listdir(review_dataset_path))

#Positive and negative reviews folder paths
pos_review_folder_path=review_dataset_path+"/"+"pos"
neg_review_folder_path=review_dataset_path+"/"+"neg"

#Positive and negative file names
pos_review_file_names=os.listdir(pos_review_folder_path)
neg_review_file_names=os.listdir(neg_review_folder_path)


def load_text_from_textfile(path):
    file = open(path, "r")
    review = file.read()
    file.close()
    return review


def load_review_from_textfile(path):
    return load_text_from_textfile(path)

def get_data_target(folder_path, file_names, review_type):
    data=list()
    target =list()
    for file_name in file_names:
        full_path = folder_path + "/" + file_name
        review =load_review_from_textfile(path=full_path)
        data.append(review)
        target.append(review_type)
    return data, target

pos_data, pos_target=get_data_target(folder_path=pos_review_folder_path,
               file_names=pos_review_file_names,
               review_type="positive")
neg_data, neg_target = get_data_target(folder_path = neg_review_folder_path,
                                      file_names= neg_review_file_names,
                                      review_type="negative")

data = pos_data + neg_data
target_ = pos_target + neg_target

data1=pd.DataFrame(data)

data2=pd.DataFrame(target_)

new_data=pd.concat([data1,data2],join='outer',axis=1)
print(new_data)
new_data.columns=['text','tag']
new_data = pd.DataFrame(new_data)
new_data.to_csv("dataset.csv", sep=',', index=True, header=True)


## Spring 2021 ANLY521 Final Project 

#### Team Member: Dongru Jia, Xiwen Zhang




## Overview
In this project, Sentiment Analysis is applied to predict the sentiment of each movie review by ‘reading’ the review text. We approached this problem by implementing multiple machine learning and deep learning algorithms as well as experimenting various word embedding methods. The models included in this project are Logistic Regression, Support Vector Machine, and Convolutional Neural Network. The word embedding methods applied are Bag of Words, Tf-idf, Word2Vec, and a self-trained word embedding by the CNN model.
Among all these models we utilized, 
the Support Vector Machine (SVM) classifier on TF-IDF gained the best accuracy of 80.75%, 
and Logistic Regression achieved the highest accuracy of 82.75% on TF-IDF 
and Convolutional Neural Network (CNN) performed the highest accuracy of 84.25%
 

## Files  Description

### `Requirements.txt` 
Library requirements documented

### `movie_review : pos & neg folder`
This folder includeds the data files we applied in this project. "pos" folder includes 1000 positive review files and "neg" folder includes 1000 negative review files. 
It is from Kaggle https://www.kaggle.com/nltkdata/movie-review

### `data_combine.py` 
2000 data files were combined into one csv file.

### `dataset.csv`
The dataset atfer doing combination. 

### `project_cleaning.py` 
The data-cleaning steps.

### `final_project.py` 
Include vecorization and model function 

### `.py` 
The saved weight file of the CNN model.


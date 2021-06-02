import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

class dataloader():
    def __init__(self,path):
        self.path = path


        #load csv file

        ratings_df = pd.read_csv(os.path.join(path, "ratings.csv"), encoding='utf-8')
        movies_df = pd.read_csv(os.path.join(path, 'movies.csv'), encoding='utf-8')

        #genre dummy
        genres_onehot_df = movies_df.genres.str.get_dummies("|")
        #genre_split ==> 나중에 cross var 위해 사용
        movies_df.genres = movies_df.genres.str.split("|")

        #genre onehot added
        movies_df = pd.concat([movies_df, genres_onehot_df], axis=1)
        #extract year
        movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))').astype("str")
        movies_df['year'] = movies_df['year'].apply(lambda x: x.replace("(", "").replace(")", ""))
        movies_df.year = movies_df.year.astype("float32").astype("int32")
        movies_df.drop(movies_df[movies_df.year == 0].index, inplace=True)
        movies_df.drop("title", axis=1, inplace=True)
        #make year categorical feature
        bins = list(range(1900, 2021, 20))
        labels = list(range(len(bins) - 1))
        year_label = []
        for i in range(len(labels)):
            year_label.append(f"year_{i}")
        year_level_df = pd.cut(movies_df.year, bins, labels=year_label, right=False)
        movies_df["year_level"] = year_level_df
        year_onehot_df = movies_df.year_level.astype('str').str.get_dummies()
        #year added
        movies_df = pd.concat([movies_df, year_onehot_df], axis=1)


        #cross feature

        movies_df["cross_var"] = movies_df.apply(lambda x: [x["year_level"] + "_" + genre for genre in x["genres"]],
                                                 axis=1)

        mlb = MultiLabelBinarizer()

        cross_var_df = pd.DataFrame(mlb.fit_transform(movies_df["cross_var"]), columns=mlb.classes_,
                                    index=movies_df.index)
        #cross var added
        movies_df = pd.concat([movies_df, cross_var_df], axis=1)

        #안쓰는 col drop
        movies_df.drop(["genres", "cross_var", "year", "year_level"], axis=1, inplace=True)


        #merge ratings_df with final_movies_df

        ratings_df = pd.merge(ratings_df, movies_df, "inner", on="movieId")

        ratings_df.drop('timestamp', axis=1, inplace=True)

        self.target_df = ratings_df["rating"]

        ratings_df.drop("rating", axis=1, inplace=True)
        self.X = ratings_df


    def make_binary_set(self,with_cross_var=True,test_size=0.1):
        binary_target_df = self.target_df.apply(lambda x: np.where(x < 4.0, 0, 1))
        if with_cross_var == False:
            X_without_cross_var = self.X.iloc[:, :28]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_without_cross_var, binary_target_df,
                                                                                    test_size=0.1)
            return self.X_train, self.X_test, self.y_train, self.y_test
        self. X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, binary_target_df, test_size=test_size)

        return self.X_train,self.X_test,self.y_train,self.y_test



    def make_regression_set(self,with_cross_var=True,test_size=0.1):
        if with_cross_var == False:
            X_without_cross_var = self.X.iloc[:, :28]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_without_cross_var, self.target_df,
                                                                                    test_size=0.1)
            return self.X_train, self.X_test, self.y_train, self.y_test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.target_df, test_size=test_size)

        return self.X_train, self.X_test, self.y_train, self.y_test
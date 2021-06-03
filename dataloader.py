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
from sklearn.preprocessing import StandardScaler


class dataloader():
    def __init__(self,path):
        self.path = path


        #load csv file
        tags_df = pd.read_csv(os.path.join(path, "tags.csv"), encoding='utf-8')
        ratings_df = pd.read_csv(os.path.join(path, "ratings.csv"), encoding='utf-8')
        movies_df = pd.read_csv(os.path.join(path, 'movies.csv'), encoding='utf-8')

        #tags sentiment score
        tag_value_count = tags_df["tag"].value_counts()
        # num over 3 tags
        tags_df['tag'] = tags_df["tag"].apply(lambda x: np.where(tag_value_count[x] >= 3, x, None))
        tags_ratings_df = pd.merge(tags_df, ratings_df, on=["movieId", "userId"])
        tags = tags_df.tag.unique()

        tags_rating_sum = dict.fromkeys(tags, 0)
        print("making sentiment tags_df")
        for tag in tqdm(tags_rating_sum):
            for index, row in tags_ratings_df.iterrows():
                if row["tag"] == tag:
                    tags_rating_sum[tag] += row["rating"]
        tags_rating_sum.pop(None)

        tags_sentiment_score = tags_rating_sum
        for tag, score_sum in tqdm(tags_sentiment_score.items()):
            tags_sentiment_score[tag] = score_sum / tag_value_count[tag]

        # tags_df 에 감성점수 추가
        tags_df["sentiment"] = 0.0

        for index, row in tags_df.iterrows():
            tag = row["tag"]
            if (tag in tags_sentiment_score):
                tags_df.at[index, "sentiment"] = tags_sentiment_score[tag]

        movie_sentiment_avg_df = tags_df.groupby("movieId").mean()["sentiment"]
        ratings_sentiment_df = pd.merge(ratings_df, movie_sentiment_avg_df, how="left", on=["movieId"])

        ratings_sentiment_df = ratings_sentiment_df.fillna(0)


        #genre dummy
        genres_onehot_df = movies_df.genres.str.get_dummies("|")
        #genre_split ==> 나중에 cross var 위해 사용
        movies_df.genres = movies_df.genres.str.split("|")

        #genre onehot added
        movies_df = pd.concat([movies_df, genres_onehot_df], axis=1)
        movies_df = movies_df.drop((movies_df[movies_df["(no genres listed)"] == 1]).index, axis=0)
        movies_df.drop("(no genres listed)", axis=1, inplace=True)
        #extract year
        movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))').astype("str")
        movies_df['year'] = movies_df['year'].apply(lambda x: x.replace("(", "").replace(")", ""))
        movies_df.year = movies_df.year.astype("float32")
        movies_df.year = movies_df.year.fillna(0)
        movies_df.drop(movies_df[movies_df.year == 0].index, inplace=True)
        movies_df.year = movies_df.year.astype("int32")
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
        movies_df = pd.concat([movies_df, year_onehot_df], axis=1)


        #num rated per user, per movie
        user_ratings_count_df = ratings_sentiment_df.groupby("userId")["timestamp"].count()
        movie_ratings_count_df = ratings_sentiment_df.groupby("movieId").count()["userId"]
        ratings_sentiment_df = pd.merge(ratings_sentiment_df, user_ratings_count_df, on="userId")
        ratings_sentiment_df = pd.merge(ratings_sentiment_df, movie_ratings_count_df, on="movieId")
        ratings_sentiment_df = ratings_sentiment_df.rename(
            columns={"userId_x": "userId", "timestamp_x": "timestamp", "timestamp_y": "user_num_rated",
                     "userId_y": "movie_num_rated"})



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


        #merge final concatenated dataset

        concated_ratings_df = pd.merge(ratings_sentiment_df, movies_df, "inner", on="movieId")

        concated_ratings_df.drop('timestamp', axis=1, inplace=True)
        #sentiment 0인 row 제거
        concated_ratings_df = concated_ratings_df.drop(concated_ratings_df[concated_ratings_df.sentiment == 0].index,
                                                       axis=0)



        self.target_df = concated_ratings_df["rating"]

        concated_ratings_df.drop("rating", axis=1, inplace=True)
        #drop userId , movieId
        concated_ratings_df_without_user_movie = concated_ratings_df.drop(["userId", "movieId"], axis=1)


        #scaling continuous cols
        scaler = StandardScaler()
        concated_ratings_df_without_user_movie[["user_num_rated", "movie_num_rated"]] = scaler.fit_transform(
            concated_ratings_df_without_user_movie[["user_num_rated", "movie_num_rated"]])
        self.X = concated_ratings_df_without_user_movie

        # self.embedding_cols = ["userId","movieId"]

    def make_binary_set(self,test_size=0.1):
        binary_target_df = self.target_df.apply(lambda x: np.where(x < 4.0, 0, 1))

        self.X_wide_train, self.X_wide_test, self.y_train, self.y_test = train_test_split(self.X, binary_target_df, test_size=test_size)

        self.X_deep_train = self.X_wide_train.iloc[:, :28]
        self.X_deep_test = self.X_wide_test.iloc[:, :28]

        return self.X_wide_train,self.X_deep_train,self.X_wide_test,self.X_deep_test,self.y_train,self.y_test



    def make_regression_set(self,test_size=0.1):

        self.X_wide_train, self.X_wide_test, self.y_train, self.y_test = train_test_split(self.X, self.target_df,
                                                                                          test_size=test_size)

        self.X_deep_train = self.X_wide_train.iloc[:, :28]
        self.X_deep_test = self.X_wide_test.iloc[:, :28]

        return self.X_wide_train, self.X_deep_train, self.X_wide_test, self.X_deep_test, self.y_train, self.y_test


import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

movies = pd.read_csv('/kaggle/input/movierecommenderdataset/movies.csv')
ratings = pd.read_csv('/kaggle/input/movierecommenderdataset/ratings.csv')
df = movies.merge(ratings, how="left", on="movieId")



### User Based Recommendation


def check_df(dataframe, head=5):
    print("################### Shape ####################")
    print(dataframe.shape)
    print("#################### Info #####################")
    print(dataframe.info())
    print("################### Nunique ###################")
    print(dataframe.nunique())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles #################")
    print(dataframe.describe([0,0.01, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("#################### Head ####################")
    print(dataframe.head(head))

check_df(df)


# We are deleting the missing values with a small number of missing values in the dataset
df.dropna(inplace=True)

# We are converting the number of times movies are rated into a dataframe.
comment_counts = pd.DataFrame(df["title"].value_counts())

# We are selecting the movies that have less than 50 ratings.
rare_movies = comment_counts[comment_counts["title"] <= 50].index

# From our dataset, we are removing the movies with less than 50 ratings.
common_movies = df[~df["title"].isin(rare_movies)]

# We are creating a table that includes the average ratings.
user_movie_df = common_movies.pivot_table("rating", "userId", "title")

user_movie_df. head()


# We are selecting a random user.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=20).values)

# The status of the selected user with all the movies.
random_user_df = user_movie_df[user_movie_df.index == random_user]

# We are finding the movies that the random user has watched and rated
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# We are finding the number of movies that the random user has watched and rated
len(movies_watched)


# We are selecting the status of all users with these 47 movies.
movies_watched_df = user_movie_df[movies_watched]

# We are finding how many of the 47 movies other users have watched
user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count.head()


# We are accessing the user IDs of those who have watched at least 20 movies out of the 47 movies.
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# We are examining the relationship of users who have watched at least 20 of the 47 films with these 47 films
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],     # 3203 rows x 33 columns
                      random_user_df[movies_watched]])

final_df.head()


# We are taking the correlations between the movies these users watched and the ratings they gave for each movie.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

# We are renaming the index names of corr_df as user_id_1 and user_id_2.
corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

corr_df.head()


# We select the users whose correlation with the random user is at least 0.50.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.50)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users.head()


# We add the columns userId, movieId, and rating.
top_users_ratings = top_users.merge(ratings[["userId", "movieId", "rating"]], how='inner')

# We remove the random user themselves, who has a correlation of 1.00 in the first row of the dataframe.
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# To obtain scores, we multiply the correlations with the ratings that the users gave to the movies.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.head()


# We are sorting the average weighted_rating scores by movies.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df.head()


# We select the ones with a score higher than 3 and sort them in descending order.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3].\
                         sort_values("weighted_rating", ascending=False)

# To see the movie titles corresponding to the MovieId, we perform this operation.
movies_to_be_recommend = movies_to_be_recommend.merge(movies[["movieId", "title"]])


movies_to_be_recommend.head(10)


# We can recommend to random user, movies with high correlation, watched by other users which are similar to the movies that the random user we selected watched and rated




### Item Based Recommendation


# We had created a user, movie, and rating table by removing films with less than 50 votes.
user_movie_df.head()

# Movies with less than 50 votes and all users


# We are selecting a random movie.
movie_name = pd.Series(user_movie_df.columns).sample(1, random_state=20).values[0]

print(movie_name)


# We are only seeing the variable of the selected film.
movie_name_df = user_movie_df[movie_name]

# We calculate the correlation of the randomly selected movie with all other movies and sort them in descending order.
movie_corr = user_movie_df.corrwith(movie_name_df).sort_values(ascending=False)

movie_corr.head(10)


# We are seeing at the films with the closest similarity pattern to the randomly selected film.
# This way, we can select the movie we want and find the movies with the closest rating patterns to that movie.¶




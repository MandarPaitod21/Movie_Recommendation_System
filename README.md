# Movie_Recommendation_System
# Load the libraries
import numpy as pd
import pandas as pd
# Load the Dataset
df1 = pd.read_csv('tmdb_5000_credits.csv')
df1.head()
# Information of the Data
df1.shape
df1.info()
# Movie Dataset
df2 = pd.read_csv('tmdb_5000_movies.csv')
df2.head()
df2.shape
df2.info()
# Merge the Dataframes
df1.columns = ['id','title','cast','crew']
df2 = df2.merge(df1,on = 'id')
df2.head()
df2.shape
df2.columns
C = df2['vote_average'].mean()
C
# Minimum Votes
m = df2['vote_count'].quantile(0.9)
m
# Listed Movies
lists_movies = df2.copy().loc[df2['vote_count'] >=m]
lists_movies.shape
# Define Function
def weighted_rating(x , m=m , C=C):
    v = x['vote_count']
    R = x['vote_average']
    #based on the formula (m=1838,c=6.09)
    return(v/(v+m) *R) + (m/(m+v) * C)
# define a new feature score and calculate its value
lists_movies['score'] = lists_movies.apply(weighted_rating , axis = 1) 
lists_movies.head()
lists_movies.shape
# sort movies based on scores
lists_movies = lists_movies.sort_values('score' , ascending = False)
# print the movies
lists_movies[['title_x','vote_count','vote_average','score']].head(10)
# Popular Movies
pop = df2.sort_values('popularity' , ascending = False)
import matplotlib.pyplot as plt
plt.figure(figsize = (12,4))
plt.barh(pop['title_x'].head(6),pop['popularity'].head(6),align = 'center' , color = 'm')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
df2.columns
# Budget
pop = df2.sort_values('budget' , ascending = False)
import matplotlib.pyplot as plt
plt.figure(figsize = (12,4))
plt.barh(pop['title_x'].head(6),pop['budget'].head(6),align = 'center' , color = 'green')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Budget Movies")
# Revenue
pop = df2.sort_values('revenue' , ascending = False)
import matplotlib.pyplot as plt
plt.figure(figsize = (12,4))
plt.barh(pop['title_x'].head(6),pop['revenue'].head(6),align = 'center' , color = 'blue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Revenue on Movies")
# Overview Column
df2['overview'].head(10)
from sklearn.feature_extraction.text import TfidfVectorizer
# define a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words = 'english')
# replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')
# construct the matrix
tfidf_matrix = tfidf.fit_transform(df2['overview'])
# output
tfidf_matrix.shape
# import linear kernel
from sklearn.metrics.pairwise import linear_kernel
# compute
cosine_sim = linear_kernel(tfidf_matrix , tfidf_matrix)
def get_recommendations(title, cosine_sim=cosine_sim):
# Get the index of the movie that matches the title
    idx = indices.get(title)
# Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
# Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
# Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
# Return the top 10 most similar movies
    return df2['title_x'].iloc[movie_indices]
    get_recommendations('The Dark Knight Rises')

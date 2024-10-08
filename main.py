from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def get_index_from_title(title,data_frame):
    return data_frame[data_frame.title == title]['index'].values[0]

def get_title_from_index(index,data_frame):
    return data_frame[data_frame.index == index]['title'].values[0]

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " "  +row['genres'] + " " + row['director']
    except:
        print("Error: ", row)


def get_top_movie_recs(movie_choice):
    #1 read csv file
    data_frame =pd.read_csv('movie_dataset.csv')

    #2 select features
    features = ['keywords','cast','genres','director']


    for feature in features:
        data_frame[feature] = data_frame[feature].fillna('') #Replaces Nan values with empty string

        
    #3 combine features
    # to use cosine similarity, need to create a column that contains all of the above features similar to the 'londond london paris' strings
    data_frame['similar'] = data_frame.apply(combine_features,axis=1)

    #step 4 create count matrix from this new combined collumn
    cv = CountVectorizer()

    count_matrix= cv.fit_transform(data_frame['similar'])
    #print(count_matrix)

    #5 cosine similarity of movies based on features 

    cosine_sim= cosine_similarity(count_matrix)

     
    index_choice = get_index_from_title(movie_choice,data_frame)

    similar_movies = cosine_sim[index_choice]
    similar_movies_enumerated = list(enumerate(similar_movies)) # returns list of tuples where first value is index of this row in the similarity matrix and second value is the cosine similarity value

    similar_movies_sorted = sorted(similar_movies_enumerated,key= lambda x:x[1],reverse=True)
    for i in range(10):
        print(str(similar_movies_sorted[i])+ get_title_from_index(similar_movies_sorted[i][0],data_frame)) 

#1 read csv file


#print(data_frame.head())
#print(data_frame.columns)

#2 select features

#3 combine features
# to use cosine similarity, need to create a column that contains all of the above features similar to the 'londond london paris' strings



#print column of combined featurse.
#print(data_frame['similar'].head())




#5 cosine similarity of movies based on features 



#6 take a movie and  return the top 5 most similar movies

movie_choice = "Captain America: Civil War"

get_top_movie_recs(movie_choice)
 








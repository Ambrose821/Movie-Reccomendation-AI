#Cosine similarity equation.
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
#example
text = ['London Paris London', 'Paris Paris London']

cv = CountVectorizer()
#This gives us our vector matrix(array of vectors)
count_matrix = cv.fit_transform(text).toarray() #Without to array it is a weird matrix formula from scikit toarray() is regular matrix that you are use to


print(count_matrix)
print(cosine_similarity(count_matrix))
import numpy as np
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

pd.set_option('max_colwidth',1000000)#display maximum columns width
pd.options.display.max_rows=1000000#display maximum rows
pd.options.display.max_columns=5
df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

df = df[['Title']]
# print(df['Title']=='Guardians of the Galaxy')
# print(df['Title'])
stopwords_list = stopwords.words('english')
vectorizer = TfidfVectorizer(analyzer='word')
#build book-title tfidf matrix
tfidf_matrix = vectorizer.fit_transform(df['Title'])

tfidf_feature_name = vectorizer.get_feature_names()
# print(tfidf_matrix.shape)

# comping cosine similarity matrix using linear_kernal of sklearn
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

df = df.reset_index(drop=True)
indices = pd.Series(df['Title'].index)

#Function to get the most similar books
def recommend(index, method):
    id = indices[index]
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(method[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    
    #Get the books index
    books_index = [i[0] for i in similarity_scores]
    
    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return df['Title'].iloc[books_index]

# #input the index of the book
# print(recommend(225, cosine_similarity))
# print(df.iloc[225])

recom = recommend(199, cosine_similarity)
for book in recom:
    print(book)
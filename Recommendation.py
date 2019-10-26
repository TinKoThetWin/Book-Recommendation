import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('max_colwidth',1000000)#display maximum columns width
pd.options.display.max_rows=1000000#display maximum rows
pd.options.display.max_columns=5
df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

df = df[['Title','Plot']]
df.set_index('Title',inplace=True)
# print(df.loc['Guardians of the Galax','Plot'])
#print(df.loc['Guardians of the Galaxy'])
# print(df.iloc[0:10])
#print(df.head())

# initializing the new column
df['bag_of_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']
    plot = plot.lower()
    
    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column for the corresponding movie
    keyword = list(key_words_dict_scores.keys())
    listToStr =' '.join(map(str,keyword))
    row['bag_of_words'] = listToStr

# dropping the Plot column
df.drop(columns = ['Plot'], inplace = True)
# df.bag_of_words =df.bag_of_words.astype(str)
#print(df.head())

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
indices = pd.Series(df.index)

#  defining the function that takes in movie title 
# as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim = cosine_sim):
    
    # initializing the empty list of recommended movies
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies

#print(recommendations('The godfather'))
recom = recommendations('Guardians of the Galaxy')
for book in recom:
    print(book)
    
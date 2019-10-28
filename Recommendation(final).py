import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import json
import string

pd.set_option('max_colwidth',1000000)#display maximum columns width
pd.options.display.max_rows=1000000#display maximum rows
pd.options.display.max_columns=5

jsons = [{"title":"Hello",
"abstract":["keyword1","keyword2","keyword3"]},
{"title":"god and devil world",
"abstract":["key1","key2","key3"]},
{"title":"Transcending the nine heavens",
"abstract":["ok","dollar","True"]}]
books= json.dumps(jsons) # convert list to json string
df = pd.read_json(books)
df.to_csv()
# print(df.head())
# print(csv)

# initializing the new column
df['final_text'] = ""
for index, row in df.iterrows():
    title=row['title']
    abstract=row['abstract']

    tokens = title.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w.lower() in stop_words]
    title =' '.join(map(str,tokens))
    abstract =' '.join(map(str,abstract))
    text = title+" "+abstract
    row['final_text'] = text

# dropping the abstract column
df.drop(columns = ['abstract'], inplace = True)
#print(df)

vectorizer = TfidfVectorizer(analyzer='word')
#build book-title tfidf matrix
tfidf_matrix = vectorizer.fit_transform(df['final_text'])
tfidf_feature_name = vectorizer.get_feature_names()

# comping cosine similarity matrix using linear_kernal of sklearn
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

df = df.reset_index(drop=True)
indices = pd.Series(df['title'].index)

#Function to get the most similar books
def recommend(index, method):
    id = indices[index]
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 10
    similarity_scores = list(enumerate(method[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    
    #Get the books index
    books_index = [i[0] for i in similarity_scores]
    
    #Return the top 10 most similar books using integar-location based indexing (iloc)
    return df['title'].iloc[books_index]

#input the index of the book
recom = recommend(0, cosine_similarity)

for book in recom:
    print(book)

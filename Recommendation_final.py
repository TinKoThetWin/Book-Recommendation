import pandas as pd
from flask import Flask, jsonify, make_response, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import json
import string

app = Flask(__name__)
#for bookId and Title json
output_json=[]
@app.route('/',methods=['POST'])
# @app.route('/')

def recommendation():
    bk = request.get_json()

    # ###### Example  #################
    # jsons = """{"book_id":2,
    # "book_title":"Hello",
    # "abstract":"fkjdkfaksd dkfja ldsjf",
    # "json_file": "keyword.json"}"""
    # books = json.loads(jsons) ## jsonstring
    # # books= json.dumps(books) # convert list to json string
    # bk = pd.DataFrame(books,index=[0]) ## books is json string
    # print(bk)
    # #################################

    bk_id = bk['book_id']
    path = bk['json_file']
    print(bk_id.values,path.values)
    num = str(bk_id.values)
    path = str(path.values)
    num = int(num.replace("[","").replace("]","")) ## chosen id from web
    print("chosen id",num)
    path = path.translate(str.maketrans({'[':'',']':'','\'':''}))
    print("path",path) ## path for keyword extracted json

    ## Start ##
    # nltk.data.path.append('./nltk_data/')
    ## Start ##
    with open(path) as books:
        df = pd.read_json(books) ## books is json string
    # print(df)
    check = df[['book_id','title']]
    df.set_index('book_id',inplace=True)
    # initializing the new column
    df['final_text'] = ""
    # get final text for recommendation
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
    # print(df)

    vectorizer = TfidfVectorizer(analyzer='word')
    #build final-text to tfidf matrix
    tfidf_matrix = vectorizer.fit_transform(df['final_text'])
    # tfidf_feature_name = vectorizer.get_feature_names()

    # comping cosine similarity matrix using linear_kernal of sklearn
    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

    df = df.reset_index(drop=True)
    indices = pd.Series(df['title'].index)
    #Function to get the most similar books
    def recommend(index, method):
        id = indices[index]
        # Get the pairwise similarity scores of all books compared that book,
        # sorting them and getting top 5
        similarity_scores = list(enumerate(method[id]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:6]

        #Get the books index
        books_index = [i[0] for i in similarity_scores]

        #Return the top 10 most similar books using integar-location based indexing (iloc)
        return df['title'].iloc[books_index]

    # print(check)
    #input the index of the book and get top 10 book recommendation
    row_no = str(check[check['book_id'] == num].index.values)
    row_no = int(row_no.replace("[","").replace("]","")) ## chosen id from web
    # print("Index values for recommendation",row_no)
    recom = recommend(row_no, cosine_similarity) ## need to edit, chosen id from web instead of 1

    book_id= []
    #for book title
    book_title = []
    for book in recom:
        book_title.append(book)
    check= check.set_index('title')
    for i in range(len(book_title)):
        book_id.append(int(check.loc[book_title[i],'book_id'])) #for book Id

    #for bookId and Title json
    output_json=[]
    for i in range(len(book_title)):
        output_json.append({"book_id": book_id[i],"book_title":book_title[i]})
    print(output_json)
    return make_response(jsonify(output_json),200)
    # return "hello"
if __name__ == '__main__':
    app.run(host = "127.0.0.1",port = "5000",debug=True)

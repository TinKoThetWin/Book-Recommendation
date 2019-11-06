# print(check)
#input the index of the book and get top 10 book recommendation
row_no = str(check[check['book_id'] == num].index.values)
row_no = int(row_no.replace("[","").replace("]",""))
# print("Index values for recommendation",row_no)
recom = recommend(row_no, cosine_similarity,indices,df,check) ## need to edit, chosen id from web instead of 1

book_id= []
book_title = []
for id in recom:
    book_id.append(id)
check= check.set_index('book_id')
for i in range(len(book_id)):
    book_title.append(check.loc[book_id[i],'title']) #for book Id

# for bookId and Title json
output_json=[]
for i in range(len(book_id)):
    output_json.append({"book_id": book_id[i],"book_title":book_title[i]})
return json.dumps(output_json)

def recommend(index, method,indices,df,check):

    id = indices[index]
    #print("id",id)
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(method[id]))
    #print("sim",df)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
      
        #Get the books index
    books_index = [i[0] for i in similarity_scores]
        
        #Return the top 5 most similar books using integar-location based indexing (iloc)
    ##print("result",df['title'].iloc[books_index])
    return check['book_id'].iloc[books_index]

if __name__ == '__main__':
   app.run(host = "127.0.0.1",port = "5000",debug=True)
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from flask_table import Table, Col
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#building flask table for showing recommendation results
class Results(Table):
    idx = Col('Id', show=False)
    title = Col('Recommendation List')

app = Flask(__name__)

#Input page Page
@app.route('/')
def home():
    return render_template("home.html")    

# result page
@app.route("/recommendation", methods = ["GET", "POST"])
def recommendation():
    if request.method == "POST":
        #reading the dataset 
        title = request.form["title"]

        metadata = pd.read_csv("./q_movies_nn.csv", low_memory=False)

# define vectorizer object, remove stop words like the, a
        tfidf = TfidfVectorizer(stop_words = "english")

# replace nan with empty string
        metadata["description"] = metadata["description"].fillna("")

        tfidf_matrix = tfidf.fit_transform(metadata["description"])

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        indices = pd.Series(metadata.index, index = metadata["title"]).drop_duplicates()

        def get_recommendations(title, cosine_sim = cosine_sim):
    
    #get the index of the movie that matches the title
            idx = indices[title]
    #pairwise similarity scores of all movies
            sim_scores = list(enumerate(cosine_sim[idx]))
    # sort the movies based on the similarity scores
    # add second element of list to the function
            sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    # get the scores of the 10 most similar movies
            sim_scores = sim_scores[1:11]
    # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]
    # return the top 10 most similar movies
            return metadata["title"].iloc[movie_indices]

        output= get_recommendations(title)
        table = Results(output)
        return render_template('recommendation.html', table = table)

if __name__ == "__main__":
    app.run()
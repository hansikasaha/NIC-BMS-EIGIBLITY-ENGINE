from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load the data
rec = pd.read_csv('recommendation_dataset.csv')

# Preprocess the data and create the TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(rec['description'])

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 5

def get_recommendations(user_id: int, num_recommendations: int = 5):
    if user_id not in rec['user_id'].values:
        raise HTTPException(status_code=404, detail="User not found")
    user_index = rec[rec['user_id'] == user_id].index[0]
    
    # Compute cosine similarities only for the given user
    user_tfidf_vector = tfidf_matrix[user_index]
    cosine_similarities = cosine_similarity(user_tfidf_vector, tfidf_matrix).flatten()
    
    # Get the scores for the most similar items
    similar_indices = cosine_similarities.argsort()[-(num_recommendations + 1):-1][::-1]
    
    return rec.iloc[similar_indices].to_dict('records')

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    recommendations = get_recommendations(request.user_id, request.num_recommendations)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse

rec = pd.read_csv('recommendation_dataset.csv')

# After preprocessing
with open('cleaned_rec.pkl', 'wb') as file:
    pickle.dump(rec, file)

try:
    with open('cleaned_rec.pkl', 'rb') as file:
        rec = pickle.load(file)
except FileNotFoundError:
    # Proceed with preprocessing if the file does not exist
    rec = pd.read_csv('recommendation_dataset.csv')
    # ... (preprocessing steps)

print(rec.head())

print(rec.info())


# Drop rows with missing values (NaN values)
rec = rec.dropna()


# Check for any remaining missing values after dropping
missing_values = rec.isnull().sum()

if missing_values.any():
    print("There are still missing values after dropping.")
    print(missing_values)
else:
    print ("No missing values found after dropping.")


column_name = 'scheme_name'  

# Get the value counts for the specified column
term_counts = rec[column_name].value_counts()
 
# Display the total occurrences of each term
print("Total occurrences of each term in column", column_name, "are:")
print(term_counts)


# Map age ranges to numerical codes
#This part converts age ranges specified as strings into numerical codes. For example, 'Below 10' is mapped to 0, '10-15' to 1, and so on.
age_mapping = {
    'Below 10': 0, '10-15': 1, '16-20': 2, '21-25': 3, '26-30': 4, 
    '31-35': 5, '36-40': 6, '41-45': 7, '46-50': 8, 'Above 50': 9
}
rec['age'] = rec['age'].map(age_mapping)

# Convert categorical variables to numerical codes, including SC, ST, and OBC for caste
caste_mapping = {'SC': 0, 'ST': 1, 'OBC': 2}
rec['social_category'] = rec['social_category'].map(caste_mapping)

# Convert categorical variables to numerical codes, including M, F, and T for gender
gender_mapping = {'M': 0, 'F': 1, 'T': 2}
rec['gender'] = rec['gender'].map(gender_mapping)

# Convert domicile to numerical (binary) variable
rec['domicile_of_tripura'] = rec['domicile_of_tripura'].map({'Y': 1, 'N': 0})

rec['scheme_text'] = rec['scheme_name'] + ' ' + rec['description']


# TDF-IDF VECTORIZATION for combined scheme text 

vectorizer=TfidfVectorizer()
tfidf_matrix=vectorizer.fit_transform(rec['scheme_text'])

#tfidf_matrix: This is a matrix where each row is a TF-IDF vector representing the text of a scheme (combining scheme name and description).



def content_based_filtering(search_terms, age, social_category, gender, domicile_of_tripura, num_recommendations=5):
    # Convert the search terms and parameters into a single string for TF-IDF
    search_query = ' '.join(search_terms) + ' ' + age + ' ' + social_category + ' ' + gender + ' ' + domicile_of_tripura
    
    # Vectorize the descriptions and the search query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(rec['description'])
    query_vector = vectorizer.transform([search_query])
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices of top similar schemes
    top_similar_indices = cosine_similarities.argsort()[::-1]
    
    # Remove duplicates while maintaining order
    seen = set()
    unique_indices = [index for index in top_similar_indices if not (rec.iloc[index]['scheme_name'] in seen or seen.add(rec.iloc[index]['scheme_name']))]
    
    # Select top unique recommendations
    unique_recommendations = unique_indices[:num_recommendations]
    
    # Print the suggested schemes
    print("Content-based Filtering:")
    print("The suggested schemes based on search terms and additional parameters are: \n")
    for index in unique_recommendations:
        print(rec.iloc[index]['scheme_name'])


# After vectorization
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
with open('tfidf_matrix.pkl', 'wb') as file:
    pickle.dump(tfidf_matrix, file)

try:
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('tfidf_matrix.pkl', 'rb') as file:
        tfidf_matrix = pickle.load(file)
except FileNotFoundError:
    # Proceed with vectorization if the files do not exist
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(rec['description'])


# Function to recommend schemes using both content-based and collaborative filtering
def recommend_schemes(search_terms, age, social_category, gender, domicile_of_tripura):
    # Content-based filtering
    content_based_filtering(search_terms, age, social_category, gender, domicile_of_tripura)

# # Example usage:
# search_terms = ['Mukhya']
# age = '21-25'
# social_category = 'ST'
# gender = 'M'
# domicile_of_tripura = 'Y'
# recommend_schemes(search_terms, age, social_category, gender, domicile_of_tripura)



# geting input from user
def get_user_input():
    search_terms = input("Enter search terms (separated by space): ").split()
    age = input("Enter age range: ")
    social_category = input("Enter social category (SC/ST/OBC): ")
    gender = input("Enter gender (M/F/T): ")
    domicile_of_tripura = input("Is domicile of Tripura? (Y/N): ")
    return search_terms, age, social_category, gender, domicile_of_tripura

def main():
#     # Get user input
    search_terms, age, social_category, gender, domicile_of_tripura = get_user_input()

#     # Call the recommend_schemes function with user input
    recommend_schemes(search_terms, age, social_category, gender, domicile_of_tripura)

if __name__ == "__main__":
    main()
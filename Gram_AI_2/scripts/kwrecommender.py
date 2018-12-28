import pandas as pd

metadata = pd.read_csv("..\\data\\absrecord.csv")
keywords = pd.read_csv("..\\data\\keywords.csv")

metadata = metadata.merge(keywords, on='filename')
print(metadata.head())

# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['keywords']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        # if len(names) > 3:
        #     names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    return ' '.join(x['keywords']) 

# Create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)

print(metadata.head())

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['body'])

def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    journal_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[journal_indices].values

print(get_recommendations('adiposity', cosine_sim2))
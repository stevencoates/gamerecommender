# Let the user know that we're waiting for data to process.
print("Loading... Please wait.", end="\r")

import pandas as pd
import numpy as np
import re
import unicodedata
import os
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to convert comma separate lists into actual lists.
def split_list(x):
	return x.split(",")

# Function to "clean" strings, stripping out spaces and making it lower case.
def clean_string(x):
	return ''.join(i for i in str.lower(str(x).replace(" ", "")) if ord(i) < 128)
	
# Function to remove unicode characters from a string.
def remove_unicode(x):
	return ''.join(i for i in str(x) if ord(i) < 128)

# Function to create metadata "soup", a string containing all the metadata we want to consider.
def create_soup(x):
	return ' '.join(x['genre_list']) + ' ' + ' '.join(x['game_details_list']) + ' ' + ' '.join(x['popular_tags_list']) + ' ' + ' '.join(x['developer_list']) + ' ' + ' '.join(x['publisher_list'])

# Check if we have a folder for caching processed data.
if not os.path.exists('cache'):
	os.makedirs('cache')

# Check if we have a cached set of processed data.
if os.path.isfile('cache/cosine_sim.npy') and os.path.isfile('cache/indices.pkl') and os.path.isfile('cache/dataset.pkl'):
	cosine_sim = np.load('cache/cosine_sim.npy')
	indices = pd.read_pickle('cache/indices.pkl')
	dataset = pd.read_pickle('cache/dataset.pkl')
else:
	# Load in our dataset.
	dataset = pd.read_csv('dataset_steam_games.csv', low_memory=False)

	# Drop any items from the dataset that are listed as a bundle or DLC.
	dataset = dataset[~dataset.types.str.contains("bundle", na=False)]
	dataset = dataset[~dataset.game_details.str.contains("Downloadable Content", na=False)]
	# Reset the index, now that we've removed a load of items.
	dataset = dataset.reset_index()
		
	# Create a field for clean, searchable names of games.
	dataset['searchable_name'] = dataset['name'].apply(clean_string)
	# And for the names, just strip out the unicode characters.
	dataset['name'] = dataset['name'].apply(remove_unicode)

	# Create lists of publishers, developers, genres, details and tags, from the comma separate listings that are in the dataset.
	features = ['publisher', 'developer', 'genre', 'popular_tags', 'game_details']
	for feature in features:
		dataset[feature] = dataset[feature].apply(clean_string)
		dataset[feature+'_list'] = dataset[feature].apply(split_list)

	# Create our soup from the newly available metadata.
	dataset['soup'] = dataset.apply(create_soup, axis=1)
	# Save the final dataset out to a file, for future use.
	dataset.to_pickle('cache/dataset.pkl')

	# Create a count matrix to use, based on our soup, and then a cosine similarity based on that.
	count = CountVectorizer(stop_words='english')
	count_matrix = count.fit_transform(dataset['soup'])
	cosine_sim = cosine_similarity(count_matrix, count_matrix)
	# Save the cosine similarity out to a file, for future use.
	np.save('cache/cosine_sim.npy', cosine_sim)

	# Construct a reverse mapping of the indices from titles.
	indices = pd.Series(dataset.index, index=dataset['searchable_name'])
	# Save the indices out to a file, for future use.
	indices.to_pickle('cache/indices.pkl')
	

# Function to get a recommendation, given a game's name.
def get_recommendations(name, cosine_sim=cosine_sim):
	# Convert the name into a searchable version of itself.
	name = clean_string(name)
	
	# Check whether or not the game given is available.
	if name in indices:
		# Get the index of the game that matches the name.
		index = indices[name]
		
		# Get the pairwise similarity of scores of all games with that game.
		sim_scores = list(enumerate(cosine_sim[index]))
		
		# Sort the games based on the similarity scores.
		sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
		
		# Get the scores of the 10 most similar games.
		sim_scores = sim_scores[1:11]
		
		# Get the game indices.
		game_indices = [i[0] for i in sim_scores]
		
		# Print the top 10 most similar games.
		print(dataset['name'].iloc[game_indices])
	else:
		print("No game could be found in our records, by that name.")


search = input("Enter the name of a game that you like (or enter 'q' to quit): ")
while True:
	if search == "q":
		exit()
	get_recommendations(search)
	search = input("Enter the name of another game that you like (or enter 'q' to quit): ")
	
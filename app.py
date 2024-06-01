import math
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import h5py
import requests
from joblib import load

app = Flask(__name__)

# # Open the file 'filename.h5' in read-only mode
# with h5py.File('model/workingmyanime.weights.h5', 'r') as f:
#     # Print the keys of the file
#     print(f['layers/anime_embedding/anime_embedding/embeddings:0'])
#     for key in f.keys():
#         print(key)
#         print(f[key])

def extract_weights(file_path, layer_name):
    with h5py.File(file_path, 'r') as h5_file:
        print("Layers in the model:", list(h5_file.keys()))
        if layer_name in h5_file:
            weight_layer = h5_file[layer_name]
            if isinstance(weight_layer, h5py.Dataset):  # Check if it's a dataset (contains weights)
                weights = weight_layer[()]
                # Normalize the weights if needed
                weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
                return [weights]

    raise KeyError(f"Unable to find weights for layer '{layer_name}' in the HDF5 file.")

# Provide the correct file path to the model
scriptdir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(scriptdir, '/anime-recommand-scheduling/data/workingmyanime.h5')
anime_enc_path = os.path.join(scriptdir, '/anime-recommand-scheduling/data/anime_encoder.pkl')
user_enc_path = os.path.join(scriptdir, '/anime-recommand-scheduling/data/user_encoder.pkl')
anime_path = os.path.join(scriptdir, '/anime-recommand-scheduling/data/all_animes.csv')
scores_path = os.path.join(scriptdir, '/anime-recommand-scheduling/data/users-score-2023.csv')

# Extract weights for anime embeddings
anime_weights = extract_weights(file_path, 'layers/embedding_1/vars/0')

# Extract weights for user embeddings
user_weights = extract_weights(file_path, 'layers/embedding/vars/0')

anime_weights=anime_weights[0]
user_weights=user_weights[0]

print(user_weights[0])
print(anime_weights[0])

with open(anime_enc_path, 'rb') as file:
    anime_encoder = load(file)

with open(user_enc_path, 'rb') as file:
    user_encoder = load(file)

# # Load the dataset for additional information
# with open('model/anime-dataset-2023.pkl', 'rb') as file:
#     df_anime = load(file)
df_anime = pd.read_csv(anime_path, low_memory=True)
df_anime['Rank'] = df_anime['rank']
# Split the genres and count their occurrences
import ast

def convertToName(obj):
    L = []
    if (obj == "[]"):
        return "UNKNOWN"
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    str = ", ".join(L)
    return str

df_anime['genres'] = df_anime['genres'].apply(convertToName)
df_anime['studios'] = df_anime['studios'].apply(convertToName)
df_anime['producers'] = df_anime['producers'].apply(convertToName)
df_anime['licensors'] = df_anime['licensors'].apply(convertToName)

df_anime['premiered'] = df_anime['season'].str.cat(df_anime['year'].astype(str), sep=' ')

# Define a function to extract the desired date range string
def get_date_range(aired_dict):
  aired_dict = ast.literal_eval(aired_dict)
  start_date = f"{aired_dict['prop']['from']['day']}/{aired_dict['prop']['from']['month']}/{aired_dict['prop']['from']['year']}"
  end_date = f"{aired_dict['prop']['to']['day']}/{aired_dict['prop']['to']['month']}/{aired_dict['prop']['to']['year']}"
  return f"{start_date} - {end_date}"

# Apply the function to 'aired' column using vectorized approach (efficient)
df_anime['aired'] = df_anime['aired'].apply(get_date_range)


df_anime = df_anime.replace("UNKNOWN", "")

# print(df_anime)
# print(ast.literal_eval(df_anime['images'][df_anime['title'] == 'One Piece'].values[0]).get('jpg', {}).get('image_url'))
    
# Load the user ratings dataset
df=pd.read_csv(scores_path, low_memory=True)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# User-based collaborative filtering
## Finding similar users
def find_similar_users(item_input, n=10, return_dist=False, neg=False):
    try:
        index = item_input
        encoded_index = user_encoder.transform([index])[0]
        weights = user_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
            
        SimilarityArr = []
        
        for close in closest:
            similarity = dists[close]
            if isinstance(item_input, int):
                decoded_id = user_encoder.inverse_transform([close])[0]
                SimilarityArr.append({"similar_users": decoded_id, "similarity": similarity})
        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
        return Frame
    except:
        print('\033[1m{}\033[0m, Not Found in User list'.format(item_input))

## Function to get user preferences
def get_user_preferences(user_id):
    animes_watched_by_user = df[df['user_id'] == user_id]
    
    if animes_watched_by_user.empty:
        print("User #{} has not watched any animes.".format(user_id))
        return pd.DataFrame()
    
    user_rating_percentile = np.percentile(animes_watched_by_user.rating, 75)
    animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >= user_rating_percentile]
    top_animes_user = (
        animes_watched_by_user.sort_values(by="rating", ascending=False).anime_id.values
    )
    
    anime_df_rows = df_anime[df_anime["mal_id"].isin(top_animes_user)]
    anime_df_rows = anime_df_rows[["title", "genres"]]
    return anime_df_rows

## Finally recommending animes for specific users
def get_recommended_animes(similar_users, user_pref, n=10):
    recommended_animes = []
    anime_list = []
    for user_id in similar_users.similar_users.values:
        pref_list = get_user_preferences(int(user_id))
        if not pref_list.empty:  # Check if user has watched any animes
            pref_list = pref_list[~pref_list["title"].isin(user_pref["title"].values)]
            anime_list.append(pref_list.title.values)
    if len(anime_list) == 0:
        print("No anime recommendations available for the given users.")
        return pd.DataFrame()
    anime_list = pd.DataFrame(anime_list)
    sorted_list = pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)
    # Count the occurrences of each anime in the entire dataset
    anime_count = df['anime_id'].value_counts()
    for i, anime_name in enumerate(sorted_list.index):
        if isinstance(anime_name, str):
            anime_image_url = ast.literal_eval(df_anime[df_anime['title'] == anime_name]['images'].values[0]).get('jpg', {}).get('image_url')
            anime_id = df_anime[df_anime.title == anime_name].mal_id.values[0]
            genre = df_anime[df_anime.title == anime_name].genres.values[0]                
            Synopsis = df_anime[df_anime.title == anime_name].synopsis.values[0]
            n_user_pref = anime_count.get(anime_id, 0)  # Get the total count of users who have watched this anime
            english_name = df_anime[df_anime.title == anime_name]['title_english'].values[0]
            other_name = df_anime[df_anime.title == anime_name]['title_japanese'].values[0]
            score = df_anime[df_anime.title == anime_name].score.values[0]
            Type = df_anime[df_anime.title == anime_name].type.values[0]
            status = df_anime[df_anime.title == anime_name].status.values[0]
            aired = df_anime[df_anime.title == anime_name].aired.values[0]
            episodes = df_anime[df_anime.title == anime_name].episodes.values[0]
            premiered = df_anime[df_anime.title == anime_name].premiered.values[0]
            studios = df_anime[df_anime.title == anime_name].studios.values[0]
            source = df_anime[df_anime.title == anime_name].source.values[0]
            rating = df_anime[df_anime.title == anime_name].rating.values[0]
            rank = df_anime[df_anime.title == anime_name].Rank.values[0]
            favorites = df_anime[df_anime.title == anime_name].favorites.values[0]
            duration = df_anime[df_anime.title == anime_name].duration.values[0]
            
            # Handling status column values
            if status == "Not yet aired" and aired == "Not available":
                aired = "TBA"
            else:
                aired = "" if aired == "Not available" else aired.replace(" to ", "-")

            # Handling episodes column values
            if episodes != "":
                episodes = int(float(episodes))
                if status == "Currently Airing":
                    episodes = str(episodes)+"+ EPS"
                else:
                    episodes = str(episodes)+" EPS"
            else:
                if status == "Currently Airing":
                    aired_year = df_anime[df_anime.Name == anime_name].Aired.values[0]
                    if ',' in aired_year:
                        aired_year = aired_year.split(',')[1].strip()
                        aired_year = aired_year.split(' to ')[0].strip()
                    else:
                        aired_year = aired_year.split(' to ')[0].strip()
                    if aired_year != "Not available" and int(aired_year) <= 2020:
                        episodes = "∞"
                    else:
                        episodes = ""
                else:
                    episodes = ""

            # Handling Rating column values
            rating = rating if rating == "" else rating.split(' - ')[0]
            
            # Handling Rank column values
            rank = rank if rank == "" else "#"+str(int(float(rank)))
            
            # Making new column episode_duration
            episode_duration = ""
            if episodes != "":
                time = ""
                if 'hr' in duration:
                    hours, minutes = 0, 0
                    time_parts = duration.split()
                    for i in range(len(time_parts)):
                        if time_parts[i] == "hr":
                            hours = int(time_parts[i-1])
                        elif time_parts[i] == "min":
                            minutes = int(time_parts[i-1])
                    time = str(hours * 60 + minutes) + " min"
                else:
                    time= duration.replace(" per ep","")
                episode_duration = "("+ episodes + "  x " + time +")"
            else:
                episode_duration = "("+ duration +")"


            recommended_animes.append({"anime_image_url": anime_image_url, "n": n_user_pref,"anime_name": anime_name, "Genres": genre,
                                        "Synopsis": Synopsis,"English Name": english_name,"Native name": other_name,"Score": score,
                                        "Type": Type, "Aired": aired, "Premiered": premiered, "Episodes": episodes, "Status": status,
                                        "Studios": studios,"Source": source, "Rating": rating, "Rank": rank, "Favorites": favorites,
                                        "Duration": duration, "Episode Duration": episode_duration,"anime_id":anime_id})
    return pd.DataFrame(recommended_animes)

# Item-based collaborative filtering
def find_similar_animes(name, n=10, return_dist=False, neg=False):
        anime_row = df_anime[df_anime['title'] == name].iloc[0]
        index = anime_row['mal_id']
        encoded_index = anime_encoder.transform([index])[0]
        weights = anime_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1            
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
        print('Animes closest to {}'.format(name))
        if return_dist:
            return dists, closest
        SimilarityArr = []
        for close in closest:
            decoded_id = anime_encoder.inverse_transform([close])[0]
            anime_frame = df_anime[df_anime['mal_id'] == decoded_id]
            anime_id=anime_frame['mal_id'].values[0]
            anime_image_url = ast.literal_eval(anime_frame['images'].values[0]).get('jpg', {}).get('image_url')
            anime_name = anime_frame['title'].values[0]
            genre = anime_frame['genres'].values[0]
            Synopsis = anime_frame['synopsis'].values[0]
            similarity = dists[close]
            similarity = "{:.2f}%".format(similarity * 100)
            
            english_name = anime_frame['title_english'].values[0]
            other_name = anime_frame['title_japanese'].values[0]
            score = anime_frame['score'].values[0]
            Type = anime_frame['type'].values[0]
            status = anime_frame['status'].values[0]
            aired = anime_frame['aired'].values[0]
            episodes = anime_frame['episodes'].values[0]
            premiered = anime_frame['premiered'].values[0]
            studios = anime_frame['studios'].values[0]
            source = anime_frame['source'].values[0]
            rating = anime_frame['rating'].values[0]
            rank = anime_frame['rank'].values[0]
            favorites = anime_frame['favorites'].values[0]
            duration = anime_frame['duration'].values[0]
            
            # Handling status column values
            if status == "Not yet aired" and aired == "Not available":
                aired = "TBA"
            else:
                aired = "" if aired == "Not available" else aired.replace(" to ", "-")
                
            # Handling episodes column values
            if episodes != "":
                if not math.isnan(float(episodes)):
                    episodes = int(float(episodes))
                    if status == "Currently Airing":
                        episodes = str(episodes)+"+ EPS"
                    else:
                        episodes = str(episodes)+" EPS"
            else:
                if status == "Currently Airing":
                    aired_year = anime_frame['Aired'].values[0]
                    if ',' in aired_year:
                        aired_year = aired_year.split(',')[1].strip()
                        aired_year = aired_year.split(' to ')[0].strip()
                    else:
                        aired_year = aired_year.split(' to ')[0].strip()
                    if aired_year != "Not available" and int(aired_year) <= 2020:
                        episodes = "∞"
                    else:
                        episodes = ""
                else:
                    episodes = ""
                
            # Handling Rating column values
            rating = rating if rating == "" else rating.split(' - ')[0]
            
            # Handling Rank column values
            rank = rank if rank == "" else "#"+str(int(float(rank)))
            
            # Making new column episode_duration
            episode_duration = ""
            if episodes != "":
                time = ""
                if 'hr' in duration:
                    hours, minutes = 0, 0
                    time_parts = duration.split()
                    for i in range(len(time_parts)):
                        if time_parts[i] == "hr":
                            hours = int(time_parts[i-1])
                        elif time_parts[i] == "min":
                            minutes = int(time_parts[i-1])
                    time = str(hours * 60 + minutes) + " min"
                else:
                    time= duration.replace(" per ep","")
                episode_duration = "("+ str(episodes) + "  x " + str(time) +")"
            else:
                episode_duration = "("+ str(duration) +")"
            
            
            SimilarityArr.append({"anime_image_url": anime_image_url,"Name": anime_name, "Similarity": similarity, "Genres": genre,
                                  "Synopsis":Synopsis,"English Name": english_name,"Native name": other_name,"Score": score,"Type": Type,
                                  "Aired": aired, "Premiered": premiered, "Episodes": episodes, "Status": status, "Studios": studios,
                                  "Source": source, "Rating": rating, "Rank": rank, "Favorites": favorites,"Duration": duration,
                                  "Episode Duration": episode_duration,"anime_id":anime_id})
        Frame = pd.DataFrame(SimilarityArr).sort_values(by="Similarity", ascending=False)
        return Frame[Frame.Name != name]

# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    recommendation_type = request.form['recommendation_type']
    num_recommendations = int(request.form['num_recommendations'])

    if recommendation_type == "user_based":
        user_id = request.form['user_id']
        
        if not user_id:
            return render_template('index.html', error_message="Please enter a User ID.", recommendation_type=recommendation_type)
        try:
            user_id = int(user_id)
        except ValueError:
            return render_template('index.html', error_message="Please enter a valid User ID (must be an integer).", recommendation_type=recommendation_type)
        
        # Find similar users based on preferences
        similar_user_ids = find_similar_users(user_id, n=15, neg=False)
        if similar_user_ids is None or similar_user_ids.empty:
            url = f'https://api.jikan.moe/v4/users/userbyid/{user_id}'  # Check if the user_id exists using Jikan API
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                username = data.get('username')
                if 'data' not in data:
                    message1 = "Available"
                else:
                    message1 = "No anime recommendations available for the given user.(REASON :- User may not have rated any anime)"
            else:
                message1 = "User with user_id " + str(user_id) + " does not exist in the database."
            return render_template('recommendations.html', message=message1, animes=None, recommendation_type=recommendation_type)

        similar_user_ids = similar_user_ids[similar_user_ids.similarity > 0.4]
        similar_user_ids = similar_user_ids[similar_user_ids.similar_users != user_id]

        # Get user preferences from the dataset
        user_pref = get_user_preferences(user_id)

        # Get recommended animes for the user
        recommended_animes = get_recommended_animes(similar_user_ids, user_pref, n=num_recommendations)
        username = df[df['user_id'] == user_id]['Username'].values[0]
        return render_template('recommendations.html', animes=recommended_animes, recommendation_type=recommendation_type, username=username)

    elif recommendation_type == "item_based":
        anime_name = request.form['anime_name']
        
        if not anime_name:
            return render_template('index.html', error_message="Please enter Anime name.", recommendation_type=recommendation_type)
        
        recommended_animes = find_similar_animes(anime_name, n=num_recommendations, return_dist=False, neg=False)
        if recommended_animes is None or recommended_animes.empty:
            message2 = "Anime " + str(anime_name) + " does not exist"
            return render_template('recommendations.html', message=message2, animes=None, recommendation_type=recommendation_type)

        anime_row = df_anime[df_anime['title'] == anime_name].iloc[0]
        index = anime_row['mal_id']
        anime_frame = df_anime[df_anime['mal_id'] == index]
        anime_id=anime_frame['mal_id'].values[0]
        anime_image_url = ast.literal_eval(anime_frame['images'].values[0]).get('jpg', {}).get('image_url')
        anime_name = anime_frame['title'].values[0]
        genre = anime_frame['genres'].values[0]
        Synopsis = anime_frame['synopsis'].values[0]
        english_name = anime_frame['title_english'].values[0]
        other_name = anime_frame['title_japanese'].values[0]
        score = anime_frame['score'].values[0]
        Type = anime_frame['type'].values[0]
        status = anime_frame['status'].values[0]
        aired = anime_frame['aired'].values[0]
        episodes = anime_frame['episodes'].values[0]
        premiered = anime_frame['premiered'].values[0]
        studios = anime_frame['studios'].values[0]
        source = anime_frame['source'].values[0]
        rating = anime_frame['rating'].values[0]
        rank = anime_frame['rank'].values[0]
        favorites = anime_frame['favorites'].values[0]
        duration = anime_frame['duration'].values[0]
        
        # Handling status column values
        if status == "Not yet aired" and aired == "Not available":
            aired = "TBA"
        else:
            aired = "" if aired == "Not available" else aired.replace(" to ", "-")
            
        # Handling episodes column values
        if episodes != "":
            if not math.isnan(float(episodes)):
                episodes = int(float(episodes))
                if status == "Currently Airing":
                    episodes = str(episodes)+"+ EPS"
                else:
                    episodes = str(episodes)+" EPS"
        else:
            if status == "Currently Airing":
                aired_year = anime_frame['Aired'].values[0]
                if ',' in aired_year:
                    aired_year = aired_year.split(',')[1].strip()
                    aired_year = aired_year.split(' to ')[0].strip()
                else:
                    aired_year = aired_year.split(' to ')[0].strip()
                if aired_year != "Not available" and int(aired_year) <= 2020:
                    episodes = "∞"
                else:
                    episodes = ""
            else:
                episodes = ""
            
        # Handling Rating column values
        rating = rating if rating == "" else rating.split(' - ')[0]
        
        # Handling Rank column values
        rank = rank if rank == "" else "#"+str(int(float(rank)))
        
        # Making new column episode_duration
        episode_duration = ""
        if episodes != "":
            time = ""
            if 'hr' in duration:
                hours, minutes = 0, 0
                time_parts = duration.split()
                for i in range(len(time_parts)):
                    if time_parts[i] == "hr":
                        hours = int(time_parts[i-1])
                    elif time_parts[i] == "min":
                        minutes = int(time_parts[i-1])
                time = str(hours * 60 + minutes) + " min"
            else:
                time= duration.replace(" per ep","")
            episode_duration = "("+ str(episodes) + "  x " + str(time) +")"
        else:
            episode_duration = "("+ str(duration) +")"
        anime_obj = {"anime_image_url": anime_image_url,"Name": anime_name, "Genres": genre,
                                  "Synopsis":Synopsis,"English Name": english_name,"Native name": other_name,"Score": score,"Type": Type,
                                  "Aired": aired, "Premiered": premiered, "Episodes": episodes, "Status": status, "Studios": studios,
                                  "Source": source, "Rating": rating, "Rank": rank, "Favorites": favorites,"Duration": duration,
                                  "Episode Duration": episode_duration,"anime_id":anime_id}
        
        return render_template('recommendations.html', animes=recommended_animes, recommendation_type=recommendation_type, anime_name = anime_name, anime_obj = anime_obj)

    else:
        return render_template('index.html', error_message="Please select a recommendation type.")

# New route to handle anime name autocomplete
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search_term = request.args.get('term')
    if search_term:
        filtered_animes = df_anime[df_anime['title'].str.contains(search_term, case=False)]
        anime_names = filtered_animes['title'].tolist()
    # else:
    #     anime_names = df_anime['Name'].tolist()
    return jsonify(anime_names)

if __name__ == '__main__':
    app.run(debug=True, port=80)

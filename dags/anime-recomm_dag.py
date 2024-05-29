import csv
from datetime import datetime, timedelta
import random
from bs4 import BeautifulSoup

from datetime import date
import json
from jikanpy import Jikan
import json
import time
import pandas
import requests
from airflow import DAG
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings(action='ignore')

# Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Model Training
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

## Import necessary modules for collaborative filtering
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from airflow.operators.python import PythonOperator

from wordcloud import WordCloud
from collections import defaultdict
from collections import Counter

## Import necessary modules for content-based filtering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import ssl

with open('/opt/airflow/data/animelast.txt', 'r', encoding='utf-8') as f:
    anime_start_index = int(f.read())
    last_page = anime_start_index

with open('/opt/airflow/data/userlast.txt', 'r', encoding='utf-8') as f:
    user_start_index = int(f.read()) + 1

ssl._create_default_https_context = ssl._create_unverified_context
def crawl_anime(**kwargs):
    index = kwargs["anime_start"]
    jikan = Jikan()
    result = jikan.search('anime', '')
    finish = True
    last_page = index
    all_animes = []
    
    try:
        for _ in range(index, int(result['pagination']['last_visible_page'])+1):
            # for _ in range(int(1)):
            last_page = last_page + 1
            tmp_page = json.loads(
                requests.get(
                    f'https://api.jikan.moe/v4/anime?page={_}').text)['data']
            time.sleep(0.6)
            for i in range(len(tmp_page)):
                anime = tmp_page[i]
                mal_id = anime['mal_id']
                # if str(mal_id) == '15':
                #     raise KeyError('TEST')
                print(mal_id)
                # recommendationsAPI = requests.get(
                #     f'https://api.jikan.moe/v4/anime/{mal_id}/recommendations')
                # time.sleep(0.6)
                # charAPI = requests.get(
                #     f'https://api.jikan.moe/v4/anime/{mal_id}/characters')
                # time.sleep(0.6)
                # staffAPI = requests.get(
                #     f'https://api.jikan.moe/v4/anime/{mal_id}/staff')
                # time.sleep(0.6)
                # if recommendationsAPI.status_code != 404:
                #     recommendations: list = json.loads(
                #         recommendationsAPI.text)['data'][:5]
                #     recommendationResult = []
                #     for j in range(len(recommendations)):
                #         url: str = recommendations[j]['entry']['url'][8:]
                #         # print(url)
                #         recommendationResult.append(url.split('/')[3])
                #         # From url = 'https://myanimelist.net/anime/205/Samurai_Champloo', get 'Samurai_Champloo'
                #     anime['crawl_recommendations'] = recommendationResult
                # else:
                #     anime['crawl_recommendations'] = []
                # if charAPI.status_code != 404:
                #     characters: list = json.loads(charAPI.text)['data'][:10]
                #     anime['crawl_characters'] = characters
                # else:
                #     anime['crawl_characters'] = []
                # if staffAPI.status_code != 404:
                #     staff: list = json.loads(staffAPI.text)['data'][:5]
                #     anime['crawl_staff'] = staff
                # else:
                #     anime['crawl_staff'] = []
                    
                all_animes.append(anime)
            # all_animes.extend(tmp_page)
            print(f'LENGTH: {len(all_animes)}')
            # print(tmp_page)
    except Exception:
        print("ERROR!")
        print(len(all_animes))
        finish = False
        
        with open("/opt/airflow/data/all_animes.csv", "r", encoding='utf-8') as f:
            existing_data = list(csv.DictReader(f))

        with open("/opt/airflow/data/all_animes.csv", "w", encoding='utf-8', newline='') as f:
            print("First data: ", existing_data[1])
            existing_data.extend(all_animes)
            fieldnames = list(existing_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
        with open('/opt/airflow/data/animelast.txt', 'w', encoding='utf-8') as f:
            f.write(str(last_page))

    if finish is True:
        with open("/opt/airflow/data/all_animes.csv", "r", encoding='utf-8') as f:
            existing_data = list(csv.DictReader(f))

        with open("/opt/airflow/data/all_animes.csv", "w", encoding='utf-8', newline='') as f:
            print("First data: ", existing_data[1])
            existing_data.extend(all_animes)
            fieldnames = list(existing_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
        with open('/opt/airflow/data/animelast.txt', 'w', encoding='utf-8') as f:
            f.write(str(last_page))
    # pandas.read_json('all_animes.json').to_csv('all_animes.csv')

def crawl_user_list(**kwargs): 
    output_file = '/opt/airflow/data/userlist/userlist.csv'
    index = kwargs["user_start"]

    start_id = index  # Starting user ID
    end_id = index + 10000  # Ending user ID

    count = 0  # keep count of users for the current session
    with open(output_file, 'r', newline='', encoding='utf-8') as csvfile:
        users = list(csv.DictReader(csvfile))

    for user_id in range(start_id, end_id + 1):
        print(user_id)
        apiUrl = f'https://api.jikan.moe/v4/users/userbyid/{user_id}'
        response = requests.get(apiUrl)

        # Check if the request was successful after retries
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Check if the 'data' field exists in the response
            if 'data' in data:
                # Extract the user details
                user_url = data['data']['url']
                username = data['data']['username']

                # Create a dictionary for the user
                user = {
                    'user_id': user_id,
                    'username': username,
                    'user_url': user_url
                }

                # Append the user dictionary to the list
                users.append(user)
                print(f'{count}. User data fetched for username: {user_id}')
                count += 1
            else:
                print(f'No user data found for ID: {user_id}')
    # else:
    #     print(f'Error occurred while fetching user data for ID: {user_id}')

    # Save the user details in a CSV file
    if users:
        fieldnames = list(users[0].keys())  # Get the fieldnames from the first user
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(users)
        with open('/opt/airflow/data/userlast.txt', 'w', encoding='utf-8') as f:
            f.write(str(end_id))

def crawl_user_profile():
    # Read usernames from CSV file
    usernames = []
    with open("/opt/airflow/data/userlist/userlist.csv", "r", encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            usernames.append(row["username"])

    # Prepare headers for the user_details.csv file
    headers = [
        "Mal ID", "Username", "Gender", "Birthday", "Location", "Joined",
        "Days Watched", "Mean Score", "Watching",
        "Completed", "On Hold", "Dropped",
        "Plan to Watch", "Total Entries", "Rewatched",
        "Episodes Watched"
    ]

    # Create a list to store user details
    # with open('/opt/airflow/data/users-details-2023.csv', 'r', encoding='utf-8') as file:
    #     user_details = list(csv.DictReader(file))
    user_details = []
    def count_rows(filename):
        with open(filename, 'r') as file:
            return sum(1 for line in file)


    # Initialize counter and timer variables
    total_usernames = len(usernames)
    total_details = count_rows('/opt/airflow/data/users-details-2023.csv') - 1
    fetch_count = 0
    total = 0
    start_time = time.time()

    # Set batch size and delay between batches
    batch_size = 3
    batch_delay = 1.0  # Delay in seconds between batches

    # Set maximum runtime to 11 hours (in seconds)
    max_runtime = 2 * 60

    # Iterate over usernames in batches
    for i in range(total_details + 1, total_usernames, batch_size):
        batch_usernames = usernames[i:i+batch_size]

        batch_user_details = []

        # Fetch user details for each username in the batch
        for username in batch_usernames:
            url = f"https://api.jikan.moe/v4/users/{username}/full"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                user_data = data.get("data", {})

                mal_id = user_data.get("mal_id")
                username = user_data.get("username")
                gender = user_data.get("gender")
                birthday = user_data.get("birthday")
                location = user_data.get("location")
                joined = user_data.get("joined")

                anime_statistics = user_data.get("statistics", {}).get("anime", {})
                days_watched = anime_statistics.get("days_watched")
                mean_score_anime = anime_statistics.get("mean_score")
                watching = anime_statistics.get("watching")
                completed_anime = anime_statistics.get("completed")
                on_hold = anime_statistics.get("on_hold")
                dropped = anime_statistics.get("dropped")
                plan_to_watch = anime_statistics.get("plan_to_watch")
                total_entries_anime = anime_statistics.get("total_entries")
                rewatched = anime_statistics.get("rewatched")
                episodes_watched = anime_statistics.get("episodes_watched")

                batch_user_details.append([
                    mal_id, username, gender, birthday, location, joined,
                    days_watched, mean_score_anime, watching,
                    completed_anime, on_hold, dropped,
                    plan_to_watch, total_entries_anime, rewatched,
                    episodes_watched
                ])

                fetch_count += 1
                total += 1
                time.sleep(0.35)

            else:
                print(f"Error occurred while fetching 'full' data for username: {username}")
                print(f"HTTP Error {response.status_code}: {response.reason}")
                print(f"Error message: {response.text}")

        # Add batch user details to the main user details list
        user_details.extend(batch_user_details)

        # Calculate and display progress
        if fetch_count >= 1000:
            progress = (i + len(batch_usernames)) / total_usernames * 100
            print(f"Progress: {progress:.2f}%")
            fetch_count = 0

        # Wait for the batch delay
        time.sleep(batch_delay)

        # Check elapsed time and exit loop if exceeding maximum runtime
        elapsed_time = time.time() - start_time
        if elapsed_time > max_runtime:
            print("Maximum runtime exceeded. Stopping the process.")
            break

    # Calculate elapsed time and usernames fetched per second
    elapsed_time = time.time() - start_time
    usernames_per_second = total / elapsed_time

    # Save user details to a csv file
    with open("/opt/airflow/data/users-details-2023.csv", "a", encoding='utf-8', newline="") as file:
        writer = csv.writer(file)
        # writer.writerow(headers)
        writer.writerows(user_details)

    print("User details saved to user_details.csv.")
    print(f"Fetched {total} usernames in {elapsed_time:.2f} seconds.")
    print(f"Average usernames fetched per second: {usernames_per_second:.2f}")

def scrape_user_profile(username, user_id, status_code):
    url = f"https://myanimelist.net/animelist/{username}?status={status_code}"
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept':
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Connection': 'keep-alive'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        table_1 = soup.find("table", {"data-items": True})
        table_2 = soup.find_all("table", {"border": "0", "cellpadding": "0", "cellspacing": "0", "width": "100%"})

        if table_1:
            data_items = table_1["data-items"]
            try:
                data_items_parsed = json.loads(data_items)
            except json.JSONDecodeError:
                return None

            data = []
            for data_item in data_items_parsed:
                anime_id = data_item["anime_id"]
                title = data_item["anime_title"]
                score = data_item["score"]
                if score != 0:
                    data.append([user_id, username, anime_id, title, score])

            return data

        elif table_2:
            data = []

            for table in table_2:
                row = table.find("tr")

                if row:
                    cells = row.find_all("td")

                    if len(cells) >= 5:
                        anime_title_cell = cells[1]
                        score_cell = cells[2]

                        anime_title_link = anime_title_cell.find("a", class_="animetitle")
                        anime_id = anime_title_link["href"].split("/")[2] if anime_title_link else ""
                        anime_title = anime_title_link.find("span").text.strip() if anime_title_link else ""

                        score_label = score_cell.find("span", class_="score-label")
                        score = score_label.text.strip() if score_label else "-"

                        if anime_title and score != "-":
                            data.append([user_id, username, anime_id, anime_title, score])

            return data

    return None

def crawl_user_score():
    with open('/opt/airflow/data/userlist/userlist.csv', 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        usernames = [row['username'] for row in rows]
        user_ids = [row['user_id'] for row in rows]

    # with open('/opt/airflow/data/users-score-2023.csv', 'r', encoding='utf-8') as file:
    #     user_scores = list(csv.reader(file))

    # with open('/opt/airflow/data/users-details-2023.csv', 'r', encoding='utf-8') as file:
    #     user_details = list(csv.reader(file))
    def count_rows(filename):
        with open(filename, 'r') as file:
            return sum(1 for line in file)

    num_rows = count_rows('/opt/airflow/data/users-details-2023.csv') - 1
    status_code = 7
    batch_size = 50  # Number of usernames to fetch in each batch
    min_delay_seconds = 90  # Minimum delay duration between requests in seconds
    max_delay_seconds = 120  # Maximum delay duration between requests in seconds

    with open('/opt/airflow/data/users-score-2023.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # writer.writerow(["User ID", "Username", "Anime ID", "Anime Title", "Score"])  # Write column names
        user_scores = []
        for i in range(num_rows, len(usernames), batch_size):
            usernames_batch = usernames[i:i + batch_size]
            user_ids_batch = user_ids[i:i + batch_size]

            for username, user_id in zip(usernames_batch, user_ids_batch):
                data = scrape_user_profile(username, user_id, status_code)
                if data:
                    user_scores.extend(data)
                    print(f"User details fetched successfully for username: {username}")
                    writer.writerows(data)
                else:
                    print(f"No user details found for username: {username}")

            if i + batch_size < len(usernames):
                # Add random delay between requests
                delay_seconds = random.randint(min_delay_seconds, max_delay_seconds)
                time.sleep(delay_seconds)
                print(f"Waiting for {delay_seconds} seconds before the next batch...")
                time.sleep(delay_seconds)

    if user_scores:
        print("All user details fetched successfully.")
    else:
        print("No user details found.")

def email():
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)
    out_csv_file_path = '/opt/airflow/data/users-score-2023.csv'
    import base64
    message = Mail(
        from_email='anime.recomm@gmail.com',
        to_emails='thaiduiqn@gmail.com',
        subject='Your file is here!',
        html_content='<img src="https://miai.vn/wp-content/uploads/2022/01/Logo_web.png"> Dear Admin,<br> Your file is in attachment<br>Thank you!'
    )

    with open(out_csv_file_path, 'rb') as f:
        data = f.read()
        f.close()
    encoded_file = base64.b64encode(data).decode()

    attachedFile = Attachment(
        FileContent(encoded_file),
        FileName('users-score-2023.csv'),
        FileType('text/csv'),
        Disposition('attachment')
    )
    message.attachment = attachedFile


    try:
        sg = SendGridAPIClient("Send Grid Token here")
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
        print(datetime.now())
    except Exception as e:
        print(e.message)

    return True

dag = DAG(
    'anime_dag',
    default_args={
        'retries': 3,
        'retry_delay': timedelta(seconds=30),
        'email_on_failure': True,
        'email': ['thaiduiqn@gmail.com'],
    },
    description='An anime recommendation system with update data',
    schedule_interval=timedelta(days=1),
    start_date= datetime.today() - timedelta(days=1),
    tags=['thai130102']
)


crawl_anime_operator = PythonOperator(
    task_id='crawl_anime',
    python_callable=crawl_anime,
    op_kwargs={"anime_start": anime_start_index},
    dag=dag
)

crawl_user_list_operator = PythonOperator(
    task_id='crawl_user_list',
    python_callable=crawl_user_list,
    op_kwargs={"user_start": user_start_index},
    dag=dag
)

crawl_user_profile_operator = PythonOperator(
    task_id='crawl_user_profile',
    python_callable=crawl_user_profile,
    dag=dag
)

crawl_user_score_operator = PythonOperator(
    task_id='crawl_user_score',
    python_callable=crawl_user_score,
    dag=dag
)

email_operator = PythonOperator(
    task_id='email_operator',
    python_callable=email,
    dag=dag
)

crawl_user_list_operator >> crawl_anime_operator >> crawl_user_profile_operator >> crawl_user_score_operator >> email_operator

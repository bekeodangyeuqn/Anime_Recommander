from airflow import DAG
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

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

def RecommenderNet(num_users, num_animes, embedding_size=128):
    # User input layer and embedding layer
    user = Input(name='user_encoded', shape=[1])
    user_embedding = Embedding(name='user_embedding', input_dim=num_users, output_dim=embedding_size)(user)
    
    # Anime input layer and embedding layer
    anime = Input(name='anime_encoded', shape=[1])
    anime_embedding = Embedding(name='anime_embedding', input_dim=num_animes, output_dim=embedding_size)(anime)
    
    # Dot product of user and anime embeddings
    dot_product = Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
    flattened = Flatten()(dot_product)
    
    # Dense layers for prediction
    dense = Dense(64, activation='relu')(flattened)
    output = Dense(1, activation='sigmoid')(dense)
    
    # Create and compile the model
    model = Model(inputs=[user, anime], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=["mae", "mse"])
    
    return model

# Learning rate schedule function
def lrfn(epoch):
    # Define the initial learning rate, minimum learning rate, maximum learning rate, and batch size
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.00005

    # Define the number of epochs for ramp-up, sustain, and exponential decay
    rampup_epochs = 5
    sustain_epochs = 0
    exp_decay = .8

    if epoch < rampup_epochs:
        return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch - rampup_epochs - sustain_epochs) + min_lr

def train_model():
    df=pd.read_csv('/opt/airflow/data/users-score-2023.csv', usecols=["user_id","anime_id","rating"])
    
    # Scaling our "rating" column
    # Create a MinMaxScaler object
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the 'score' column between 0 and 1
    df['scaled_score'] = scaler.fit_transform(df[['rating']])

    # Encoding categorical data

    ## Encoding user IDs
    user_encoder = LabelEncoder()
    df["user_encoded"] = user_encoder.fit_transform(df["user_id"])
    num_users = len(user_encoder.classes_)

    # Save the user_encoder to a .pkl file
    with open('user_encoder.pkl', 'wb') as file:
        pickle.dump(user_encoder, file)

    ## Encoding anime IDs
    anime_encoder = LabelEncoder()
    df["anime_encoded"] = anime_encoder.fit_transform(df["anime_id"])
    num_animes = len(anime_encoder.classes_)
    with open('anime_encoder.pkl', 'wb') as file:
        pickle.dump(anime_encoder, file)

    # Shuffle the dataset
    df = shuffle(df, random_state=100)

    # Create feature matrix X and target variable y
    X = df[['user_encoded', 'anime_encoded']].values
    y = df["scaled_score"].values

    test_set_size = 10000  # Number of samples to include in the test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=73)

    # Prepare input data for model training and evaluation
    X_train_array = [X_train[:, 0], X_train[:, 1]]
    X_test_array = [X_test[:, 0], X_test[:, 1]]

    model = RecommenderNet(num_users, num_animes)

    # Learning rate scheduler callback
    lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)

    # File path for saving the model weights
    checkpoint_filepath = '/opt/airflow/data/workingmyanime.weights.h5'

    # Model checkpoint callback to save the best weights
    model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True)

    # Early stopping callback to prevent overfitting
    early_stopping = EarlyStopping(patience=3, monitor='val_loss', mode='min', restore_best_weights=True)

    # Define the list of callbacks
    my_callbacks = [
        model_checkpoints,
        lr_callback,
        early_stopping
    ]

    batch_size = 10000

    # Model training
    history = model.fit(
        x=X_train_array,
        y=y_train,
        batch_size=batch_size,
        epochs=20,
        verbose=1,
        validation_data=(X_test_array, y_test),
        callbacks=my_callbacks
    )

    model.load_weights(checkpoint_filepath)

    # Training results visualization

    # plt.plot(history.history["loss"][0:-2])
    # plt.plot(history.history["val_loss"][0:-2])
    # plt.title("Training Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.legend(["Train", "Validation"], loc="upper left")
    # plt.show()

dag = DAG(
    'train_dag',
    default_args={
        'retries': 3,
        'retry_delay': timedelta(seconds=30),
        'email_on_failure': True,
        'email': ['thaiduiqn@gmail.com'],
    },
    description='A process to train the model for recommendation system',
    schedule_interval=timedelta(days=7),
    start_date= datetime.today() - timedelta(days=7),
    tags=['thai130102']
)

PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

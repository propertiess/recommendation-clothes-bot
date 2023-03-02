import numpy as np
import pandas as pd
import os
import shutil
import tensorflow as tf
import tensorflow.keras as keras
from keras import Model
from keras.applications import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from keras.utils import plot_model
import cv2
import pathlib
from sklearn.metrics.pairwise import linear_kernel
import glob


DATASET_PATH = './data'
TEMP_PATH = './temp_recommendation'


def init_data():
    df = pd.read_csv(f'{DATASET_PATH}/styles.csv',
                     nrows=6000, on_bad_lines='skip')
    df['image'] = df.apply(lambda x: str(x['id']) + ".jpg", axis=1)
    df = df.reset_index(drop=True)

    return df


def init_model():
    # image dim
    img_width, img_height, chnl = 200, 200, 3

    # DenseNet121
    densenet = DenseNet121(include_top=False, weights='imagenet',
                           input_shape=(img_width, img_height, chnl))
    densenet.trainable = False

    # Add Layer Embedding
    model = keras.Sequential([
        densenet,
        GlobalMaxPooling2D()
    ])

    return model


def get_embeddings():
    return pd.read_csv('embeddings.csv')


def get_recommendations_list(index, df, cosine_sim):
    # Get the pairwise similarity scores of all clothes with that one
    sim_scores = list(enumerate(cosine_sim[index]))

    # Sort the clothes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar clothes
    sim_scores = sim_scores[1:6]

    cloth_indexes = [i[0] for i in sim_scores]

    return df['image'].iloc[cloth_indexes]


def clear_dir():
    folder = TEMP_PATH
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def recommendation(index, df, embeddings):
    clear_dir()

    embeddings = get_embeddings()

    isExist = index < 6000 and index >= 0

    if not isExist:
        return False

    cosine_sim = linear_kernel(embeddings, embeddings)

    recommendation_list = get_recommendations_list(index, df, cosine_sim)

    chosen_img = cv2.imread(f'{DATASET_PATH}/images/' + df.iloc[index].image)
    cv2.imwrite('original.jpg', chosen_img)

    for i in recommendation_list:
        img = cv2.imread(f'{DATASET_PATH}/images/{i}')
        cv2.imwrite(f'{TEMP_PATH}/{i}', img)

    return True


def predict(model, img):
    img = image.load_img(f'{DATASET_PATH}.images/{img}',
                         target_size=(200, 200))

    var = image.img_to_array(img)

    var = np.expand_dims(var, axis=0)
    var = preprocess_input(var)

    return model.predict(var).reshape(-1)


def write_embeddings(model, data):
    embeddings = data['image'].apply(lambda x: predict(model, x))

    embeddings.to_csv('embeddings.csv')

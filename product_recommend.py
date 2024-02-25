### Importing required libraries####
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
# from annoy import AnnoyIndex
import pickle
import os
from tqdm import tqdm
import time
st.title('Fashion Recommender system')


@st.cache_resource
def loader():
    model = ResNet50(weights='imagenet', include_top=False,
                     input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPool2D()])
    model.summary()
    return model


@st.cache_data
def load_data():
    df = pd.read_csv(
        r'E:/Projects/DeepLearning_projects/datasets/fashion/styles.csv')
    return df

# -- Training and Saving the features
# def image_preprocess(path, model):
#     img = image.load_img(path, target_size=(224, 224))
#     img_arr = image.img_to_array(img)
#     ex_img_arr = np.expand_dims(img_arr, axis=0)
#     pre_pr_img = preprocess_input(ex_img_arr)
#     result = model.predict(pre_pr_img).flatten()
#     normal_result = result/norm(result)
#     return normal_result

# path = r'E:\Projects\DeepLearning_projects\datasets\fashion\images'

# images = [os.path.join(path, files) for files in os.listdir(path)]


model = loader()
df = load_data()

# pickle.dump(images, open('images.pkl', 'wb'))
# feature_list = []
# for file in tqdm(images):
#     feature_list.append(image_preprocess(file, model))
# pickle.dump(feature_list, open('features.pkl', 'wb'))


file_img = pickle.load(open(
    r'E:\Projects\DeepLearning_projects\images.pkl', 'rb'))
feature_list = (pickle.load(open(
    r'E:\Projects\DeepLearning_projects\features.pkl', 'rb')))


def Save_img(upload_img):
    with open(os.path.join('E:/Projects/Deeplearning_Projects/uploads/', upload_img.name), 'wb') as f:
        f.write(upload_img.getbuffer())
        return True
    return False


def feature_extraction(path, model):
    # Load image in size of 224,224,3
    img = image.load_img(path, target_size=(224, 224))
    img_arr = image.img_to_array(img)  # storing into array
    # Expanding the dimension of image
    ex_img_arr = np.expand_dims(img_arr, axis=0)
    pre_pr_img = preprocess_input(ex_img_arr)  # preprocessing the image
    result = model.predict(pre_pr_img).flatten()  # to make 1d vector
    # Normalize the result using norm func from linalg(numpy)
    normal_result = result/norm(result)
    return normal_result


def prod_recom(features, feature_list):
    # using brute force algo here as data is not too big
    neb = NearestNeighbors(
        n_neighbors=10, algorithm='brute', metric='euclidean')
    neb.fit(feature_list)  # fit with feature list
    # return distance and index but we use index to find out nearest images from stored features vector
    dist, ind = neb.kneighbors([features])
    return ind


# To display upload button on screen
upload_img = st.file_uploader(
    "Choose the product image", type=['png', 'jpeg', 'jpg'])

# Condition to check if image got uploaded then call save_img method to save and preprocess image followed by extract features and recommendation
if upload_img is not None:
    if Save_img(upload_img):
        st.image(Image.open(upload_img))
        st.subheader('Extracting :blue[features...]', divider='rainbow')
        features = feature_extraction(
            os.path.join("uploads", upload_img.name), model)
        progress_text = "Generating recommendations..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.02)
            # to add progress bar untill feature got extracted
            my_bar.progress(percent_complete + 1, text=progress_text)
        # calling recom. func to get 10 recommendation
        ind = prod_recom(features, feature_list)
        my_bar.empty()
        # to create 10 section of images into the screen
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(
            10)

        # for each section image shown by below code
        with col1:
            st.image(Image.open(file_img[ind[0][0]]))
        with col2:
            st.image(Image.open(file_img[ind[0][1]]))
        with col3:
            st.image(Image.open(file_img[ind[0][2]]))
        with col4:
            st.image(Image.open(file_img[ind[0][3]]))
        with col5:
            st.image(Image.open(file_img[ind[0][4]]))
        with col6:
            st.image(Image.open(file_img[ind[0][5]]))
        with col7:
            st.image(Image.open(file_img[ind[0][6]]))
        with col8:
            st.image(Image.open(file_img[ind[0][7]]))
        with col9:
            st.image(Image.open(file_img[ind[0][8]]))
        with col10:
            st.image(Image.open(file_img[ind[0][9]]))

        # st.text("Using Spotify ANNoy")
        # df = pd.DataFrame({'img_id':file_img, 'img_repr': feature_list})
        # f=len(df['img_repr'][0])
        # ai=AnnoyIndex(f,'angular')
        # for i in tqdm(range(len(feature_list))):
        #     v=feature_list[i]
        #     ai.add_item(i,v)
        # ai.build(10) # no of binary tress want to build more number of tree more accuracy
        # neigh=(ai.get_nns_by_item(0,5))
        # with col1:
        #         st.image(Image.open(file_img[neigh[0]]))
        # with col2:
        #                 st.image(Image.open(file_img[neigh[1]]))
        # with col3:
        #                 st.image(Image.open(file_img[neigh[2]]))
        # with col4:
        #                 st.image(Image.open(file_img[neigh[3]]))

        # for i in range(len(neigh)):
        #     with st.columns(i):
        #         st.image(Image.open(file_img[neigh[i]]))
    else:
        st.header("Can't process the uploaded image")

st.scatter_chart(df, y="masterCategory", x="gender")

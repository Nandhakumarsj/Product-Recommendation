### Importing required libraries####
import pickle
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from werkzeug.utils import secure_filename
# from annoy import AnnoyIndex
# from tqdm import tqdm

# Streamlit title
st.title('Fashion Recommender system')

# Load the ResNet50 model without the top layers, and freeze the weights


@st.cache_resource
def loader():
    # Load the ResNet50 model with 'imagenet' weights
    model = ResNet50(weights='imagenet', include_top=False,
                     input_shape=(224, 224, 3))
    # Set the model's weights to be non-trainable
    model.trainable = False
    # Create a new Sequential model with the ResNet50 model and a GlobalMaxPool2D layer
    model = tf.keras.Sequential([model, GlobalMaxPool2D()])
    # Print the summary of the new model
    model.summary()
    # Return the new model
    return model


# Load the data from the CSV file
@st.cache_data
def load_data():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(
        r'E:/Projects/DeepLearning_projects/Image Recommendation/datasets/fashion/styles.csv')

    # Return the DataFrame
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

# path = r'E:\Projects\DeepLearning_projects\Image Recommendation\datasets\fashion\images'

# images = [os.path.join(path, files) for files in os.listdir(path)]


# Load the Model
model = loader()
# Load the Dataframe for visualization
df = load_data()

#  Function to show the prediction results on the Image Uploader
# def predictor(image_file):
# Check if an image has been uploaded
# if image_file:
# Pre-process the image using the defined function
# pred = image_preprocess(image_file, model)

# Display the Prediction Results
# st.write("The predicted style is     : ", classes[np.argmax(pred)])
# st.write("Confidence of the prediction : ", max(pred)*100,"%")

# Dumps images in binary format for Streamlit app performance
# pickle.dump(images, open('images.pkl', 'wb'))
# feature_list = []
# for file in tqdm(images):
#     feature_list.append(image_preprocess(file, model))
# Dumps Feature for future use
# pickle.dump(feature_list, open('features.pkl', 'wb'))

# Loads the Images and Features from the dumps
file_img = pickle.load(open(
    r'E:\Projects\DeepLearning_projects\Image Recommendation\images.pkl', 'rb'))
feature_list = (pickle.load(open(
    r'E:\Projects\DeepLearning_projects\Image Recommendation\features.pkl', 'rb')))


def Save_img(upload_img):
    """
    This function saves an uploaded image to a specified directory.

    Parameters:
    upload_img (FileStorage): The uploaded image file.

    Returns:
    str: The path to the saved image.
    """

    # Get the filename of the uploaded image
    filename = secure_filename(upload_img.name)

    # Define the path to save the image
    img_path = os.path.join(
        "E:/Projects/Deeplearning_Projects/Image Recommendation/uploads/", filename)

    # Save the uploaded image to the specified path
    with open(img_path, 'wb') as f:
        f.write(upload_img.getbuffer())

    # Return the path to the saved image
    return img_path


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

# Condition to check if image got uploaded then call save_img method to save and preprocess image
if upload_img is not None:
    # Temporarily saving images in the uploads directory for processing
    if (img_path := Save_img(upload_img)):
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

        # Cleans up the upload directory after use
        os.remove(img_path)
    else:
        st.header("Can't process the uploaded image")

# Visualize the chart based on the uploaded image
st.scatter_chart(df, y="masterCategory", x="gender")

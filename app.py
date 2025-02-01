import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# loading the trained model
model = tf.keras.models.load_model('regression_model.h5')

# loading the scalers
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('onehot_encoder_gender.pkl', 'rb') as file:
    onehot_encoder_gender = pickle.load(file)

# Let's build the UI with Streamlit


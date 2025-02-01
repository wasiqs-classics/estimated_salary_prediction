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

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# Let's build the UI with Streamlit

st.title("Salary Prediction Assignment")

# User input
geography = st.selectbox('🗺️Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('🚻Gender', label_encoder_gender.classes_)
age = st.slider('📅Age',18,92)
balance = st.number_input('💵Balance')
credit_Score = st.number_input('💯Credit Score')
tenure = st.slider('⌛Tenure',0,10)
num_of_products = st.slider('🔢Number of Products',1,4)
has_cr_card = st.select_slider('💳Has Credit Card',[0,1])
is_active_member = st.selectbox('✅Is Active Member',[0,1])
has_exited = st.selectbox('❌Has Exited',[0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_Score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' :[age],
    'Tenure' :[tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [has_exited]

})

# One hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one - hot encoded columns with input data

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the salary

prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(f"Predicted Salary: 💰{prediction_probability:.2f}")



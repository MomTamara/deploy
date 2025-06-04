import joblib
import streamlit as st
import numpy as np

model = joblib.load('Esther.pkl')
st.title('Iris Flower Predictor')


sepal_length = st.number_input('sepal length')
sepal_width = st.number_input('sepal width')
petal_length = st.number_input('petal length')
petal_width = st.number_input('petal width')

if st.button('predict'):

    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    probabilities = model.predict_proba(user_input)
    st.write(f'{probabilities}')
    species = ('Setosa', 'Versicolor', 'Virginca')

    for i in range(3):
        st.write(f'{species[i]} : {probabilities[0][i]*100}')

    prediction = model.predict(user_input)
    st.success(f'The predicted class is {species[prediction[0]]}')

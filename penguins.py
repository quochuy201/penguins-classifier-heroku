import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


def main():
    st.write("""
    # Penguin Prediction App

    This app predicts the **Palmer Penguin** species!

    Data obtained from palmerpenguins library.

    This app is from freeCodeCamp.org - Data Professor
    """)

    st.sidebar.header('User Input Feature')
    st.sidebar.markdown("""[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)""")


    # Collects user input features into dataframe
    uploaded_file =  st.sidebar.file_uploader("Upload your input CSV file", type =["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = user_input_features()

    # Combines user input features with entire penguins dataset
    # this will be userful for the encoding phase
    penguins_raw = pd.read_csv('penguins_cleaned.csv')
    penguins = penguins_raw.drop(columns=['species'])
    df = pd.concat([input_df, penguins], axis = 0)

    # Encoding categorical features
    cols = ['sex', 'island']
    df = encoding(df, cols)

    # select user input
    userInput = df[:1]
    
    # Displays the user input features
    st.subheader('User Input features')
    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(df)

    st.write(userInput)
    # load classification model:
    with open('penguins_clf.pkl', 'rb') as f:
       load_clfier = pickle.load(f)

    # apply model to make predictions
    prediction = load_clfier.predict(userInput)
    prediction_proba = load_clfier.predict_proba(userInput)

    st.subheader('Prediction (Adelie,Chinstrap,Gentoo)')
    penguins_species = ['Adelie','Chinstrap','Gentoo']
    #st.write(prediction)
    st.write(penguins_species[int(prediction)])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)


def user_input_features():
    island = st.sidebar.selectbox('Island', ('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex',('male','female'))
    bill_length_mm = st.sidebar.slider('Bill length (mm)',32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 210.0)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features

def encoding(df, cols):
    for col in cols:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    return df

# If the program is run (instead of imported), run the game:
if __name__ == '__main__':
    main()
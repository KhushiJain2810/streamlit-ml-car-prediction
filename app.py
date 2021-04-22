import streamlit as st
import pickle
import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler

# reads .pkl file and loads the trained model in app.py file
model, ct, sc = pickle.load(open('D:/JUPYTER/YAY/DEPLOYMENT/random_forst_regression_model.pkl', 'rb'))

def predict_price(ct, sc, years_old, pp, kms_driven, fuel_type, seller_type, transmission, owner):
    
    input = pd.Series([years_old, pp, kms_driven, fuel_type, seller_type, transmission, owner])
    input = input.values

    # one hot encoding + feature scaling
    input = ct.transform([input])
    input[:, 9:] = sc.transform(input[:, 9:])
    
    # Prediction
    prediction = model.predict(input)
    return float(prediction)


def main():
    st.title('Car Price Prediction')

    html_temp = """

        <div style="background-color:#025246 ;padding:10px;">
        <h2 style="color:white;text-align:center;">Used Car Price Prediction ML App </h2>
        </div>

    """

    st.markdown(html_temp, unsafe_allow_html=True)

    years_old = st.text_input('How many years old ?', 'Enter number of years')
    pp = st.text_input('What is the current market value of the car??', 'Enter price')
    kms_driven = st.text_input('How many kilometers the car has driven?', 'Enter distance in kilometers')
    fuel_type = st.text_input('What is the type of fuel used?', 'CNG/Diesel/Petrol')
    seller_type = st.text_input('What is the type of seller?', 'Dealer/Individual')
    transmission = st.text_input('What is the type of Transmission?', 'Automatic/Manual')
    owner = st.text_input('How many owners has the car had?', '0/1/3')
    
    if st.button('Predict'):
        output = predict_price(ct, sc, int(years_old), float(pp), int(kms_driven), fuel_type, seller_type, transmission, int(owner))
        # st.success('The selling price of this car will be approximately {} lakhs'.format(round(output, 2)))
        st.success(output)

if __name__ == '__main__':
    main()
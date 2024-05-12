import streamlit as st
import numpy as np
import pandas as pd

from joblib import load

model = load('app/gradient_boost_new.pkl')

# Function to predict values
def predict(dataframe):
    # Assuming the model expects columns in a certain order, you should reorder them if necessary
    dataframe = dataframe[['age', 'bmi', 'hypertension', 'heart_disease', 
                           'hba1c_level', 'blood_glucose_level', 
                           'gender_Female', 'gender_Male', 
                           'smoking_history_current', 'smoking_history_ever', 
                           'smoking_history_former', 'smoking_history_never', 'smoking_history_no_info',
                           'smoking_history_not current']]
    
    # Convert DataFrame to numpy array for prediction
    prediction = model.predict(dataframe)
    return prediction

# Streamlit app
def main():
    st.title('Medical Predictor')

    # User input
    age = st.slider('Age', min_value=1, max_value=120, value=25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    bmi = st.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=50.0, value=20.0)
    hypertension = st.checkbox('Hypertension')
    heart_disease = st.checkbox('Heart Disease')
    smoking_history = st.selectbox('Smoking History',  ["Never","Current","Former","Ever","Not Current","No Info"])
    hba1c_level = st.number_input('HbA1c Level', min_value=3.0, max_value=15.0, value=5.0)
    blood_glucose_level = st.number_input('Blood Glucose Level', min_value=50, max_value=300, value=100)

    if st.button('Predict'):
        # Define smoking history categories
        smoking_history_categories = {
        "Never": "smoking_history_never",
        "Current": "smoking_history_current",
        "Former": "smoking_history_former",
        "Ever": "smoking_history_ever",
        "Not Current": "smoking_history_not current",
        "No Info": "smoking_history_no_info"
        }
    
        # Convert gender to binary
        gender_binary = 1 if gender == 'Female' else 0

        # Create the prediction dictionary
        predict_dict = {
            'age': [age],  # Wrap in list to ensure it's treated as a sequence
            'bmi': [bmi],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'hba1c_level': [hba1c_level],
            'blood_glucose_level': [blood_glucose_level],
            'gender_Female': [gender_binary],
            'gender_Male': [1 - gender_binary],  # inverse of Female
        }

        # Add smoking history category
        for category, column_name in smoking_history_categories.items():
            predict_dict[column_name] = [1 if smoking_history == category else 0]

        # Make prediction
        df = pd.DataFrame(predict_dict)
        prediction = predict(df)

        # Show prediction
        st.write('Predicted Value:', prediction)


if __name__ == '__main__':
    main()
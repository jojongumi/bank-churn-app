import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Churn for Bank Customers')
st.header('This app predicts customer churn in a bank')
st.write("""### Click 'Predict Results' for your prediction""")
st.sidebar.write("""### Please fill in the information below to predict the customer churn in a bank:""")

def user_input_features():
	CreditScore = st.sidebar.slider("What is the customer's credit score? (Min: 300, Max: 850)", min_value=300,
									max_value=850)

	Age = st.sidebar.number_input("How old is the customer?", min_value=18, max_value=110)
	Tenure = st.sidebar.number_input("What is the number of years that the customer has been a client of the bank?", min_value=0, max_value=100)
	Balance = st.sidebar.number_input("What is the customer's account balance?")
	NumOfProducts = st.sidebar.number_input("What is the number of products that a customer has purchased through the bank?", min_value=0)

	HasCrCard = st.sidebar.selectbox("Does the customer have a credit card?", ("Yes", "No"))
	HasCrCard = 0 if HasCrCard == 'Yes' else 1

	IsActiveMember = st.sidebar.selectbox("Is the customer an active member?", ("Yes", "No"))
	IsActiveMember = 0 if HasCrCard == 'Yes' else 1

	EstimatedSalary = st.sidebar.number_input("What is the customer's estimated salary?", min_value=0)

	Gender = st.sidebar.selectbox("What is the customer's gender?", ("Male", "Female"))
	Gender = 0 if Gender == 'Male' else 1

	Geography = st.sidebar.selectbox("Select customer's location:", ("France", "Germany", "Spain"))

	data = {
		'CreditScore': CreditScore,
		'Age ': Age,
		'Tenure': Tenure,
		'Balance': Balance,
		'NumOfProducts': NumOfProducts,
		'HasCrCard': HasCrCard,
		'IsActiveMember': IsActiveMember,
		'EstimatedSalary': EstimatedSalary,
		'Gender': Gender,
		'Geography': Geography,
	}

	features = pd.DataFrame(data,index=[0])
	return features

input_df = user_input_features()

encode = ['Geography']
for col in encode:
	dummy = pd.get_dummies(input_df[col], prefix=col)
	df = pd.concat([input_df, dummy], axis=1)
	del df[col]
input_df = df[:1]

result = ''

if st.button('Prediction Result'):
	load_model = pickle.load(open("myModel.pkl", "rb"))
	st.subheader('User Input Features:')
	st.write(input_df)
	prediction = load_model.predict(input_df)
	prediction_probability = load_model.predict_proba(input_df)
	st.subheader("Prediction:")
	if prediction[0] == 0:
		st.success('The customer did not leave')
	else:
		st.success('The customer left')
	st.subheader("Prediction Probability:")
	st.write(prediction_probability)
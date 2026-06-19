import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os
st.title("Passenger Survival Chance in the Titanic Journey")  

pclass=st.slider('Enter the Passenger class for the user', 1,3)
sex=st.selectbox("Enter the Passenger Gender",['male','female'])
sibsp=st.slider('Enter the Passenger Total number of Siblings and Spouse', 1,8)
parch=st.slider('Enter the Passenger Total number of Parents and childs', 1,6)
fare=st.number_input('Enter the Fare Passager Paid')
embarked=st.selectbox('Enter the station from where the Passenger has started journey',['Southampton','Chebourg','Queenstown'])


data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])
 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(
    os.path.join(BASE_DIR, "titanic_model.h5")
)

import pickle

with open(r'C:\Users\himan\.ipython\Gen-AI Tasks(Himanshu)\Deep_Learning\ANN Model\label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

import pickle

with open(r'C:\Users\himan\.ipython\Gen-AI Tasks(Himanshu)\Deep_Learning\ANN Model\onehot_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)
import pickle

with open(r'C:\Users\himan\.ipython\Gen-AI Tasks(Himanshu)\Deep_Learning\ANN Model\scaler_encoder.pkl', 'rb') as file:
    scaler_encoder = pickle.load(file)       


data['Sex']=label_encoder.transform(data['Sex'])

embarked=onehot_encoder.transform(data[['Embarked']])

embarked=pd.DataFrame(embarked,columns=onehot_encoder.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler_encoder.transform(data[['Pclass','SibSp','Parch','Fare']])

y=model.predict(data)

y=y[0][0]

def Chance(y):
    if y>0.5:
      st.write('The Pessenger Will Servive')
    else:
        st.write('The Pessenger Will Not Servive')  

if st.button('Predict Survival Chance'):
    st.write('Probability of Passanger Survival is',y)
    (Chance(y))        
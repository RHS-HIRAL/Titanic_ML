import streamlit as st
import pandas as pd
import joblib  # or pickle
from sklearn.ensemble import RandomForestClassifier

@st.cache
def load_model():
    return joblib.load("titanic_rf.pkl")

@st.cache
def load_data():
    return pd.read_csv("data/titanic.csv")

st.title("Titanic Survival Prediction")

df = load_data()
st.write("### First five rows of the dataset", df.head())

model = load_model()

st.sidebar.header("Passenger Features")
# e.g., allow user inputs
age = st.sidebar.slider("Age", 0, 100, 30)
pclass = st.sidebar.selectbox("Pclass", df["Pclass"].unique())
# … add other features …

# Construct input DataFrame
X_new = pd.DataFrame([[age, pclass]], columns=["Age", "Pclass"])
pred = model.predict(X_new)[0]
proba = model.predict_proba(X_new)[0,1]

st.write(f"**Predicted Survival:** {'Yes' if pred else 'No'}")
st.write(f"**Probability:** {proba:.2%}")

import os
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Function to load data with dynamic path
@st.cache_data()
def load_data():
    # Build path relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "titanic.csv")
    return pd.read_csv(csv_path)

# Function to train model on-the-fly and cache it
@st.cache_resource()
def train_model(df):
    # Feature and target selection
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']

    # Numeric pipeline: median imputation
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    numeric_transformer = SimpleImputer(strategy='median')

    # Categorical pipeline: most frequent imputation + one-hot encoding
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Full pipeline: preprocessing + classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X, y)
    return model

# Streamlit App
st.title("Titanic Survival Prediction")

# Load data and preview
df = load_data()
st.write("### Dataset Preview", df.head())

# Train model once and cache
titanic_model = train_model(df)

# Sidebar inputs
st.sidebar.header("Passenger Features")
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=30)
pclass = st.sidebar.selectbox("Pclass", sorted(df['Pclass'].unique()))
sex = st.sidebar.selectbox("Sex", df['Sex'].unique())
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.sidebar.number_input("Parents/Children Aboard", min_value=0, max_value=8, value=0)
fare = st.sidebar.slider("Fare", min_value=0.0, max_value=float(df['Fare'].max()), value=float(df['Fare'].median()))
embarked = st.sidebar.selectbox("Embarked", df['Embarked'].dropna().unique())

# Prepare input DataFrame for prediction
input_df = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}])

# Predict and display results
prediction = titanic_model.predict(input_df)[0]
probability = titanic_model.predict_proba(input_df)[0][1]

st.write(f"**Predicted Survival:** {'Yes' if prediction else 'No'}")
st.write(f"**Probability of Survival:** {probability:.2%}")

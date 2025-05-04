import os
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Function to load data with dynamic path and normalize columns
@st.cache_data()
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "titanic.csv")
    df = pd.read_csv(csv_path)
    # Normalize column names: strip spaces and lowercase
    df.columns = df.columns.str.strip().str.lower()
    return df

# Function to train model on-the-fly and cache it
@st.cache_resource()
def train_model(df):
    # Ensure required columns exist
    required = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in data: {missing}")
        st.stop()

    # Feature and target selection
    X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
    y = df['survived']

    # Numeric pipeline: median imputation
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    numeric_transformer = SimpleImputer(strategy='median')

    # Categorical pipeline: most frequent imputation + one-hot encoding
    categorical_features = ['pclass', 'sex', 'embarked']
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

# Main Streamlit App
st.title("Titanic Survival Prediction")

# Load data
df = load_data()

# Show columns in case of mismatch
st.write("**Available columns:**", list(df.columns))

# Train model
titanic_model = train_model(df)

# Sidebar inputs
st.sidebar.header("Passenger Features")
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=30)
pclass = st.sidebar.selectbox("Pclass", sorted(df['pclass'].unique()))
sex = st.sidebar.selectbox("Sex", sorted(df['sex'].unique()))
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", min_value=0, max_value=int(df['sibsp'].max()), value=0)
parch = st.sidebar.number_input("Parents/Children Aboard", min_value=0, max_value=int(df['parch'].max()), value=0)
fare = st.sidebar.slider("Fare", min_value=0.0, max_value=float(df['fare'].max()), value=float(df['fare'].median()))
embarked_vals = df['embarked'].dropna().unique()
embarked = st.sidebar.selectbox("Embarked", sorted(embarked_vals))

# Prepare input DataFrame
input_df = pd.DataFrame([{
    'pclass': pclass,
    'sex': sex,
    'age': age,
    'sibsp': sibsp,
    'parch': parch,
    'fare': fare,
    'embarked': embarked
}])

# Predict and display
prediction = titanic_model.predict(input_df)[0]
probability = titanic_model.predict_proba(input_df)[0][1]

st.write(f"**Predicted Survival:** {'Yes' if prediction else 'No'}")
st.write(f"**Probability of Survival:** {probability:.2%}")

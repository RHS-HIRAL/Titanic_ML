import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Cache the dataset loading to speed up reruns
@st.cache_data()
def load_data():
    df = pd.read_csv("data/titanic.csv")
    return df

# Cache the model training; trains once and reuses
@st.cache_resource()
def train_model(df):
    # Feature and target selection
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
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

# Application title
st.title("Titanic Survival Prediction")

# Load data and show preview
df = load_data()
st.write("### Dataset Preview", df.head())

# Train or load cached model
model = train_model(df)

# Sidebar for user inputs
st.sidebar.header("Passenger Features")
age = st.sidebar.slider("Age", 0, 100, 30)
pclass = st.sidebar.selectbox("Pclass", sorted(df["Pclass"].unique()))
sex = st.sidebar.selectbox("Sex", df["Sex"].unique())
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.sidebar.number_input("Parents/Children Aboard", min_value=0, max_value=8, value=0)
fare = st.sidebar.slider("Fare", 0.0, float(df["Fare"].max()), float(df["Fare"].median()))
embarked = st.sidebar.selectbox("Embarked", df["Embarked"].dropna().unique())

# Create a new DataFrame for prediction
X_new = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}])

# Predict and display results
prediction = model.predict(X_new)[0]
probability = model.predict_proba(X_new)[0][1]

st.write(f"**Predicted Survival:** {'Yes' if prediction else 'No'}")
st.write(f"**Probability of Survival:** {probability:.2%}")

# Show confusion matrix
st.write("### Confusion Matrix")
st.write(pd.crosstab(df['Survived'], model.predict(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]), rownames=['Actual'], colnames=['Predicted']))
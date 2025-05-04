import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Utility: load and preprocess raw data
def load_raw_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "titanic.csv")
    df = pd.read_csv(path)
    # strip column names
    df.columns = df.columns.str.strip()
    return df

# Build preprocessing pipeline
def make_preprocessor(df):
    # Identify numeric and categorical features (exclude Survived)
    features = [c for c in df.columns if c != 'Survived']
    numeric = df[features].select_dtypes(include=['number']).columns.tolist()
    categorical = [c for c in features if c not in numeric]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric),
        ('cat', cat_pipeline, categorical)
    ])
    return preprocessor, numeric, categorical

# Train both models and return pipelines + metrics
def train_models(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    preprocessor, numeric, categorical = make_preprocessor(df)

    # Logistic Regression pipeline
    log_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # SVC pipeline with scaling and probability
    svc_pipe = Pipeline([
        ('pre', preprocessor),
        ('scale', StandardScaler(with_mean=False)),
        ('clf', SVC(kernel='linear', probability=True))
    ])

    # Fit
    log_pipe.fit(X_train, y_train)
    svc_pipe.fit(X_train, y_train)

    # Evaluate
    models = {'Logistic Regression': log_pipe, 'SVM (linear)': svc_pipe}
    results = {}
    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cv = cross_val_score(pipe, X_test, y_test, cv=KFold(n_splits=5))
        results[name] = {'accuracy': acc, 'confusion_matrix': cm, 'cv_scores': cv}

    return models, results, X_test, y_test

# Plotting functions
def plot_confusion(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    return fig

# Main App
st.title("Titanic ML: Model Comparison & Prediction")

# Load data once
df_raw = load_raw_data()

# Sidebar navigation
page = st.sidebar.radio("Go to", ['Model Comparison', 'Make a Prediction'])

if page == 'Model Comparison':
    st.header("Model Evaluation and Comparison")
    st.write("Dataset preview:")
    st.dataframe(df_raw.head())
    st.write("### Missing Data Heatmap")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_raw.isnull(), cbar=False)
    ax1.set_title('Missing Data Heatmap')
    st.pyplot(fig1)

    # Train and get results
    models, results, X_test, y_test = train_models(df_raw)

    # Show metrics
    for name, info in results.items():
        st.subheader(name)
        st.write(f"Accuracy: {info['accuracy']:.2%}")
        st.write(f"Cross-validation scores: {info['cv_scores']} (mean {info['cv_scores'].mean():.2%})")
        fig_cm = plot_confusion(info['confusion_matrix'], f"Confusion Matrix - {name}")
        st.pyplot(fig_cm)

elif page == 'Make a Prediction':
    st.header("Predict Survival for a New Passenger")
    st.write(df_raw.head(0))  # show columns
    # Sidebar inputs for each feature
    inputs = {}
    for col in df_raw.columns:
        if col == 'Survived':
            continue
        if df_raw[col].dtype == 'int64' or df_raw[col].dtype == 'float64':
            if col in ['Age', 'SibSp', 'Parch', 'Fare']:
                inputs[col] = st.sidebar.slider(col, float(df_raw[col].min()), float(df_raw[col].max()), float(df_raw[col].median()))
            else:
                inputs[col] = st.sidebar.number_input(col, float(df_raw[col].min()), float(df_raw[col].max()), float(df_raw[col].median()))
        else:
            options = df_raw[col].dropna().unique().tolist()
            inputs[col] = st.sidebar.selectbox(col, options)

    # Choose model
    model_name = st.sidebar.selectbox("Model", ['Logistic Regression', 'SVM (linear)'])

    # Load models (cached)
    models, _, _, _ = train_models(df_raw)
    model = models[model_name]

    # Build DataFrame
    X_new = pd.DataFrame([inputs])

    # Prediction
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]
    st.write(f"**Survival Prediction:** {'Yes' if pred else 'No'}")
    st.write(f"**Probability of Survival:** {proba:.2%}")

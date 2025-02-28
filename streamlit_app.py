import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Employee Attrition Prediction")

uploaded_file = st.file_uploader("/content/IBMEmployee_data.csv", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview")
        st.dataframe(df.head())

        df.dropna(inplace=True)

        drop_columns = ['EmployeeNumber', 'StockOptionLevel']
        df.drop(columns=[col for col in drop_columns if col in df.columns], axis=1, inplace=True)

        selected_features = ['Department', 'JobRole', 'MaritalStatus', 'OverTime', 'JobSatisfaction', 'Age']

        categorical_features = df[selected_features].select_dtypes(include=['object']).columns.tolist()

        encoder = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        X = df[selected_features]
        y = df['Attrition']

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_encoded = encoder.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        st.write("Select Values for Prediction")
        user_input = []
        user_data = {}

        if st.button("Input Values for Prediction"):
            for feature in selected_features:
                if feature in categorical_features:
                    options = df[feature].unique().tolist()
                    value = st.selectbox(f"Select value for {feature}", options)
                else:
                    value = st.slider(f"Select value for {feature}", int(df[feature].min()), int(df[feature].max()), int(df[feature].mean()))

                user_data[feature] = [value]
                user_input.append(value)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        selected_model_name = st.selectbox("Select a classification model", list(models.keys()))
        selected_model = models[selected_model_name]

        st.write("Model Training and Evaluation")

        start_time = time.time()
        selected_model.fit(X_train, y_train)
        train_time = time.time() - start_time

        start_time = time.time()
        y_pred = selected_model.predict(X_test)
        test_time = time.time() - start_time

        train_accuracy = accuracy_score(y_train, selected_model.predict(X_train))
        test_accuracy = accuracy_score(y_test, y_pred)

        st.write(f"Training Accuracy: {train_accuracy:.2f}")
        st.write(f"Testing Accuracy: {test_accuracy:.2f}")
        st.write(f"Training Time: {train_time:.4f} seconds")
        st.write(f"Testing Time: {test_time:.4f} seconds")

        if st.button("Show Classification Report"):
            st.write(f"{selected_model_name} Classification Report")
            st.text(classification_report(y_test, y_pred))

        if st.button("Compare Model Accuracies"):
            st.write("Comparison of Model Accuracies")
            accuracy_results = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy_results[model_name] = accuracy_score(y_test, y_pred)

            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), palette="coolwarm")
            plt.ylabel("Accuracy")
            plt.title("Comparison of Model Accuracies")
            st.pyplot(plt)

        if st.button("Predict Attrition"):
            user_df = pd.DataFrame(user_data)
            user_input_transformed = encoder.transform(user_df)
            prediction = selected_model.predict(user_input_transformed)
            pred_class = label_encoder.inverse_transform(prediction)
            st.write(f"Predicted Attrition: {pred_class[0]}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

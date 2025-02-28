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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Streamlit Title
st.title("🔍 Employee Attrition Prediction")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("### 📊 Dataset Preview")
        st.dataframe(df.head())

        # Drop missing values
        df.dropna(inplace=True)

        # Drop unnecessary columns if present
        drop_columns = ['EmployeeNumber', 'StockOptionLevel']
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

        # Selected features for training
        selected_features = ['Department', 'JobRole', 'MaritalStatus', 'OverTime', 'JobSatisfaction', 'Age']

        # Ensure selected features exist in the dataset
        existing_features = [feature for feature in selected_features if feature in df.columns]

        if not existing_features:
            st.error("❌ None of the selected features are found in the dataset. Please check your CSV file.")
            st.stop()

        # Identify categorical features
        categorical_features = df[existing_features].select_dtypes(include=['object']).columns.tolist()

        # Debugging Print
        print(f"Categorical Features: {categorical_features}")

        # Encoder for categorical variables
        encoder = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        # Define X and y
        X = df[existing_features]
        if 'Attrition' not in df.columns:
            st.error("❌ 'Attrition' column not found in dataset. Please check your CSV file.")
            st.stop()

        y = df['Attrition']

        # Handle missing target values
        if y.isnull().sum() > 0:
            st.error("❌ The target variable contains missing values. Please clean your data.")
            st.stop()

        # Encode target labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Transform categorical variables
        if X.empty:
            st.error("❌ The feature set (X) is empty. Please check your dataset.")
            st.stop()

        X_encoded = encoder.fit_transform(X)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # ---------------- EDA (Exploratory Data Analysis) ---------------- #
        st.write("## 📈 Exploratory Data Analysis (EDA)")

        if st.button("🔍 Show EDA Graphs"):
            # Attrition Count Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=df['Attrition'], palette="coolwarm", ax=ax)
            ax.set_title("Attrition Count")
            st.pyplot(fig)

            # Department Distribution
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.countplot(y=df['Department'], palette="Blues_r", ax=ax)
            ax.set_title("Department-wise Employee Count")
            st.pyplot(fig)

            # Age Distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['Age'], bins=20, kde=True, color="green", ax=ax)
            ax.set_title("Age Distribution of Employees")
            st.pyplot(fig)

            # Job Satisfaction Levels
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=df['JobSatisfaction'], palette="magma", ax=ax)
            ax.set_title("Job Satisfaction Levels")
            st.pyplot(fig)

        # ---------------- Model Training ---------------- #
        st.write("## ⚙️ Model Training and Evaluation")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        selected_model_name = st.selectbox("📌 Select a classification model", list(models.keys()))
        selected_model = models[selected_model_name]

        start_time = time.time()
        selected_model.fit(X_train, y_train)
        train_time = time.time() - start_time

        start_time = time.time()
        y_pred = selected_model.predict(X_test)
        test_time = time.time() - start_time

        train_accuracy = accuracy_score(y_train, selected_model.predict(X_train))
        test_accuracy = accuracy_score(y_test, y_pred)

        st.write(f"✅ **Training Accuracy:** {train_accuracy:.2f}")
        st.write(f"✅ **Testing Accuracy:** {test_accuracy:.2f}")
        st.write(f"⏳ **Training Time:** {train_time:.4f} seconds")
        st.write(f"⏳ **Testing Time:** {test_time:.4f} seconds")

        # ---------------- Classification Report ---------------- #
        if st.button("📋 Show Classification Report"):
            st.write(f"### 📊 {selected_model_name} Classification Report")
            st.text(classification_report(y_test, y_pred))

        # ---------------- Model Accuracy Comparison ---------------- #
        if st.button("📊 Compare Model Accuracies"):
            st.write("### 📈 Model Accuracy Comparison")
            accuracy_results = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy_results[model_name] = accuracy_score(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), palette="coolwarm", ax=ax)
            ax.set_ylabel("Accuracy")
            ax.set_title("Comparison of Model Accuracies")
            st.pyplot(fig)

        # ---------------- Prediction Section ---------------- #
        if st.button("🎯 Input Values for Prediction"):
            user_input = []
            user_data = {}

            for feature in existing_features:
                if feature in categorical_features:
                    options = df[feature].unique().tolist()
                    value = st.selectbox(f"🔹 Select value for {feature}", options)
                else:
                    value = st.slider(f"🔹 Select value for {feature}", int(df[feature].min()), int(df[feature].max()), int(df[feature].mean()))

                user_data[feature] = [value]

            if st.button("🔮 Predict Attrition"):
                user_df = pd.DataFrame(user_data)

                if user_df.empty:
                    st.error("❌ Error: User input data is empty!")
                    st.stop()

                user_input_transformed = encoder.transform(user_df)
                prediction = selected_model.predict(user_input_transformed)
                pred_class = label_encoder.inverse_transform(prediction)

                st.success(f"🔮 Predicted Attrition: **{pred_class[0]}**")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")


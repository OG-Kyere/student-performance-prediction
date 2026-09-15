# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION SYSTEM
# ============================================================
# Author: Python Data Science Project
# Description:
#   Generates student data, cleans it, performs statistical
#   analysis, trains a machine-learning model, evaluates it,
#   creates visualizations, and predicts performance.
#
# Required packages:
#   pip install numpy pandas matplotlib seaborn scikit-learn
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

np.random.seed(42)

DATA_FILE = "student_data.csv"
RESULT_FILE = "student_predictions.csv"
FIGURE_FOLDER = "figures"

os.makedirs(FIGURE_FOLDER, exist_ok=True)


# ============================================================
# 2. GENERATE DATA
# ============================================================

def generate_student_data(n=1000):

    print("\nGenerating student dataset...")

    study_hours = np.random.normal(12, 4, n)
    attendance = np.random.normal(82, 10, n)
    sleep_hours = np.random.normal(6.8, 1.2, n)
    previous_score = np.random.normal(65, 15, n)
    assignments = np.random.normal(75, 12, n)

    # Keep values realistic
    study_hours = np.clip(study_hours, 1, 30)
    attendance = np.clip(attendance, 40, 100)
    sleep_hours = np.clip(sleep_hours, 3, 10)
    previous_score = np.clip(previous_score, 20, 100)
    assignments = np.clip(assignments, 20, 100)

    # Create a relationship between predictors and final score
    final_score = (
        0.35 * study_hours
        + 0.20 * attendance
        + 0.25 * previous_score
        + 0.10 * assignments
        + 1.5 * sleep_hours
        + np.random.normal(0, 7, n)
    )

    final_score = np.clip(final_score, 0, 100)

    data = pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Score": previous_score,
        "Assignment_Score": assignments,
        "Final_Score": final_score
    })

    # Introduce some missing values deliberately
    for column in ["Study_Hours", "Attendance", "Sleep_Hours"]:
        indexes = np.random.choice(
            data.index,
            size=int(0.03 * n),
            replace=False
        )
        data.loc[indexes, column] = np.nan

    return data


# ============================================================
# 3. DATA CLEANING
# ============================================================

def clean_data(data):

    print("\nCleaning dataset...")

    print("\nMissing values before cleaning:")
    print(data.isnull().sum())

    # Replace missing numerical values with median
    numerical_columns = data.select_dtypes(
        include=np.number
    ).columns

    for column in numerical_columns:
        data[column] = data[column].fillna(
            data[column].median()
        )

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    return data


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

def statistical_analysis(data):

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    print(data.describe())

    print("\nCorrelation Matrix:")
    print(data.corr().round(3))


# ============================================================
# 5. VISUALIZATION
# ============================================================

def create_visualizations(data):

    print("\nCreating visualizations...")

    # --------------------------------------------------------
    # Distribution of final scores
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data["Final_Score"],
        bins=30,
        kde=True
    )

    plt.title("Distribution of Final Examination Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "final_score_distribution.png"
        )
    )

    plt.close()

    # --------------------------------------------------------
    # Study hours vs final score
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=data,
        x="Study_Hours",
        y="Final_Score"
    )

    sns.regplot(
        data=data,
        x="Study_Hours",
        y="Final_Score",
        scatter=False
    )

    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Final Score")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "study_hours_vs_score.png"
        )
    )

    plt.close()

    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "correlation_heatmap.png"
        )
    )

    plt.close()

    print("Graphs saved in:", FIGURE_FOLDER)


# ============================================================
# 6. MACHINE LEARNING MODEL
# ============================================================

def train_model(data):

    print("\n" + "=" * 60)
    print("MACHINE LEARNING MODEL")
    print("=" * 60)

    features = [
        "Study_Hours",
        "Attendance",
        "Sleep_Hours",
        "Previous_Score",
        "Assignment_Score"
    ]

    X = data[features]
    y = data["Final_Score"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining observations:", len(X_train))
    print("Testing observations:", len(X_test))

    # Random Forest regression model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest model...")

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Model evaluation
    # --------------------------------------------------------

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nMODEL PERFORMANCE")
    print("-" * 40)

    print(f"MAE  : {mae:.3f}")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")
    print("-" * 40)
    print(importance)

    # --------------------------------------------------------
    # Actual vs predicted plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 7))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6
    )

    plt.plot(
        [0, 100],
        [0, 100],
        linestyle="--"
    )

    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Actual vs Predicted Student Scores")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "actual_vs_predicted.png"
        )
    )

    plt.close()

    return model


# ============================================================
# 7. PREDICT A NEW STUDENT
# ============================================================

def predict_new_student(model):

    print("\n" + "=" * 60)
    print("NEW STUDENT PREDICTION")
    print("=" * 60)

    print("\nEnter the student's information.")

    try:

        study_hours = float(
            input("Study hours per week: ")
        )

        attendance = float(
            input("Attendance percentage: ")
        )

        sleep_hours = float(
            input("Average sleep hours per night: ")
        )

        previous_score = float(
            input("Previous examination score: ")
        )

        assignment_score = float(
            input("Assignment score: ")
        )

        student = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Sleep_Hours": [sleep_hours],
            "Previous_Score": [previous_score],
            "Assignment_Score": [assignment_score]
        })

        prediction = model.predict(student)[0]

        print("\n" + "=" * 50)
        print(f"Predicted Final Score: {prediction:.2f}")
        print("=" * 50)

        if prediction >= 80:
            category = "Excellent"

        elif prediction >= 70:
            category = "Very Good"

        elif prediction >= 60:
            category = "Good"

        elif prediction >= 50:
            category = "Pass"

        else:
            category = "At Risk"

        print("Performance Category:", category)

    except ValueError:

        print(
            "\nInvalid input. Please enter numerical values."
        )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_data(data):

    data.to_csv(
        DATA_FILE,
        index=False
    )

    print(
        f"\nDataset saved as: {DATA_FILE}"
    )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("     STUDENT ACADEMIC PERFORMANCE ANALYTICS SYSTEM")
    print("=" * 65)

    # Generate data
    data = generate_student_data(1000)

    # Clean data
    data = clean_data(data)

    # Save dataset
    save_data(data)

    # Statistical analysis
    statistical_analysis(data)

    # Visualizations
    create_visualizations(data)

    # Machine learning
    model = train_model(data)

    # Predict new student
    predict_new_student(model)

    print("\n" + "=" * 65)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nGenerated files:")
    print("1.", DATA_FILE)
    print("2. figures/final_score_distribution.png")
    print("3. figures/study_hours_vs_score.png")
    print("4. figures/correlation_heatmap.png")
    print("5. figures/actual_vs_predicted.png")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
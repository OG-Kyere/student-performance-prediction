# Student Academic Performance Prediction

This project is a small end-to-end machine-learning exercise built around a simple question: **given a few academic and lifestyle variables, how well can we estimate a student's final examination score?**

The data are generated inside the program rather than collected from real students, so the repository is best viewed as a workflow demonstration rather than a real-world predictive study.

## Variables

The model uses five predictors:

- study hours per week
- class attendance
- average sleep hours
- previous examination score
- assignment score

The target is the student's **Final Score**.

Before fitting a model, the script handles missing values, summarizes the data, checks correlations, and creates exploratory plots.

## Machine learning model

The prediction model is a **Random Forest Regressor** from Scikit-learn.

The dataset is split into training and testing sets:

- 80% for training
- 20% for evaluation

Performance is summarized with:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

The script also reports feature importance so it is easier to see which variables the fitted model relies on most.

## Visualizations

The analysis produces:

- distribution of final examination scores
- study hours versus final score
- correlation heatmap
- actual versus predicted scores
- feature importance

Generated plots are saved in the `figures/` folder.

## Project structure

```text
student-performance-prediction/
│
├── figures/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   ├── final_score_distribution.png
│   └── study_hours_vs_score.png
│
├── student_data.csv
├── student_prediction.py
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/OG-Kyere/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Install the dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run the program

```bash
python student_prediction.py
```

On Windows, if `python` is not recognized, use:

```bash
py student_prediction.py
```

The script generates the dataset, runs the exploratory analysis, trains and evaluates the model, saves the figures, and then asks for information about a new student.

## Making a prediction

For a new student, the program asks for:

```text
Study hours per week
Attendance percentage
Average sleep hours per night
Previous examination score
Assignment score
```

The predicted final score is then placed into one of five categories:

| Predicted Score | Category |
| ---: | --- |
| 80 and above | Excellent |
| 70–79 | Very Good |
| 60–69 | Good |
| 50–59 | Pass |
| Below 50 | At Risk |

## Workflow

```text
Generate Data
      ↓
Clean Missing Values
      ↓
Descriptive Statistics
      ↓
Correlation Analysis
      ↓
Create Visualizations
      ↓
Split Training and Testing Data
      ↓
Train Random Forest Model
      ↓
Evaluate Model
      ↓
Examine Feature Importance
      ↓
Predict New Student Performance
```

## Technologies

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## About the data

The current dataset is synthetic. That keeps the project self-contained and avoids using sensitive student records, but it also means the reported model performance should not be treated as evidence of how the approach would perform in a real school setting.

A useful next step would be to test the same workflow on a real academic-performance dataset and compare several regression models under cross-validation.

## Possible next steps

- compare Random Forest with other regression models
- add cross-validation
- tune hyperparameters
- test a real student-performance dataset
- add classification for grade categories
- build an interactive dashboard
- deploy the predictor as a small web application

## Author

**Kyere Ofosu Gideon**

\# Student Academic Performance Prediction



A Python project that explores what factors are associated with students' academic performance and uses machine learning to estimate a student's final examination score.



The project follows a fairly complete data science workflow: generating and cleaning the data, exploring relationships between variables, visualizing the results, training a prediction model, and testing how well the model performs.



\## What the project does



The system works with five main factors:



\* Study hours per week

\* Class attendance

\* Average sleep hours

\* Previous examination score

\* Assignment score



These variables are used to predict a student's \*\*Final Score\*\*.



Rather than jumping straight into machine learning, the project first looks at the data statistically. Missing values are handled, descriptive statistics are produced, and correlations between the variables are examined before the prediction model is trained.



\## Machine Learning



For the prediction task, I used a \*\*Random Forest Regressor\*\* from Scikit-learn.



The dataset is divided into training and testing sets, with 80% used to train the model and 20% reserved for evaluation.



The model is evaluated using:



\* \*\*MAE\*\* — Mean Absolute Error

\* \*\*MSE\*\* — Mean Squared Error

\* \*\*RMSE\*\* — Root Mean Squared Error

\* \*\*R²\*\* — Coefficient of Determination



Feature importance is also calculated to get a better idea of which variables contribute most to the model's predictions.



\## Visualizations



The project generates a few plots to make the analysis easier to understand:



\* Distribution of final examination scores

\* Study hours versus final score

\* Correlation heatmap

\* Actual versus predicted scores

\* Feature importance



The generated plots are saved in the `figures/` folder.



\## Project structure



```text

student-performance-prediction/

│

├── figures/

│   ├── actual\_vs\_predicted.png

│   ├── correlation\_heatmap.png

│   ├── final\_score\_distribution.png

│   └── study\_hours\_vs\_score.png

│

├── student\_data.csv

├── student\_prediction.py

└── README.md

```



\## Getting started



\### 1. Clone the repository



```bash

git clone https://github.com/OG-Kyere/student-performance-prediction.git

cd student-performance-prediction

```



\### 2. Install the dependencies



```bash

pip install numpy pandas matplotlib seaborn scikit-learn

```



\### 3. Run the program



```bash

python student\_prediction.py

```



On Windows, if `python` is not recognized, you can run:



```bash

py student\_prediction.py

```



Once the program starts, it generates the dataset, performs the analysis, trains the model, creates the visualizations, and asks for information about a new student.



\## Making a prediction



For a new student, the program asks for:



```text

Study hours per week

Attendance percentage

Average sleep hours per night

Previous examination score

Assignment score

```



It then estimates the student's final examination score and places the prediction into one of five categories:



| Predicted Score | Category  |

| --------------: | --------- |

|    80 and above | Excellent |

|           70–79 | Very Good |

|           60–69 | Good      |

|           50–59 | Pass      |

|        Below 50 | At Risk   |



\## The workflow



The project follows this process:



```text

Generate Data

&#x20;     ↓

Clean Missing Values

&#x20;     ↓

Descriptive Statistics

&#x20;     ↓

Correlation Analysis

&#x20;     ↓

Create Visualizations

&#x20;     ↓

Split Training and Testing Data

&#x20;     ↓

Train Random Forest Model

&#x20;     ↓

Evaluate Model

&#x20;     ↓

Examine Feature Importance

&#x20;     ↓

Predict New Student Performance

```



\## Technologies



The project was built with:



\* \*\*Python\*\*

\* \*\*NumPy\*\*

\* \*\*Pandas\*\*

\* \*\*Matplotlib\*\*

\* \*\*Seaborn\*\*

\* \*\*Scikit-learn\*\*



\## A note about the data



The current dataset is generated within the program rather than collected from real students. This makes the project useful for demonstrating the full analytical and machine-learning workflow without relying on a sensitive student dataset.



A natural next step would be to test the approach on a real-world academic performance dataset and compare the results.



\## Possible improvements



There are several directions this project could take from here:



\* Compare Random Forest with other regression models

\* Use cross-validation to assess model stability

\* Tune the model's hyperparameters

\* Work with a real student performance dataset

\* Add classification for grade categories

\* Build an interactive dashboard

\* Deploy the prediction system as a web application



\## About



\*\*Kyere Ofosu Gideon\*\*



If you find the project useful, feel free to ⭐ the repository.




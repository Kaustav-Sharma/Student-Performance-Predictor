# Student Performance Prediction System 🎓

## Overview
This project is a Supervised Machine Learning system designed to predict student academic performance and flag at-risk students for early educational intervention. It processes demographic, behavioral, and academic data to predict final grades using both Classification (Pass/Fail) and Regression (Exact Score) models.

## Real-Life Application
Designed for educational institutions, this system assists school counselors by:
1. Identifying high-risk students before final exams.
2. Extracting **Feature Importance** to understand *why* a student is struggling (e.g., absences, lack of study time, or previous failures).

## Algorithms Used
This system trains, optimizes, and compares three distinct algorithms:
* **Logistic Regression:** Serves as the linear baseline model.
* **Decision Trees:** Captures non-linear behavioral patterns and provides highly interpretable feature importance for counselors.
* **Artificial Neural Networks (ANN):** A Multi-Layer Perceptron utilizing backpropagation to capture complex, multi-variable relationships.

## Key Technical Concepts Demonstrated
* **Hyperparameter Tuning:** Automated `GridSearchCV` to mathematically optimize model depths, solvers, and activation functions.
* **Cross-Validation:** 10-Fold K-Fold Cross Validation to ensure model stability and prevent overfitting.
* **Evaluation Metrics:** Analyzed via Confusion Matrices, ROC/AUC Curves, MAE (Mean Absolute Error), and R² Scores.
* **Data Preprocessing:** Handled via `scikit-learn`'s `StandardScaler` and `LabelEncoder`.

## How to Run
1. Ensure `student-mat.csv` is in the same directory as the script.
2. Install the required dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Run the script: `python model.py`

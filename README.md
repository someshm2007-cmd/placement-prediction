# Placement Prediction Model

This project builds a Binary Classification model using **Logistic Regression** to predict whether a student will get placed based on their **CGPA** and **IQ** score.

## 📌 Project Overview
- **Objective:** Predict student placement outcomes based on academic performance and cognitive score.
- **Dataset:** 100 rows containing student CGPA, IQ, and Placement Status (0 = Not Placed, 1 = Placed).
- **Algorithm:** Logistic Regression (Scikit-Learn).
- **Accuracy:** ~90%

---

## 🛠️ Project Workflow

1. **Data Preprocessing & Cleaning**
   - Removed unnecessary index columns.
   - Checked for missing/null values (`df.info()`).

2. **Exploratory Data Analysis (EDA)**
   - Visualized feature distribution and class separation using `matplotlib` scatter plots.

3. **Feature Selection & Train-Test Split**
   - Features ($X$): `cgpa`, `iq`
   - Target ($y$): `placement`
   - Split ratio: 90% Training data, 10% Testing data.

4. **Feature Scaling**
   - Applied `StandardScaler` to scale features to zero mean and unit variance (`fit_transform` on train data, `transform` on test data).

5. **Model Training & Evaluation**
   - Trained a `LogisticRegression` classifier.
   - Evaluated accuracy using `accuracy_score`.
   - Plotted decision boundaries using `mlxtend.plotting.plot_decision_regions`.

6. **Model Deployment Prep**
   - Exported the trained model using `pickle` (`model.pkl`).

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python installed along with the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend

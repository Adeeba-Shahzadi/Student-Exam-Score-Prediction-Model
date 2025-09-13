# 🎓 Student Exam Score Prediction

This project predicts **students' exam scores** based on various performance factors such as study hours, attendance, previous performance, and tutoring sessions.”
It uses **machine learning regression models** to train and evaluate predictions, with data visualization to understand correlations.  

---

## 🚀 Project Workflow

1. **Data Cleaning**  
   - Removed missing values (NaN) and Drop Outliers.

2. **Exploratory Data Analysis (EDA)**  
   - Scatter plots to visualize relationships between factors and exam score.  
   - Heatmap to check correlation between multiple features.  

3. **Feature Selection**  
   - From the heatmap, the following features showed the strongest correlation with `Exam_Score`:  
     - `Hours_Studied`  
     - `Attendance`  
     - `Previous_Scores`  
     - `Tutoring_Sessions`
   - These features were selected for training the model.  

4. **Model Training**  
   - Used **Linear Regression** for training.  
   - Used **Polynomial Regression (degree = 3)** for training.  
   - Compared predictions with actual scores.  

5. **Evaluation**  
   - **Mean Absolute Error (MAE):** ~1.15  
   - **Mean Squared Error (MSE):** ~2.06  
   - **R² Score:** ~0.82 

6. **Prediction vs Actual**  
   - Compared predicted values against actual exam scores to check performance.  

7. **Linear VS POlynomial Regression**
   -Comparison between Linear and Polynomail Regression

---

## 📊 Dataset Description

The dataset from kaggle used is **StudentPerformanceFactors.csv**, containing features that may influence a student’s exam performance.  

### Columns:
- `Hours_Studied` – Number of study hours per week  
- `Attendance` – Attendance percentage  
- `Parental_Involvement` – Level of parental support (Low/Medium/High)  
- `Access_to_Resources` – Availability of learning resources (Low/Medium/High)  
- `Extracurricular_Activities` – Participation in extracurriculars (Yes/No)  
- `Sleep_Hours` – Average sleep per night  
- `Previous_Scores` – Past exam/test performance  
- `Motivation_Level` – Self-reported motivation (Low/Medium/High)  
- `Internet_Access` – Availability of internet connection (Yes/No)  
- `Tutoring_Sessions` – Extra tutoring hours  
- `Family_Income` – Approximate family income level (Low/Medium/High)  
- `Teacher_Quality` – Quality rating of teachers (Low/Medium/High)  
- `School_Type` – Type of school (Public/Private)  
- `Peer_Influence` – Peer group effect (Positive/Neutral/Negative)  
- `Physical_Activity` – Hours of physical activity per week  
- `Learning_Disabilities` – Presence of learning disabilities (Yes/No)  
- `Parental_Education_Level` – Education level of parents (High School/College/Postgraduate)  
- `Distance_from_Home` – Distance to school (Near/Moderate/Far)  
- `Gender` – Student gender (Male/Female)  
- `Exam_Score` – 🎯 **Target variable** (final exam performance score)  

## 🛠️ Tech Stack

- **Python**  
- **Pandas** – Data cleaning & manipulation  
- **NumPy** – Numerical computations  
- **Matplotlib & Seaborn** – Data visualization  
- **Scikit-learn** – Machine learning models and evaluation  

---

## 📈 Results
- The **Polynomial Regression model** gave an R² score of **0.82**, meaning it explains ~82% of the variance in exam scores.  
- Predictions were close to actual exam scores with low error values.  

---

## 📦 Installation  

1. Clone the repository (or download the project folder):  
   ```bash
   git clone <https://github.com/Adeeba-Shahzadi/Student-Exam-Score-Prediction-Model>
   pip install -r requirements.txt
   python student_score_prediction.py
```
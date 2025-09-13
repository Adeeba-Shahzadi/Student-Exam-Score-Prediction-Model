import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats

# Load and clean data
Data = pd.read_csv('StudentPerformanceFactors.csv')
No_NUll_Data = Data.dropna(axis=0)
All_Factors = ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores','Tutoring_Sessions','Physical_Activity','Exam_Score']

# Remove Outliers by z-score
z_score = stats.zscore(No_NUll_Data[All_Factors])
Clean_Data = No_NUll_Data[abs(z_score<3).all(axis=1)]

# Correlation heatmap
Check_Correlation = Clean_Data[All_Factors]
plt.figure(figsize=(13,6))
sb.heatmap(Check_Correlation.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plots of features vs Exam Score
Exam_Score = Clean_Data['Exam_Score']
Training_Factors = ['Hours_Studied','Attendance','Previous_Scores','Tutoring_Sessions']
Input_Factors = Clean_Data[Training_Factors]

for col in Training_Factors:
    plt.figure(figsize=(6,6))
    plt.scatter(Input_Factors[col], Exam_Score, color='blue')
    plt.xlabel(col)
    plt.ylabel('Exam Score')
    plt.title(f'{col} VS Exam Score')
    plt.show()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(Input_Factors, Exam_Score, test_size=0.2, random_state=1)

#Linear Regression
model = LinearRegression()
model.fit(x_train,y_train)

#Evaluation
y_pred = model.predict(x_test)
print('Linear Regression')
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Polynomial regression
poly = PolynomialFeatures(degree=3)
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)

model = LinearRegression()
model.fit(x_poly_train, y_train)
y_poly_pred = model.predict(x_poly_test)

# Display first 10 rows with Predicted Values
result = x_test.copy() 
result['Predicted_Exam_Score'] = y_poly_pred
result['Actual_Exam_Score'] = y_test.values
print(result.head(10))

# Evaluation
print('Polynomial Regression')
print("MAE:", mean_absolute_error(y_test, y_poly_pred))
print("MSE:", mean_squared_error(y_test, y_poly_pred))
print("R²:", r2_score(y_test, y_poly_pred))

# Predicted vs Actual plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_poly_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title('Predicted vs Actual (Polynomial)')
plt.show()


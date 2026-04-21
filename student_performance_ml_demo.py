import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Generation
np.random.seed(42)
n = 1000
study_hours = np.random.normal(5, 1.5, n)
attendance_percentage = np.clip(np.random.normal(85, 10, n), 50, 100)
sleep_hours = np.clip(np.random.normal(7, 1, n), 4, 10)
social_media_hours = np.clip(np.random.normal(3, 1, n), 0, 6)
noise = np.random.normal(0, 3, n)
exam_score = (
    (study_hours / 5) * 20 +
    (attendance_percentage / 100) * 30 +
    (sleep_hours / 7) * 15 +
    (1 - social_media_hours / 6) * 20 +
    noise
)

print("Sample data:")
print(f"Study Hours: {study_hours[:5]}")
print(f"Attendance Percentage: {attendance_percentage[:5]}")
print(f"Sleep Hours: {sleep_hours[:5]}")
print(f"Social Media Hours: {social_media_hours[:5]}")
print(f"Exam Score: {exam_score[:5]}")

# Step 2: Data Processing
df = pd.DataFrame({
    'Study Hours': study_hours,
    'Attendance Percentage': attendance_percentage,
    'Sleep Hours': sleep_hours,
    'Social Media Hours': social_media_hours,
    'Exam Score': exam_score
})

nan_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[nan_indices, 'Sleep Hours'] = np.nan

df['Sleep Hours'] = df['Sleep Hours'].fillna(df['Sleep Hours'].median())

df['Study_to_Sleep_Ratio'] = df['Study Hours'] / df['Sleep Hours']

print("DataFrame shape:", df.shape)
print("Null counts:\n", df.isnull().sum())
print("First 5 rows:\n", df.head())

# Step 3: Exploratory Data Analysis
plt.figure(figsize=(8, 6))
plt.scatter(df['Social Media Hours'], df['Exam Score'], alpha=0.5)
plt.xlabel('Social Media Hours')
plt.ylabel('Exam Score')
plt.title('Scatter Plot: Social Media vs Exam Score')
plt.grid(True)
plt.savefig('student_scatter_plot.png')
plt.close()

plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.savefig('student_correlation_heatmap.png')
plt.close()

print("Plots saved as 'student_scatter_plot.png' and 'student_correlation_heatmap.png'")

# Step 4: Machine Learning Pipeline
X = df.drop('Exam Score', axis=1)
y = df['Exam Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R²: {r2:.4f}')
print(f'MSE: {mse:.4f}')

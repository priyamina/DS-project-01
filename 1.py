# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load the Dataset
url = 'https://raw.githubusercontent.com/yourusername/yourrepo/main/telco_customer_churn.csv'
data = pd.read_csv(url)

# Step 3: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Drop unnecessary columns
data.drop(columns=['customerID'], inplace=True)

# Convert categorical variables to numerical
data = pd.get_dummies(data, drop_first=True)

# Step 4: Split the Data
X = data.drop('Churn_Yes', axis=1)  # Features
y = data['Churn_Yes']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model Building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

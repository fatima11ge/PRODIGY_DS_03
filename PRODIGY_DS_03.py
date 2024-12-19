# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE


# Reload the dataset with the correct delimiter
bank_data = pd.read_csv(r"C:\Users\fatem\Desktop\Prodigy\Task 03\bank+marketing\bank\bank-full.csv", sep=";")

# Display the first few rows to verify correct parsing
print(bank_data.head())


# Display column-wise missing values
missing_values = bank_data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Check for duplicates
duplicate_rows = bank_data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}\n")

# Display a sample of the data
print("\nSample Data:")
display(bank_data.head())


# Encode categorical variables
label_encoders = {}
for column in bank_data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    bank_data[column] = label_encoders[column].fit_transform(bank_data[column])

print(label_encoders)


# Resample using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Split the resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


# Hyperparameter tuning for Decision Tree
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)


# Best model
best_tree = grid_search.best_estimator_


# Make predictions
y_pred = best_tree.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_tree.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances)

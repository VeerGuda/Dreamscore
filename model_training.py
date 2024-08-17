# Veer Guda - Dreamscore
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib  # Import joblib directly

# Load data
df = pd.read_excel('dataset_building.xlsx')

# Rename columns to remove spaces (optional for ease of reference)
df.rename(columns={
    'credit health': 'credit_health',
    'financial knowledge': 'financial_knowledge'
}, inplace=True)

# Preprocess text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Goal']).toarray()

# Normalize importance scores
y = df[['budgeting', 'saving', 'investing', 'credit_health', 'financial_knowledge']].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
forest_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
forest_model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(forest_model, 'forest_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

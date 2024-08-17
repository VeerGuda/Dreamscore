# Veer Guda - Dreamscore
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor



#Data 
user_data = [
    {"goal": "I want to buy a house in the near future", "budgeting": 8, "saving": 9, "investing": 6, "credit_health": 7, "financial_knowledge": 5},
    {"goal": "I want to retire early", "budgeting": 6, "saving": 7, "investing": 8, "credit_health": 5, "financial_knowledge": 6},
    {"goal": "I need to pay off my student loans", "budgeting": 7, "saving": 6, "investing": 5, "credit_health": 8, "financial_knowledge": 7},
]

# Preprocess text data
goals = [entry["goal"] for entry in user_data]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(goals).toarray()

# Normalize importance scores
y = np.array([[entry["budgeting"], entry["saving"], entry["investing"], entry["credit_health"], entry["financial_knowledge"]] for entry in user_data])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
forest_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
forest_model.fit(X_train, y_train)

# Evaluate the model
y_pred = forest_model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2, axis=0)
print(f"Mean Squared Error per output: {mse}")

# Prediction function
def predict_importance_scores(goal):
    goal_vector = vectorizer.transform([goal]).toarray()
    prediction = forest_model.predict(goal_vector)
    return prediction

# Example usage
user_goal = "I want to travel the world"
predicted_scores = predict_importance_scores(user_goal)
print("Predicted Importance Scores for the goal:", predicted_scores)
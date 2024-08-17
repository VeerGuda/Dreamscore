# Veer Guda - Dreamscore
import joblib

# Load the model and vectorizer
forest_model = joblib.load('forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Prediction function
def predict_importance_scores(goal):
    goal_vector = vectorizer.transform([goal]).toarray()
    prediction = forest_model.predict(goal_vector)
    return prediction

# Example usage
# user_goal = "I want to learn about investing"
# predicted_scores = predict_importance_scores(user_goal)
# print("Predicted Importance Scores for the goal:", predicted_scores[0])


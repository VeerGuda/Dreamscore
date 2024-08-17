# Veer Guda - Dreamscore
#  DreamScore - A Financial Habits metric
# Your DreamScore is calculated based on 5 metrics - Budgeting, Saving, Investing, Credit Health, and Financial Knowledge/Behavior

from model_running import predict_importance_scores as predict

# User enters goal statement
goal_statement = input("Enter an \"I want\" statement: \n")

# Get the importance scores for the five categories
scores = predict(goal_statement)[0]

print("\nImportance scores from the model:\n")
print ("\nBudgeting: ", scores[0])
print ("\nSaving: ", scores[1])
print ("\nInvesting: ", scores[2])
print ("\nCredit Health: ", scores[3])
print ("\nFinancial Literacy: ", scores[4])


# Function to calculate the subcategory scores
def calculate_subcategory_scores(data):
    scores = {}
    
    # Budgeting
    scores['Budgeting'] = {
        'Income vs. Expenses Ratio': min((data['income'] - data['expenses']) / data['income'] * 10, 10),
        'Expense Tracking Consistency': (data['tracked_months'] / data['total_months']) * 10,
        'Adherence to Budget': (data['within_budget_months'] / data['total_months']) * 10
    }
    
    # Saving
    scores['Saving'] = {
        'Emergency Fund Adequacy': min(data['emergency_fund'] / (data['monthly_expenses'] * 6), 1) * 10,
        'Savings Rate': (data['savings_per_month'] / data['income_per_month']) * 10
    }
    
    # Investing
    scores['Investing'] = {
        'Diversity of Portfolio': (data['asset_classes'] / data['target_asset_classes']) * 10,
        'Amount in Retirement Accounts': min(data['retirement_accounts'] / data['target_retirement_amount'], 1) * 10,
        'Investment Growth Rate': min((data['current_investment_value'] - data['initial_investment_value']) / data['initial_investment_value'], data['target_growth_rate']) * 10
    }
    
    # Credit Health
    scores['Credit Health'] = {
        'Credit Score': (data['credit_score'] / 850) * 10,
        'Credit Utilization Rate': min(data['credit_used'] / data['credit_available'], 1) * 10,
        'Debt-to-Income Ratio': min(data['monthly_debt_payments'] / data['monthly_income'], 1) * 10
    }

    # Financial Knowledge & Behavior
    scores['Financial Knowledge & Behavior'] = {
        'Financial Education': (data['completed_education_activities'] / data['target_education_activities']) * 10,
        'Financial Goal Setting': (data['goals_set'] / data['target_goals']) * 10,
        'Financial Habits': (data['positive_financial_habits'] / data['target_habits']) * 10
    }

    return scores

# Function to normalize weights - makes sure everything is consistent
def normalize_weights(weights):
    total = sum(weights.values())
    return {key: (value / total) * 100 for key, value in weights.items()}

# Function to adjust weights based on user rankings with a 10% variation
def adjust_weights_based_on_ranks(user_ranks, base_weight=20, variability=10):
    min_weight = base_weight - variability
    max_weight = base_weight + variability
    
    # Normalize the user ranks to sum to 1
    total_ranks = sum(user_ranks.values())
    normalized_ranks = {key: value / total_ranks for key, value in user_ranks.items()}
    
    # Calculate weights based on normalized ranks
    adjusted_weights = {}
    for key, normalized_rank in normalized_ranks.items():
        adjusted_weights[key] = min_weight + (normalized_rank * (max_weight - min_weight))
    
    # Ensure the weights sum up to 100
    total_adjusted = sum(adjusted_weights.values())
    return {key: (value / total_adjusted) * 100 for key, value in adjusted_weights.items()}

# User ranks each category from 1 to 10 based on the model output
user_ranks = {
    "Budgeting": scores[0],
    "Saving": scores[1],
    "Investing": scores[2],
    "Credit Health": scores[3],
    "Financial Knowledge & Behavior": scores[4]
}

# Adjust weights based on user rankings
adjusted_weights = adjust_weights_based_on_ranks(user_ranks)

# Sample user data - could either come from personas or from linked bank account data
data = {
    'income': 5000,
    'expenses': 3000,
    'tracked_months': 10,
    'total_months': 12,
    'within_budget_months': 8,
    'emergency_fund': 15000,
    'monthly_expenses': 3000,
    'savings_per_month': 500,
    'income_per_month': 5000,
    'asset_classes': 4,
    'target_asset_classes': 5,
    'retirement_accounts': 20000,
    'target_retirement_amount': 50000,
    'current_investment_value': 12000,
    'initial_investment_value': 10000,
    'target_growth_rate': 0.1,
    'credit_score': 750,
    'credit_used': 2000,
    'credit_available': 10000,
    'monthly_debt_payments': 500,
    'monthly_income': 5000,
    'completed_education_activities': 8,
    'target_education_activities': 10,
    'goals_set': 4,
    'target_goals': 5,
    'positive_financial_habits': 6,
    'target_habits': 7
}

# Calculate the subcategory scores
subcategory_scores = calculate_subcategory_scores(data)

print("Based on the following data, here is the Final dreamscore breakdown:\n")
print(data)
# Print the subcategory scores
for category, scores in subcategory_scores.items():
    print(f"\n{category}:")
    for subcategory, score in scores.items():
        print(f"  {subcategory}: {score:.2f}")

# Calculate total scores for each metric
metric_scores = {category: sum(scores.values()) / len(scores) for category, scores in subcategory_scores.items()}

# Normalize scores to 100
normalized_scores = {metric: (score / 10) * 100 for metric, score in metric_scores.items()}

# Calculate DreamScore
dreamscore = sum(normalized_scores[metric] * (adjusted_weights[metric] / 100) for metric in user_ranks)
print(f"\nFinal DreamScore: {dreamscore:.2f}")

# Output the scores for each metric and the final DreamScore
for metric, score in normalized_scores.items():
    print(f"{metric} Score: {score:.2f}")

print("\nFinal DreamScore: {}".format(round(dreamscore)))

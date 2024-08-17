# Veer Guda - Dreamscore
import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

# Load and clean datasets
student_spending = load_and_clean_data(r"...\student_spending.csv")
financial_behavior = load_and_clean_data(r"...\Student spending habits dataset.csv_exported.csv")

# Combine datasets
combined_data = pd.concat([student_spending, financial_behavior], axis=0)

# Create a 'text' column that combines relevant columns into a single text field
combined_data['text'] = combined_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Save combined data
combined_data.to_csv(r"...\combined_data.csv", index=False)

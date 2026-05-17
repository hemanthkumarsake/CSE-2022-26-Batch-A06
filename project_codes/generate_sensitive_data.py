import pandas as pd
import numpy as np

n = 100000 
np.random.seed(42)

# Generate wide-ranging random data
salary = np.random.randint(20000, 200000, n)
rent = np.random.randint(5000, 40000, n)
food = np.random.randint(2000, 25000, n)
entertainment = np.random.randint(500, 15000, n)
utilities = np.random.randint(500, 10000, n)
transportation = np.random.randint(500, 15000, n)
insurance = np.random.randint(0, 5000, n)
savings = np.random.randint(0, 20000, n)
subscriptions = np.random.randint(0, 5000, n)
travels = np.random.randint(0, 20000, n)
emi = np.random.randint(0, 30000, n)

# The Target MUST be sensitive to the sum of these inputs
# We add a tiny bit of random noise (5%) so it stays "AI" and not just a calculator
actual_sum = (rent + food + entertainment + utilities + transportation + 
              insurance + subscriptions + travels + emi)
future_expense_target = actual_sum * np.random.uniform(0.95, 1.05, n)

df = pd.DataFrame({
    'salary': salary, 'rent': rent, 'food': food, 'entertainment': entertainment,
    'utilities': utilities, 'transportation': transportation, 'insurance': insurance,
    'savings': savings, 'subscriptions': subscriptions, 'travels': travels, 'emi': emi,
    'future_expense_target': future_expense_target
})

df.to_csv('finance_data.csv', index=False)
print("✅ Sensitive training data generated.")
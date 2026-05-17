import pandas as pd
import numpy as np
import os

n = 100000 
np.random.seed(42)

# Generate wide-ranging random data
salary = np.random.randint(20000, 200000, n)
rent = np.random.randint(2000, 50000, n)
food = np.random.randint(1000, 30000, n)
entertainment = np.random.randint(0, 20000, n)
utilities = np.random.randint(500, 15000, n)
transportation = np.random.randint(200, 20000, n)
insurance = np.random.randint(0, 10000, n)
savings = np.random.randint(0, 50000, n)
subscriptions = np.random.randint(0, 5000, n)
travels = np.random.randint(0, 30000, n)
emi = np.random.randint(0, 40000, n)

# Logic: Target is exactly the sum of expenses plus a tiny random buffer (1-5%)
# This makes the AI "sensitive" to every single category change
current_total = (rent + food + entertainment + utilities + transportation + 
                 insurance + savings + subscriptions + travels + emi)

future_expense_target = current_total * np.random.uniform(1.01, 1.05, n)

df = pd.DataFrame({
    'salary': salary, 'rent': rent, 'food': food, 'entertainment': entertainment,
    'utilities': utilities, 'transportation': transportation, 'insurance': insurance,
    'savings': savings, 'subscriptions': subscriptions, 'travels': travels, 'emi': emi,
    'future_expense_target': future_expense_target
})

df.to_csv('finance_data.csv', index=False)
print("✅ High-impact data generated.")
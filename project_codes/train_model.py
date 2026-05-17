import pandas as pd
import xgboost as xgb
import joblib
import os

# 1. Load data
df = pd.read_csv('finance_data.csv')

# 2. Define features in the EXACT order used in views.py
features = ['salary', 'rent', 'food', 'entertainment', 'utilities', 
            'transportation', 'insurance', 'savings', 'subscriptions', 
            'travels', 'emi']

X = df[features]
y = df['future_expense_target']

# 3. Train with "Max Sensitivity" settings
model = xgb.XGBRegressor(
    n_estimators=1000,    # Increased estimators for better fit
    learning_rate=0.1,
    max_depth=12,         # Deeper trees to capture specific rupee changes
    reg_lambda=0,         # NO regularization (crucial)
    reg_alpha=0,          # NO regularization (crucial)
    n_jobs=-1,            # Use all CPU cores
    objective='reg:squarederror'
)

model.fit(X, y)

# 4. Save
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgboost_model.pkl')
print("✅ High-sensitivity model saved to models/xgboost_model.pkl")
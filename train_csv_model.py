import pandas as pd
import xgboost as xgb
import joblib
import os

# 1. Load your 100,000 rows
df = pd.read_csv('finance_data.csv')

# 2. EXACT order from your views.py
features = ['salary', 'rent', 'food', 'entertainment', 'utilities', 
            'transportation', 'insurance', 'savings', 'subscriptions', 
            'travels', 'emi']

X = df[features]
y = df['future_expense_target']

# 3. Train with "Deterministic" settings (Fixed seed)
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42, # Forces the AI to be consistent
    objective='reg:squarederror'
)

model.fit(X, y)

# 4. Save to the folder Django looks at
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgboost_model.pkl')
print("✅ AI Brain synced with 11-feature logic.")
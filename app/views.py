from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from .models import *
from .models import GoalTransaction
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Q, Count
from .models import SavingsGoal, GoalTransaction
from datetime import datetime, timedelta
from django.utils import timezone
from calendar import monthrange
import calendar
from decimal import Decimal
import json
from django.http import JsonResponse
from collections import defaultdict
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
import random
import pytz
import string
from django.core.mail import send_mail
from django.conf import settings
import joblib
import pandas as pd
import numpy as np

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db.models.functions import TruncDate
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import requests
from .utils import *

GEMINI_API_KEY = 'your_gemini_api_key_here'

# Create your views here.

MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'xgboost_model.pkl')
gradient_boosting_model = joblib.load(MODEL_PATH)
import os
import joblib
import numpy as np
import datetime  # Essential for the date fix
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render 
from django.contrib.auth.decorators import login_required
from django.db.models import Sum

# CORRECT MODELS FROM YOUR models.py
from app.models import Addexpenses, monthly_salary

# Load the trained ML model once when the server starts
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'xgboost_model.pkl')
gradient_boosting_model = joblib.load(MODEL_PATH)

@login_required
def predict_expenses(request):
    if request.method == 'POST':
        # 1. Get all expenses for this user
        all_user_expenses = Addexpenses.objects.filter(user=request.user)

        # 2. The 11 slots the AI was trained to understand (STRICT ORDER)
        ai_categories = [
            'salary', 'rent', 'food', 'entertainment', 'utilities', 
            'transportation', 'insurance', 'savings', 'subscriptions', 
            'travels', 'emi'
        ]

        # Initialize totals for each slot
        data_values = {cat: 0.0 for cat in ai_categories}

        # 3. UPDATED RELEVANCE MAPPING LOGIC
        total_actual_spending = 0
        for exp in all_user_expenses:
            amt = float(exp.spending_amount) if exp.spending_amount else 0.0
            total_actual_spending += amt
            name = exp.category_name.lower().strip() if exp.category_name else ""

            # Check for Food relevance
            if any(word in name for word in ['food', 'juice', 'snack', 'grocery', 'drink', 'eat', 'dinner', 'lunch']):
                data_values['food'] += amt
            
            # Check for Rent/Stay relevance
            elif any(word in name for word in ['rent', 'room', 'hostel', 'house', 'stay', 'pg']):
                data_values['rent'] += amt

            # Check for Utilities relevance
            elif any(word in name for word in ['utility', 'water', 'electric', 'bill', 'gas', 'power', 'light']):
                data_values['utilities'] += amt

            # Check for Transportation
            elif any(word in name for word in ['transport', 'petrol', 'fuel', 'uber', 'ola', 'bus', 'bike', 'car', 'auto']):
                data_values['transportation'] += amt

            # Check for Subscriptions
            elif any(word in name for word in ['sub', 'mobile', 'recharge', 'netflix', 'wifi', 'internet', 'prime']):
                data_values['subscriptions'] += amt

            # Check for EMI/Loan
            elif any(word in name for word in ['emi', 'loan', 'installment']):
                data_values['emi'] += amt
            
            # NEW CATEGORY MAPPING: Catch 'Dance', 'Gym', 'Class', etc.
            elif any(word in name for word in ['dance', 'class', 'gym', 'hobby', 'entertainment', 'movie']):
                data_values['entertainment'] += amt

            # CATCH-ALL: Ensure any unique name still contributes to the total prediction
            else:
                data_values['entertainment'] += amt

        # 4. Fetch Salary
        salary_record = monthly_salary.objects.filter(user=request.user).first()
        current_salary = float(salary_record.salary) if salary_record else 40000.0
        data_values['salary'] = current_salary

        # 5. Prepare the input array
        final_input = [data_values[cat] for cat in ai_categories]
        input_array = np.array([final_input])

        # 6. Generate Prediction with Sensitivity Fix
        prediction = gradient_boosting_model.predict(input_array)
        predicted_expense = float(prediction[0])

        # SENSITIVITY LOGIC: If user spends massive amounts (Outliers), 
        # force the AI to adjust rather than staying "stuck" at the training limit.
        if total_actual_spending > predicted_expense:
            # Adjust prediction by 20% of the difference to show the model is "learning"
            adjustment = (total_actual_spending - predicted_expense) * 0.20
            predicted_expense += adjustment

        predicted_savings = current_salary - predicted_expense

        return JsonResponse({
            'predicted_expense': round(predicted_expense, 2),
            'predicted_savings': round(predicted_savings, 2)
        })

    # GET request logic
    now = datetime.datetime.now()
    return render(request, 'predict_expenses.html', {
        'current_month': now.strftime('%b'),
        'current_year': now.year
    })

@login_required
def savings_goals(request):
    today = timezone.localtime(timezone.now()).date()

    # 1. Handle Form Submissions
    if request.method == 'POST':
        
        # Action A: Create a New Goal
        if 'create_goal' in request.POST:
            goal_name = request.POST.get('goal_name')
            target_amount = request.POST.get('target_amount')
            deadline = request.POST.get('deadline')
            
            SavingsGoal.objects.create(
                user=request.user,
                goal_name=goal_name,
                target_amount=target_amount,
                deadline=deadline
            )
            messages.success(request, "New savings goal created!")
            return redirect('savings_goals')

        # Action B: Add Money to a Goal (Wallet Transaction)
        elif 'add_money' in request.POST:
            goal_id = request.POST.get('goal_id')
            amount = request.POST.get('amount')
            description = request.POST.get('description', 'Manual Deposit')
            
            goal = get_object_or_404(SavingsGoal, id=goal_id, user=request.user)
            
            GoalTransaction.objects.create(
                goal=goal,
                amount=amount,
                description=description
            )
            messages.success(request, f"₹{amount} added to {goal.goal_name}!")
            return redirect('savings_goals')

    # 2. Fetch and Calculate Data for UI
    goals = SavingsGoal.objects.filter(user=request.user).order_by('-created_at')
    goals_data = []

    for goal in goals:
        # Sum all transactions for this specific goal
        saved_amount = goal.transactions.aggregate(Sum('amount'))['amount__sum'] or 0.0
        saved_amount = float(saved_amount)
        target = float(goal.target_amount)

        # Progress Math
        progress = (saved_amount / target) * 100 if target > 0 else 0
        progress = min(100, max(0, progress)) # Cap at 100% for the UI bar
        
        remaining = target - saved_amount
        if remaining < 0:
            remaining = 0.0

        # Time Math
        days_left = (goal.deadline - today).days
        daily_required = remaining / days_left if days_left > 0 else 0.0

        # Status Logic
        if saved_amount >= target:
            status = "Achieved"
            color = "success"
        elif days_left <= 0 and remaining > 0:
            status = "Overdue"
            color = "danger"
        else:
            status = "In Progress"
            color = "primary"

        goals_data.append({
            'obj': goal,
            'saved_amount': round(saved_amount, 2),
            'progress': round(progress, 1),
            'remaining': round(remaining, 2),
            'daily_required': round(daily_required, 2),
            'days_left': max(0, days_left),
            'status': status,
            'color': color
        })

    return render(request, 'savings_goals.html', {'goals': goals_data})

def register(request):
    # User.objects.all().delete()
    # monthly_salary.objects.all().delete()
    # Finanace_Category.objects.all().delete()
    # Addexpenses.objects.all().delete()
    # BudgetGoalModel.objects.all().delete()
    # subscriptionModel.objects.all().delete()
    # AddFamilyMember.objects.all().delete()
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        emp_salary = request.POST.get('employee_salary')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists!')
            return redirect('register')
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists!')
            return redirect('user_login')

        user = User.objects.create_user(
            username=username,
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password
        )

        monthly_salary.objects.create(
            user = user,
            salary = emp_salary
        )

        messages.success(request, 'User register successfully')
        return redirect('user_login')
    return render(request, 'register.html')

def user_login(request):
    if request.method == 'POST':
        email_username = request.POST.get('email_username')
        password = request.POST.get('password')
        user = authenticate(request, username=email_username, password=password)
        
        if user is not None:
            login(request, user)
            
            try:
                # Check if the user is added by another person
                added_by_other = AddFamilyMember.objects.get(Added_person=request.user)
            except AddFamilyMember.DoesNotExist:
                added_by_other = None

            if added_by_other:
                request.session['added_by_other'] = True  # Store in session if added by other
            else:
                # Make sure to clear the session if user is not added by other
                if 'added_by_other' in request.session:
                    del request.session['added_by_other']

            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid Credentials!')
            return redirect('user_login')
    return render(request, 'login.html')

# views.py (Updated logic segment for dashboard view)

from django.shortcuts import render
from django.db.models import Sum
from django.utils import timezone
from datetime import timedelta
from django.http import JsonResponse
import json

# Make sure to import your models here, including GoalTransaction
# from .models import Addexpenses, monthly_salary, BudgetGoalModel, subscriptionModel, GoalTransaction 

from django.shortcuts import render
from django.db.models import Sum
from django.utils import timezone
from datetime import timedelta
from django.http import JsonResponse
import json
from .models import Addexpenses, monthly_salary, BudgetGoalModel, subscriptionModel

import json
from django.shortcuts import render, redirect
from django.db.models import Sum
from django.utils import timezone
from django.http import JsonResponse
from datetime import timedelta
from .models import Addexpenses, Income, BudgetGoalModel, subscriptionModel # Ensure Income is imported
import json
from django.shortcuts import render
from django.db.models import Sum
from django.utils import timezone
from django.http import JsonResponse
from datetime import timedelta
from .models import Addexpenses, Income, BudgetGoalModel, subscriptionModel, monthly_salary

def dashboard(request):
    # --- Time Setup ---
    now_local = timezone.localtime(timezone.now())
    today = now_local.date()
    current_month_start = today.replace(day=1)

    # --- AJAX Request for Charts ---
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        days = int(request.GET.get('range', 7))
        ajax_labels = []
        ajax_data = []
        for i in range(days - 1, -1, -1):
            date = today - timedelta(days=i)
            daily_total = Addexpenses.objects.filter(
                user=request.user, 
                time_stamp__date=date
            ).aggregate(total=Sum('spending_amount'))['total'] or 0
            ajax_labels.append(date.strftime('%d %b'))
            ajax_data.append(float(daily_total))
        return JsonResponse({'labels': ajax_labels, 'data': ajax_data})

    # --- 1. INCOME CALCULATION ---
    try:
        sal_record = monthly_salary.objects.get(user=request.user)
        fixed_salary = float(sal_record.salary or 0)
    except (monthly_salary.DoesNotExist, TypeError, ValueError):
        fixed_salary = 0.0

    dynamic_income_total = Income.objects.filter(
        user=request.user,
        date__month=today.month,
        date__year=today.year
    ).aggregate(total=Sum('amount'))['total'] or 0

    estimated_income = fixed_salary + float(dynamic_income_total)

    # --- 2. EXPENSE CALCULATIONS ---
    monthly_expenses_qs = Addexpenses.objects.filter(
        user=request.user,
        time_stamp__date__gte=current_month_start,
        time_stamp__date__lte=today
    )
    total_expenses = monthly_expenses_qs.aggregate(total=Sum('spending_amount'))['total'] or 0
    expense_count = monthly_expenses_qs.count()
    recent_expenses = Addexpenses.objects.filter(user=request.user).order_by('-time_stamp')[:5]
    
    # --- 3. SAVINGS CALCULATION ---
    try:
        total_savings = GoalTransaction.objects.filter(goal__user=request.user).aggregate(total=Sum('amount'))['total'] or 0
    except:
        total_savings = 0
    monthly_savings = float(total_savings)
    
    available_balance = float(estimated_income) - float(total_expenses) - monthly_savings
    available_balance_msg = "✅ Positive balance." if available_balance >= 0 else "⚠️ Overspending after saving."
    
    # --- 4. BUDGETING & GOALS ---
    budget_goals = BudgetGoalModel.objects.filter(user=request.user, month__year=today.year, month__month=today.month)
    total_budget = budget_goals.aggregate(total=Sum('planned_amount'))['total'] or 0
    budget_utilization = (float(total_expenses) / float(total_budget) * 100) if total_budget > 0 else 0
    active_goals_count = BudgetGoalModel.objects.filter(user=request.user, end_of_month__gte=today).count()
    
    # --- 5. SUBSCRIPTIONS ---
    active_subscriptions = subscriptionModel.objects.filter(user=request.user, is_active=True)
    subscription_cost = 0
    upcoming_renewals = []
    for sub in active_subscriptions:
        subscription_cost += float(sub.price) if sub.plan_type == 'monthly' else float(sub.price) / 12
        days_until = 30 # Simplified for logic flow
        upcoming_renewals.append({'name': sub.name, 'amount': sub.price, 'days': days_until})

    # --- 6. TRENDS ---
    prev_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
    prev_expenses = Addexpenses.objects.filter(user=request.user, time_stamp__date__range=[prev_month_start, current_month_start - timedelta(days=1)]).aggregate(total=Sum('spending_amount'))['total'] or 0
    expense_trend = ((float(total_expenses) - float(prev_expenses)) / float(prev_expenses) * 100) if prev_expenses > 0 else 0

    # --- 7. FIXED FINANCIAL INSIGHTS (REPLACE THIS PART) ---
    clean_total_expenses = float(total_expenses or 0)
    if clean_total_expenses > 0:
        estimated_next_month = clean_total_expenses * 1.05  # 5% Trend Increase
    else:
        estimated_next_month = 0.0

    clean_estimated_income = float(estimated_income or 0)
    estimated_balance = clean_estimated_income - estimated_next_month

    if estimated_balance < 0:
        balance_status_msg = "⚠️ Your estimated expenses exceed your income."
        balance_status_color = "danger"
    else:
        balance_status_msg = "✅ You are likely to maintain a positive balance."
        balance_status_color = "success"

    # --- 8. CONTEXT ---
    category_data_qs = monthly_expenses_qs.values('category_name').annotate(total=Sum('spending_amount')).order_by('-total')
    category_labels = [item['category_name'] or 'Other' for item in category_data_qs[:5]]
    category_values = [float(item['total']) for item in category_data_qs[:5]]

    daily_labels = []
    daily_expenses = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        day_total = Addexpenses.objects.filter(user=request.user, time_stamp__date=d).aggregate(total=Sum('spending_amount'))['total'] or 0
        daily_expenses.append(float(day_total))
        daily_labels.append(d.strftime('%a'))

    context = {
        'monthly_income': round(float(estimated_income), 2),
        'monthly_expenses': round(float(total_expenses), 2),
        'monthly_savings': round(monthly_savings, 2),
        'available_balance': round(available_balance, 2),
        'balance_message': available_balance_msg,
        'budget_utilization': round(budget_utilization, 1),
        'expense_trend': round(expense_trend, 1),
        'expense_trend_direction': 'up' if expense_trend > 0 else 'down',
        'recent_transactions': [{'title': exp.Buyed_Items or 'Expense', 'category': exp.category_name, 'amount': exp.spending_amount, 'date': timezone.localtime(exp.time_stamp).strftime('%d %b'), 'type': 'expense'} for exp in recent_expenses],
        'category_labels': json.dumps(category_labels),
        'category_data': json.dumps(category_values),
        'daily_labels': json.dumps(daily_labels),
        'daily_expenses': json.dumps(daily_expenses),
        'active_goals': active_goals_count,
        'subscription_cost': round(subscription_cost, 2),
        'upcoming_renewals': upcoming_renewals[:3],
        'username': request.user.username,
        # Forecasting variables
        'estimated_next_month': estimated_next_month,
        'estimated_balance': estimated_balance,
        'balance_status_msg': balance_status_msg,
        'balance_status_color': balance_status_color,
    }
    return render(request, 'dashboard.html', context)
def get_icon_for_category(category_name):
    """Helper function to determine icon based on category"""
    if not category_name:
        return 'bi-receipt'
    
    category_lower = category_name.lower()
    
    if 'food' in category_lower or 'dining' in category_lower or 'restaurant' in category_lower:
        return 'bi-cup-hot'
    elif 'shop' in category_lower or 'amazon' in category_lower or 'flipkart' in category_lower:
        return 'bi-bag'
    elif 'transport' in category_lower or 'fuel' in category_lower or 'uber' in category_lower or 'ola' in category_lower:
        return 'bi-fuel-pump'
    elif 'entertainment' in category_lower or 'netflix' in category_lower or 'movie' in category_lower or 'spotify' in category_lower:
        return 'bi-film'
    elif 'bill' in category_lower or 'electricity' in category_lower or 'water' in category_lower or 'internet' in category_lower:
        return 'bi-lightning'
    elif 'health' in category_lower or 'medical' in category_lower or 'doctor' in category_lower:
        return 'bi-heart-pulse'
    elif 'education' in category_lower or 'course' in category_lower or 'book' in category_lower:
        return 'bi-book'
    elif 'salary' in category_lower or 'income' in category_lower:
        return 'bi-bank'
    else:
        return 'bi-receipt'

@login_required()
def add_category(request):
    get_categories = Finanace_Category.objects.filter(user=request.user)
    if request.method == 'POST':
        category_name = request.POST.get('category')
        description = request.POST.get('description')

        if Finanace_Category.objects.filter(category__icontains=category_name, user=request.user).exists():
            messages.error(request, 'Category name already exists!')
            return redirect('add_category')
        else:
            Finanace_Category.objects.create(
                user = request.user,
                category = category_name,
                description = description
            )

            messages.success(request, 'New category added successfully')
            return redirect('add_category')
        
    return render(request, 'add_category.html', {'get_categories':get_categories})

def remove_category(request, id):
    get_cat = Finanace_Category.objects.get(id=id)
    get_cat.delete()
    messages.success(request, 'Category removed successfully!')
    return redirect('add_category')

def user_logout(request):
    request.session.flush()
    logout(request)
    return redirect('index')

def add_expenses(request):
    # Get current month and year for accurate filtering
    now = datetime.datetime.now()
    current_month = now.month
    current_year = now.year
    
    # Filter expenses for the specific user in the current month/year
    monthly_expenses_query = Addexpenses.objects.filter(
        user=request.user, 
        time_stamp__month=current_month,
        time_stamp__year=current_year
    )
    
    # 1. Calculate Total Amount spent this month
    total_spending = monthly_expenses_query.aggregate(Sum('spending_amount'))
    total_amount = total_spending['spending_amount__sum'] or 0
    
    # 2. FIX: Calculate the COUNT of expenses (This was missing)
    expense_count = monthly_expenses_query.count()
    
    user_categories = Finanace_Category.objects.filter(user=request.user)
    recent_expenses = Addexpenses.objects.filter(user=request.user).order_by('-time_stamp')
    
    if request.method == 'POST':
        category_name = request.POST.get('category_name')
        spending_amount = request.POST.get('spending_amount')
        Buyed_Items = request.POST.get('Buyed_Items')
        bill = request.FILES.get('bill')
        
        # --- ANOMALY DETECTION LOGIC ---
        past_expenses = Addexpenses.objects.filter(user=request.user, category_name=category_name)
        past_count = past_expenses.count()
        anomaly_flag = False

        if past_count > 0:
            avg_expense = float(past_expenses.aggregate(Sum('spending_amount'))['spending_amount__sum'] or 0) / past_count
            if float(spending_amount) > (avg_expense * 2):
                anomaly_flag = True
        # --- END LOGIC ---
        
        Addexpenses.objects.create(
            user=request.user,
            category_name=category_name,
            spending_amount=spending_amount,
            Buyed_Items=Buyed_Items,
            bill=bill,
            is_anomaly=anomaly_flag
        )
        
        if anomaly_flag:
            messages.warning(request, f"⚠️ Anomaly Detected: This {category_name} expense is unusually high.")
        else:
            messages.success(request, 'Bill details added successfully')
            
        return redirect('add_expenses')

    # Pass 'expense_count' to the template so the dashboard updates correctly
    return render(request, 'add_expenses.html', {
        'user_categories': user_categories, 
        'recent_expenses': recent_expenses, 
        'total_amount': total_amount,
        'expense_count': expense_count  # <-- ADD THIS LINE
    })
@login_required
def budget_goals(request):
    # Get current month and year
    today = timezone.now().date()
    current_month = today.replace(day=1)
    user = request.user
    
    # Handle form submission for new budget goal
    if request.method == 'POST':
        category_id = request.POST.get('category')
        month_year = request.POST.get('month')
        planned_amount = request.POST.get('planned_amount')
        
        if month_year and planned_amount:
            # Parse the month input (format: YYYY-MM)
            month_date = datetime.datetime.strptime(month_year, '%Y-%m').date()
            
            # Calculate end of month
            last_day = monthrange(month_date.year, month_date.month)[1]
            end_of_month = month_date.replace(day=last_day)
            
            # Check if goal already exists for this category and month
            existing_goal = BudgetGoalModel.objects.filter(
                user=request.user,
                category_id=category_id if category_id else None,
                month=month_date
            ).first()
            
            if existing_goal:
                messages.warning(request, 'A budget goal already exists for this category and month!')
            else:
                # Create new budget goal
                goal = BudgetGoalModel.objects.create(
                    user=request.user,
                    category_id=category_id if category_id else None,
                    month=month_date,
                    end_of_month=end_of_month,
                    planned_amount=planned_amount
                )
                messages.success(request, 'Budget goal created successfully!')
        
        return redirect('budget_goals')
    
    # Get all budget goals for the user
    goals = BudgetGoalModel.objects.filter(user=request.user).order_by('-month', 'category')
    
    # Get all categories for the user
    categories = Finanace_Category.objects.filter(user=request.user)
    
    # Calculate progress for each goal
    goals_with_progress = []
    for goal in goals:
        # Get expenses for this category and month
        expenses_query = Addexpenses.objects.filter(
            user=request.user,
            time_stamp__date__gte=goal.month,
            time_stamp__date__lte=goal.end_of_month
        )
        
        if goal.category:
            categ = Finanace_Category.objects.get(id=goal.category.id)
            expenses_query = expenses_query.filter(category_name=goal.category.category)
        
        total_spent = expenses_query.aggregate(total=Sum('spending_amount'))['total'] or 0
        
        # Calculate progress percentage
        if goal.planned_amount > 0:
            progress = (total_spent / float(goal.planned_amount)) * 100
        else:
            progress = 0
        
        # Determine status
        if total_spent > float(goal.planned_amount):
            status = 'exceeded'
            status_color = 'danger'
            email_subject = 'Budget goal exceeded'
            category_name = goal.category.category if goal.category else "Overall"
            email_message = f'Hello {user.username},\n\nFinAI Alert: Your {category_name} budget goal has been exceeded.'
            send_mail(email_subject, email_message, settings.EMAIL_HOST_USER, [user.email])
        elif progress >= 80:
            status = 'warning'
            status_color = 'warning'
        else:
            status = 'on_track'
            status_color = 'success'
        
        remaining = float(goal.planned_amount) - total_spent
        
        goals_with_progress.append({
            'goal': goal,
            'total_spent': total_spent,
            'progress': round(progress, 1),
            'status': status,
            'status_color': status_color,
            'remaining': remaining,
            'category_name': goal.category.category if goal.category else 'Overall'
        })
    
    # Get available months for selection (current and next 5 months)
    available_months = []
    for i in range(6):
        month_date = today.replace(day=1) + timedelta(days=30*i)
        available_months.append({
            'value': month_date.strftime('%Y-%m'),
            'label': month_date.strftime('%B %Y')
        })
    
    # Get summary statistics
    active_goals = goals.filter(end_of_month__gte=today).count()
    completed_goals = goals.filter(end_of_month__lt=today).count()
    total_budget = goals.filter(month=current_month).aggregate(total=Sum('planned_amount'))['total'] or 0
    
    context = {
        'goals_with_progress': goals_with_progress,
        'categories': categories,
        'available_months': available_months,
        'current_month': current_month.strftime('%B %Y'),
        'active_goals': active_goals,
        'completed_goals': completed_goals,
        'total_budget': total_budget,
    }
    
    return render(request, 'budgets.html', context)

@login_required
def delete_budget_goal(request, goal_id):
    if request.method == 'POST':
        try:
            goal = BudgetGoalModel.objects.get(id=goal_id, user=request.user)
            goal.delete()
            messages.success(request, 'Budget goal deleted successfully!')
        except BudgetGoalModel.DoesNotExist:
            messages.error(request, 'Budget goal not found!')
    
    return redirect('budget_goals')

@login_required
def edit_budget_goal(request, goal_id):
    if request.method == 'POST':
        try:
            goal = BudgetGoalModel.objects.get(id=goal_id, user=request.user)
            new_amount = request.POST.get('planned_amount')
            
            if new_amount:
                goal.planned_amount = new_amount
                goal.save()
                messages.success(request, 'Budget goal updated successfully!')
            else:
                messages.error(request, 'Please provide a valid amount!')
                
        except BudgetGoalModel.DoesNotExist:
            messages.error(request, 'Budget goal not found!')
    
    return redirect('budget_goals')

@login_required
def expenses_report(request):
    # Set the local timezone for India
    ist_tz = pytz.timezone('Asia/Kolkata')
    today = timezone.localtime(timezone.now()).date()
    
    filter_type = request.GET.get('filter', 'month')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    category_filter = request.GET.get('category', '')
    
    if start_date and end_date:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except:
            start_date = today.replace(day=1)
            end_date = today
    else:
        if filter_type == 'month':
            start_date = today.replace(day=1)
            end_date = today
        elif filter_type == 'quarter':
            quarter_month = ((today.month - 1) // 3) * 3 + 1
            start_date = today.replace(month=quarter_month, day=1)
            end_date = today
        elif filter_type == 'year':
            start_date = today.replace(month=1, day=1)
            end_date = today
        else:  # week
            start_date = today - timedelta(days=today.weekday())
            end_date = today
    
    expenses = Addexpenses.objects.filter(
        user=request.user,
        time_stamp__date__range=[start_date, end_date]
    ).order_by('-time_stamp')
    
    if category_filter:
        expenses = expenses.filter(category_name=category_filter)
    
    categories = Finanace_Category.objects.filter(user=request.user).values_list('category', flat=True).distinct()
    
    total_expenses = expenses.count()
    total_amount = expenses.aggregate(total=Sum('spending_amount'))['total'] or 0
    avg_expense = total_amount / total_expenses if total_expenses > 0 else 0
    
    highest_expense = expenses.order_by('-spending_amount').first()
    lowest_expense = expenses.filter(spending_amount__gt=0).order_by('spending_amount').first()
    
    category_data = expenses.values('category_name').annotate(
        total=Sum('spending_amount'),
        count=Count('id')
    ).order_by('-total')
    
    categories_chart_labels = []
    categories_chart_data = []
    categories_colors = []
    
    color_palette = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD',
        '#D4A5A5', '#9B59B6', '#3498DB', '#E67E22', '#2ECC71',
        '#F1C40F', '#E74C3C', '#1ABC9C', '#34495E', '#7F8C8D'
    ]
    
    for i, item in enumerate(category_data):
        categories_chart_labels.append(item['category_name'] or 'Uncategorized')
        categories_chart_data.append(float(item['total']))
        categories_colors.append(color_palette[i % len(color_palette)])
    
    date_range = (end_date - start_date).days + 1
    daily_data = defaultdict(float)
    
    for exp in expenses:
        local_time = exp.time_stamp.astimezone(ist_tz)
        date_str = local_time.date().strftime('%Y-%m-%d')
        daily_data[date_str] += float(exp.spending_amount)
    
    daily_trend_labels = []
    daily_trend_data = []
    for i in range(date_range):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime('%Y-%m-%d')
        daily_trend_labels.append(current_date.strftime('%d %b'))
        daily_trend_data.append(float(daily_data.get(date_str, 0)))
    
    monthly_data = defaultdict(float)
    year_expenses = Addexpenses.objects.filter(
        user=request.user,
        time_stamp__year=today.year
    )
    
    for exp in year_expenses:
        local_time = exp.time_stamp.astimezone(ist_tz)
        month_str = local_time.strftime('%B')
        monthly_data[month_str] += float(exp.spending_amount)
    
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    monthly_labels = []
    monthly_data_values = []
    for month in months_order:
        if month in monthly_data:
            monthly_labels.append(month[:3])
            monthly_data_values.append(monthly_data[month])
    
    budget_goals = BudgetGoalModel.objects.filter(
        user=request.user,
        month__year=today.year
    ).select_related('category')
    
    budget_labels = []
    budget_planned = []
    budget_actual = []
    
    for goal in budget_goals:
        category_name = goal.category.category if goal.category else 'Overall'
        month_name = goal.month.strftime('%b')
        budget_labels.append(f"{category_name} ({month_name})")
        budget_planned.append(float(goal.planned_amount))
        
        actual = Addexpenses.objects.filter(
            user=request.user,
            category_name=category_name if category_name != 'Overall' else None,
            time_stamp__date__gte=goal.month,
            time_stamp__date__lte=goal.end_of_month
        ).aggregate(total=Sum('spending_amount'))['total'] or 0
        budget_actual.append(float(actual))
    
    top_items = expenses.values('Buyed_Items').annotate(
        total=Sum('spending_amount'),
        count=Count('id')
    ).filter(~Q(Buyed_Items__isnull=True) & ~Q(Buyed_Items='')
    ).order_by('-total')[:10]
    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_data = defaultdict(float)
    
    for exp in expenses:
        local_time = exp.time_stamp.astimezone(ist_tz)
        weekday = local_time.weekday()
        weekday_data[weekday] += float(exp.spending_amount)
    
    weekday_labels = weekdays
    weekday_values = [float(weekday_data.get(i, 0)) for i in range(7)]
    
    context = {
        'expenses': expenses[:20],
        'total_expenses': total_expenses,
        'total_amount': total_amount,
        'avg_expense': avg_expense,
        'highest_expense': highest_expense,
        'lowest_expense': lowest_expense,
        'categories': categories,
        'selected_category': category_filter,
        'filter_type': filter_type,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'date_range_text': f"{start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}",
        'categories_chart_labels': json.dumps(categories_chart_labels),
        'categories_chart_data': json.dumps(categories_chart_data),
        'categories_colors': json.dumps(categories_colors),
        'daily_trend_labels': json.dumps(daily_trend_labels),
        'daily_trend_data': json.dumps(daily_trend_data),
        'monthly_labels': json.dumps(monthly_labels),
        'monthly_data': json.dumps(monthly_data_values),
        'budget_labels': json.dumps(budget_labels),
        'budget_planned': json.dumps(budget_planned),
        'budget_actual': json.dumps(budget_actual),
        'weekday_labels': json.dumps(weekday_labels),
        'weekday_values': json.dumps(weekday_values),
        'top_items': top_items,
        'category_data': category_data,
        'year': today.year,
        'month': today.strftime('%B'),
    }
    
    return render(request, 'expenses_report.html', context)

@login_required
def expense_details(request, expense_id):
    try:
        expense = Addexpenses.objects.get(id=expense_id, user=request.user)
        data = {
            'id': expense.id,
            'category_name': expense.category_name,
            'spending_amount': float(expense.spending_amount),
            'time_stamp': expense.time_stamp.strftime('%d %b %Y, %I:%M %p'),
            'Buyed_Items': expense.Buyed_Items,
            'bill_url': expense.bill.url if expense.bill else None,
        }
        return JsonResponse(data)
    except Addexpenses.DoesNotExist:
        return JsonResponse({'error': 'Expense not found'}, status=404)

@login_required
def subscription_list(request):
    # Get filter parameters
    plan_filter = request.GET.get('plan', '')
    status_filter = request.GET.get('status', '')
    
    # Get all subscriptions for the logged-in user
    subscriptions = subscriptionModel.objects.filter(user=request.user).order_by('-created_at')
    
    # Apply filters
    if plan_filter:
        subscriptions = subscriptions.filter(plan_type=plan_filter)
    if status_filter:
        is_active = status_filter == 'active'
        subscriptions = subscriptions.filter(is_active=is_active)
    
    # Calculate statistics
    total_subscriptions = subscriptions.count()
    active_subscriptions = subscriptions.filter(is_active=True).count()
    monthly_total = sum([float(sub.price) for sub in subscriptions if sub.plan_type == 'monthly' and sub.is_active])
    yearly_total = sum([float(sub.price) for sub in subscriptions if sub.plan_type == 'yearly' and sub.is_active])
    
    # Calculate monthly equivalent cost
    total_monthly_cost = monthly_total
    for sub in subscriptions.filter(plan_type='yearly', is_active=True):
        if sub.price:
            total_monthly_cost += float(sub.price) / 12
    
    context = {
        'subscriptions': subscriptions,
        'total_subscriptions': total_subscriptions,
        'active_subscriptions': active_subscriptions,
        'monthly_total': monthly_total,
        'yearly_total': yearly_total,
        'total_monthly_cost': round(total_monthly_cost, 2),
        'plan_filter': plan_filter,
        'status_filter': status_filter,
    }
    
    return render(request, 'subscription_list.html', context)


# ============== ADD NEW SUBSCRIPTION ==============
@login_required
def add_subscription(request):
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            plan_type = request.POST.get('plan_type') # 'monthly' or 'yearly'
            price = request.POST.get('price')
            is_active = request.POST.get('is_active') == 'on'
            logo = request.FILES.get('logo')
            bill_slip = request.FILES.get('bill_slip')
            
            if not name or not price:
                messages.error(request, 'Name and price are required fields!')
                return redirect('add_subscription')
            
            # 1. Save the Subscription Record
            subscription = subscriptionModel.objects.create(
                user=request.user,
                name=name,
                plan_type=plan_type,
                price=price,
                is_active=is_active,
                logo=logo,
                bill_slip=bill_slip
            )
            
            # 2. Automatically record this as an Expense if active
            if is_active:
                # Calculate monthly value
                raw_price = float(price)
                monthly_value = raw_price if plan_type == 'monthly' else (raw_price / 12)
                
                Addexpenses.objects.create(
                    user=request.user,
                    category_name="Subscription",
                    spending_amount=round(monthly_value, 2),
                    Buyed_Items=f"Recurring: {name} ({plan_type.capitalize()} Plan)",
                    is_anomaly=False # Subscriptions are planned, usually not anomalies
                )
            
            messages.success(request, f'Subscription "{name}" added and recorded in expenses!')
            return redirect('subscription_list')
            
        except Exception as e:
            messages.error(request, f'Error adding subscription: {str(e)}')
            return redirect('add_subscription')
    
    return render(request, 'add_subscription.html', {'today': timezone.now().strftime('%Y-%m-%d')})

# ============== EDIT SUBSCRIPTION ==============
@login_required
def edit_subscription(request, sub_id):
    subscription = get_object_or_404(subscriptionModel, id=sub_id, user=request.user)
    
    if request.method == 'POST':
        try:
            # Update basic info
            subscription.name = request.POST.get('name')
            subscription.plan_type = request.POST.get('plan_type')
            subscription.price = request.POST.get('price')
            subscription.is_active = request.POST.get('is_active') == 'on'
            
            # Update files if new ones are provided
            if request.FILES.get('logo'):
                # Delete old logo file if it exists
                if subscription.logo:
                    if os.path.isfile(subscription.logo.path):
                        os.remove(subscription.logo.path)
                subscription.logo = request.FILES.get('logo')
            
            if request.FILES.get('bill_slip'):
                # Delete old bill slip if it exists
                if subscription.bill_slip:
                    if os.path.isfile(subscription.bill_slip.path):
                        os.remove(subscription.bill_slip.path)
                subscription.bill_slip = request.FILES.get('bill_slip')
            
            subscription.save()
            
            messages.success(request, f'Subscription "{subscription.name}" updated successfully!')
            return redirect('subscription_list')
            
        except Exception as e:
            messages.error(request, f'Error updating subscription: {str(e)}')
    
    # GET request - show form with existing data
    context = {
        'subscription': subscription,
    }
    
    return render(request, 'edit_subscription.html', context)


# ============== DELETE SUBSCRIPTION ==============
@login_required
def delete_subscription(request, sub_id):
    subscription = get_object_or_404(subscriptionModel, id=sub_id, user=request.user)
    
    if request.method == 'POST':
        try:
            # Delete associated files
            if subscription.logo:
                if os.path.isfile(subscription.logo.path):
                    os.remove(subscription.logo.path)
            if subscription.bill_slip:
                if os.path.isfile(subscription.bill_slip.path):
                    os.remove(subscription.bill_slip.path)
            
            subscription_name = subscription.name
            subscription.delete()
            
            messages.success(request, f'Subscription "{subscription_name}" deleted successfully!')
            
        except Exception as e:
            messages.error(request, f'Error deleting subscription: {str(e)}')
    
    return redirect('subscription_list')


# ============== VIEW SUBSCRIPTION DETAILS ==============
@login_required
def subscription_detail(request, sub_id):
    subscription = get_object_or_404(subscriptionModel, id=sub_id, user=request.user)
    
    # Calculate additional info
    created_date = subscription.created_at.date()
    days_since_created = (timezone.now().date() - created_date).days
    
    # Calculate next billing date (simplified - assuming monthly renews every 30 days, yearly every 365)
    if subscription.plan_type == 'monthly':
        next_billing = created_date + timedelta(days=30)
        # Add multiple months based on how many cycles have passed
        cycles_passed = days_since_created // 30
        next_billing = created_date + timedelta(days=30 * (cycles_passed + 1))
    else:  # yearly
        next_billing = created_date + timedelta(days=365)
        cycles_passed = days_since_created // 365
        next_billing = created_date + timedelta(days=365 * (cycles_passed + 1))
    
    # Calculate days until next billing
    days_until = (next_billing - timezone.now().date()).days
    
    # Calculate monthly equivalent cost
    monthly_cost = float(subscription.price) if subscription.plan_type == 'monthly' else float(subscription.price) / 12
    
    context = {
        'subscription': subscription,
        'created_date': created_date,
        'days_since_created': days_since_created,
        'next_billing': next_billing,
        'days_until': days_until,
        'monthly_cost': round(monthly_cost, 2),
        'yearly_cost': round(float(subscription.price) * 12 if subscription.plan_type == 'monthly' else float(subscription.price), 2),
    }
    
    return render(request, 'subscription_detail.html', context)


# ============== TOGGLE SUBSCRIPTION STATUS ==============
@login_required
def toggle_subscription_status(request, sub_id):
    if request.method == 'POST':
        subscription = get_object_or_404(subscriptionModel, id=sub_id, user=request.user)
        subscription.is_active = not subscription.is_active
        subscription.save()
        
        status = "activated" if subscription.is_active else "deactivated"
        messages.success(request, f'Subscription "{subscription.name}" {status} successfully!')
    
    return redirect('subscription_list')

@login_required
def profile_view(request):
    user = request.user
    today = timezone.now().date()
    
    # Get user statistics
    total_expenses = Addexpenses.objects.filter(user=user).count()
    total_spent = Addexpenses.objects.filter(user=user).aggregate(total=Sum('spending_amount'))['total'] or 0
    
    # Get current month expenses
    current_month_start = today.replace(day=1)
    current_month_expenses = Addexpenses.objects.filter(
        user=user,
        time_stamp__date__gte=current_month_start
    ).aggregate(total=Sum('spending_amount'))['total'] or 0
    
    # Get budget goals count
    active_budgets = BudgetGoalModel.objects.filter(
        user=user,
        end_of_month__gte=today
    ).count()
    
    # Get active subscriptions
    active_subscriptions = subscriptionModel.objects.filter(
        user=user,
        is_active=True
    ).count()
    
    # Get categories count
    categories_count = Finanace_Category.objects.filter(user=user).count()
    
    # Get recent activity
    recent_expenses = Addexpenses.objects.filter(user=user).order_by('-time_stamp')[:5]
    
    # Get membership duration
    membership_days = (today - user.date_joined.date()).days
    membership_years = membership_days // 365
    membership_months = (membership_days % 365) // 30
    
    if membership_years > 0:
        membership_duration = f"{membership_years} year{'s' if membership_years > 1 else ''}"
        if membership_months > 0:
            membership_duration += f" {membership_months} month{'s' if membership_months > 1 else ''}"
    elif membership_months > 0:
        membership_duration = f"{membership_months} month{'s' if membership_months > 1 else ''}"
    else:
        membership_duration = f"{membership_days} day{'s' if membership_days > 1 else ''}"
    
    # Determine user tier based on activity
    if active_subscriptions > 2 and total_expenses > 50:
        user_tier = "Premium"
        tier_color = "warning"
        tier_icon = "bi-star-fill"
    elif active_subscriptions > 0 or total_expenses > 20:
        user_tier = "Pro"
        tier_color = "info"
        tier_icon = "bi-star-half"
    else:
        user_tier = "Free"
        tier_color = "secondary"
        tier_icon = "bi-star"
    
    # Calculate profile completion percentage
    completion = 0
    if user.first_name and user.last_name:
        completion += 25
    if user.email:
        completion += 25
    if categories_count > 0:
        completion += 25
    if total_expenses > 0:
        completion += 25
    
    context = {
        'user': user,
        'total_expenses': total_expenses,
        'total_spent': total_spent,
        'current_month_expenses': current_month_expenses,
        'active_budgets': active_budgets,
        'active_subscriptions': active_subscriptions,
        'categories_count': categories_count,
        'recent_expenses': recent_expenses,
        'membership_duration': membership_duration,
        'membership_days': membership_days,
        'user_tier': user_tier,
        'tier_color': tier_color,
        'tier_icon': tier_icon,
        'profile_completion': completion,
        'date_joined': user.date_joined,
        'last_login': user.last_login,
    }
    
    return render(request, 'profile.html', context)


from .models import UserProfile

@login_required
def edit_profile(request):
    user = request.user

    if request.method == 'POST':
        user.first_name = request.POST.get('first_name')
        user.last_name = request.POST.get('last_name')
        user.email = request.POST.get('email')

        phone = request.POST.get('phone')

        user.save()

        # ✅ SAVE PHONE HERE
        profile, created = UserProfile.objects.get_or_create(user=user)
        profile.phone = phone
        profile.save()

        messages.success(request, 'Profile updated successfully!')
        return redirect('profile_view')

    return redirect('profile_view')
@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Keep user logged in
            messages.success(request, 'Password changed successfully!')
            return redirect('profile_view')
        else:
            for error in form.errors.values():
                messages.error(request, error)
    return redirect('profile_view')


@login_required
def delete_account(request):
    if request.method == 'POST':
        confirm = request.POST.get('confirm_delete')
        if confirm == 'DELETE':
            user = request.user
            # Logout user
            from django.contrib.auth import logout
            logout(request)
            # Delete account
            user.delete()
            messages.success(request, 'Your account has been permanently deleted.')
            return redirect('home')  # Redirect to home page
        else:
            messages.error(request, 'Please type DELETE to confirm account deletion.')
    return redirect('profile_view')

@login_required
def add_family_member(request):
    if request.method == 'POST':
        # Get form data
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        family_code = request.POST.get('family_code')
        relation = request.POST.get('relation')
        member_salary = request.POST.get('member_salary')
        
        # Validate required fields
        if not all([first_name, last_name, username, email, password, family_code, relation]):
            messages.error(request, 'All fields are required!')
            return redirect('add_family_member')
        
        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists! Please choose another.')
            return redirect('add_family_member')
        
        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists! Please use another email.')
            return redirect('add_family_member')
        
        try:
            # Create User in built-in User model
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
            )
            
            # Create AddFamilyMember record
            family_member = AddFamilyMember.objects.create(
                Family_code=family_code,
                Added_by=request.user,
                relation=relation,
                Added_person = user
            )
            
            monthly_salary.objects.create(
                user = user,
                salary = member_salary
            )

            email_subject = 'Welcome to FinAI'
            email_message = f'Hello {username},\n\nWelcome To Our Website!\n\nYour are added on FinAI\n\nHere are your Key details:\nUsername: {username}\nPassword: {password}\nFamily-Code: {family_code}\n\nPlease keep this information safe.\n\nBest regards,\nYour Website Team'
            send_mail(email_subject, email_message, settings.EMAIL_HOST_USER, [email])
            messages.success(request, f'Family member {first_name} {last_name} added successfully!')
            return redirect('family_members_list')
            
        except Exception as e:
            messages.error(request, f'Error adding family member: {str(e)}')
            return redirect('add_family_member')
    
    # GET request - show form
    # Generate a random family code if not exists (you can modify this logic)
    family_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    context = {
        'family_code': family_code,
        'relations': ['Spouse', 'Child', 'Parent', 'Sibling', 'Grandparent', 'Other']
    }
    return render(request, 'add_family_member.html', context)


@login_required
def family_members_list(request):
    """View to list all family members added by the user"""
    family_members = AddFamilyMember.objects.filter(Added_by=request.user).order_by('-time_stamp')
    
    # Get corresponding User objects
    members_data = []
    for member in family_members:
        # You'll need to store username in AddFamilyMember or find a way to link
        # For now, we'll show basic info
        print('[[[[[[]]]]]]', member.Added_person.id)
        members_data.append({
            'f_id': member.Added_person.id,
            'family_code': member.Family_code,
            'relation': member.relation,
            'added_on': member.time_stamp,
            # You might want to add more fields here
        })
    
    context = {
        'family_members': members_data,
        'count': len(members_data)
    }
    return render(request, 'family_members_list.html', context)


@login_required
def delete_family_member(request, member_id):
    try:
        member = User.objects.get(id=member_id)
        member.delete()
        messages.success(request, 'Family member removed successfully!')
    except User.DoesNotExist:
        messages.error(request, 'Family member not found!')
        
    return redirect('family_members_list')

def family_expenses(request):
    """
    View to display expenses of all family members added by the current user
    """
    # Check if the current user was added by someone else
    added_by_other = AddFamilyMember.objects.filter(Added_person=request.user).exists()
    
    if added_by_other:
        messages.error(request, 'You do not have permission to view family expenses.')
        return redirect('dashboard')
    
    # Get all family members added by the current user
    family_members = AddFamilyMember.objects.filter(Added_by=request.user).select_related('Added_person')
    
    # Get all user IDs in the family (including the main user)
    family_user_ids = [request.user.id]
    for member in family_members:
        if member.Added_person:
            family_user_ids.append(member.Added_person.id)
    
    # Get filter parameters
    time_range = request.GET.get('time_range', '7')  # Default to 7 days
    category_filter = request.GET.get('category', '')
    member_filter = request.GET.get('member', '')
    
    # Base queryset for expenses
    expenses = Addexpenses.objects.filter(
        user_id__in=family_user_ids,
        time_stamp__isnull=False
    ).select_related('user')
    
    # Apply time range filter
    today = timezone.now().date()
    if time_range == '7':
        start_date = today - timedelta(days=7)
    elif time_range == '30':
        start_date = today - timedelta(days=30)
    elif time_range == '90':
        start_date = today - timedelta(days=90)
    elif time_range == 'month':
        start_date = today.replace(day=1)
    elif time_range == 'year':
        start_date = today.replace(month=1, day=1)
    else:
        start_date = today - timedelta(days=int(time_range))
    
    expenses = expenses.filter(time_stamp__date__gte=start_date)
    
    # Apply category filter
    if category_filter:
        expenses = expenses.filter(category_name=category_filter)
    
    # Apply member filter
    if member_filter and member_filter != 'all':
        expenses = expenses.filter(user_id=member_filter)
    
    # Order by latest first
    expenses = expenses.order_by('-time_stamp')
    
    # Calculate statistics
    total_expenses = expenses.aggregate(total=Sum('spending_amount'))['total'] or 0
    
    # Get category-wise breakdown
    category_breakdown = expenses.values('category_name').annotate(
        total=Sum('spending_amount')
    ).order_by('-total')
    
    # Get member-wise breakdown
    member_breakdown = expenses.values('user__username', 'user__id').annotate(
        total=Sum('spending_amount'),
        expense_count=models.Count('id')
    ).order_by('-total')
    
    # Get daily expenses for chart
    daily_expenses = expenses.annotate(
        date=TruncDate('time_stamp')
    ).values('date').annotate(
        total=Sum('spending_amount')
    ).order_by('date')
    
    # Prepare data for charts
    daily_labels = [item['date'].strftime('%d %b') for item in daily_expenses]
    daily_data = [float(item['total']) for item in daily_expenses]
    
    category_labels = [item['category_name'] for item in category_breakdown]
    category_data = [float(item['total']) for item in category_breakdown]
    
    member_names = [item['user__username'] for item in member_breakdown]
    member_totals = [float(item['total']) for item in member_breakdown]
    
    # Get all categories for filter
    all_categories = Addexpenses.objects.filter(
    user_id__in=family_user_ids
).values_list('category_name', flat=True).distinct()
    
    # Get family members list for filter
    family_list = [{'id': request.user.id, 'name': request.user.username, 'relation': 'Self'}]
    for member in family_members:
        if member.Added_person:
            family_list.append({
                'id': member.Added_person.id,
                'name': member.Added_person.username,
                'relation': member.relation or 'Family Member'
            })
    
    # Calculate percentage of total for each member
    for member in member_breakdown:
        member['percentage'] = (member['total'] / total_expenses * 100) if total_expenses > 0 else 0
    
    context = {
        'expenses': expenses,
        'total_expenses': total_expenses,
        'category_breakdown': category_breakdown,
        'member_breakdown': member_breakdown,
        'daily_labels': daily_labels,
        'daily_data': daily_data,
        'category_labels': category_labels,
        'category_data': category_data,
        'member_names': member_names,
        'member_totals': member_totals,
        'all_categories': all_categories,
        'family_list': family_list,
        'current_time_range': time_range,
        'current_category': category_filter,
        'current_member': member_filter,
        'family_members_count': len(family_list),
        'start_date': start_date,
        'family_members': family_members,
    }
    
    return render(request, 'family_expenses.html', context)

def chatbot(request):
    return render(request, 'chatbot.html')

@csrf_exempt
def get_chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            
            if message:
                # Get response from the finance chatbot
                response = finance_chatbot(message)
                return JsonResponse({'response': response, 'status': 'success'})
            else:
                return JsonResponse({'response': 'Please enter a message.', 'status': 'error'})
                
        except Exception as e:
            print(f"Error in chatbot view: {e}")
            return JsonResponse({'response': 'An error occurred. Please try again.', 'status': 'error'})
    
    return JsonResponse({'response': 'Invalid request method.', 'status': 'error'})



from django.shortcuts import get_object_or_404, redirect

@login_required
def delete_expense(request, pk):
    expense = get_object_or_404(Addexpenses, pk=pk, user=request.user)
    expense.delete()
    return redirect('add_expenses') # Change 'expenses' to your actual expense page URL name
# views.py (Inside your view that handles depositing money into a savings goal)

from django.shortcuts import redirect
from django.contrib import messages
from django.db.models import Sum
from django.utils import timezone
from .models import Addexpenses, monthly_salary, GoalTransaction

def add_to_savings_goal(request, goal_id):
    if request.method == 'POST':
        deposit_amount = float(request.POST.get('amount', 0))
        
        # 1. Calculate Available Balance
        now_local = timezone.localtime(timezone.now())
        current_month_start = now_local.date().replace(day=1)
        
        try:
            total_income = float(monthly_salary.objects.get(user=request.user).salary)
        except monthly_salary.DoesNotExist:
            total_income = 0.0
            
        total_expenses = float(Addexpenses.objects.filter(
            user=request.user,
            time_stamp__date__gte=current_month_start,
            time_stamp__date__lte=now_local.date()
        ).aggregate(total=Sum('spending_amount'))['total'] or 0)
        
        # FIX: Using goal__user instead of user
        total_savings = float(GoalTransaction.objects.filter(
            goal__user=request.user
        ).aggregate(total=Sum('amount'))['total'] or 0)
        
        available_balance = total_income - total_expenses - total_savings
        
        # 2 & 4. Validation Check
        if deposit_amount > available_balance:
            messages.error(request, f"⚠️ Insufficient balance to save this amount. Your available balance is ₹{available_balance:.2f}")
            return redirect(request.META.get('HTTP_REFERER', 'savings_goals'))
            
        # 5. Optional Low Balance Warning (Alerts if balance drops below 10% of income)
        if (available_balance - deposit_amount) < (0.10 * total_income):
            messages.warning(request, "⚠️ Deposit successful, but your remaining balance is running very low.")
        else:
            messages.success(request, "✅ Money successfully added to your savings goal.")
            
        # 3. Allow Saving (Execute your save logic here)
        goal = get_object_or_404(SavingsGoal, id=goal_id, user=request.user)
        GoalTransaction.objects.create(goal=goal, amount=deposit_amount, description=request.POST.get('description', 'Manual Deposit'))
        
        return redirect('savings_goals')
    from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import SavingsGoal, GoalTransaction
from django.utils import timezone
import datetime

def savings_goals(request):
    if request.method == "POST":
        if "create_goal" in request.POST:
            goal_name = request.POST.get("goal_name")
            target_amount = request.POST.get("target_amount")
            deadline = request.POST.get("deadline")
            SavingsGoal.objects.create(
                user=request.user,
                goal_name=goal_name,
                target_amount=target_amount,
                deadline=deadline
            )
            messages.success(request, "Goal created successfully!")
            return redirect('savings_goals')

        elif "add_money" in request.POST:
            goal_id = request.POST.get("goal_id")
            amount = float(request.POST.get("amount"))
            description = request.POST.get("description", "")
            goal = get_object_or_404(SavingsGoal, id=goal_id, user=request.user)
            GoalTransaction.objects.create(goal=goal, amount=amount, description=description)
            messages.success(request, f"Added ₹{amount} to {goal.goal_name}!")
            return redirect('savings_goals')

    # Logic to calculate goal data for display
    raw_goals = SavingsGoal.objects.filter(user=request.user)
    goals_data = []
    
    for g in raw_goals:
        # Summing deposits to get saved_amount
        saved_amount = sum(d.amount for d in g.transactions.all())
        remaining = max(0, float(g.target_amount) - float(saved_amount))
        progress = min(100, (float(saved_amount) / float(g.target_amount)) * 100) if g.target_amount > 0 else 0
        
        days_left = (g.deadline - timezone.now().date()).days
        daily_req = round(remaining / days_left, 2) if days_left > 0 else remaining
        
        status = "Achieved" if progress >= 100 else "In Progress"
        color = "success" if progress >= 100 else "primary"

        goals_data.append({
            'obj': g,
            'saved_amount': saved_amount,
            'remaining': remaining,
            'progress': progress,
            'daily_required': daily_req,
            'days_left': max(0, days_left),
            'status': status,
            'color': color
        })

    return render(request, 'savings_wallet.html', {'goals': goals_data})

def delete_savings_goal(request, goal_id):
    goal = get_object_or_404(SavingsGoal, id=goal_id, user=request.user)
    goal.delete()
    messages.warning(request, "Savings goal deleted.")
    return redirect('savings_goals')

def update_savings_amount(request, goal_id):
    # Import inside the function to avoid top-level ImportError
    from .models import SavingsGoal, GoalTransaction
    
    if request.method == "POST":
        goal = get_object_or_404(SavingsGoal, id=goal_id, user=request.user)
        
        try:
            new_total = float(request.POST.get("new_total_amount", 0))
            
            # Clear old deposits and create a single balancing deposit
            goal.transactions.all().delete()
            GoalTransaction.objects.create(
                goal=goal, 
                amount=new_total, 
                description="Manual balance update"
            )
            messages.success(request, f"Balance for {goal.goal_name} updated to ₹{new_total}")
        except ValueError:
            messages.error(request, "Invalid amount entered.")
            
    return redirect('savings_goals')
from django.shortcuts import render, redirect
from django.db.models import Sum
from django.utils import timezone
from .models import Income, Expense  # Assuming Expense model exists
from django.contrib import messages

def add_income(request):
    if request.method == "POST":
        amount = request.POST.get('amount')
        source = request.POST.get('source')
        description = request.POST.get('description')
        date = request.POST.get('date')

        # Create the record
        new_income = Income(
            user=request.user,
            amount=amount,
            source=source,
            description=description
        )
        if date:
            new_income.date = date
        new_income.save()

        messages.success(request, "Income entry added successfully!")
        return redirect('dashboard')
        
    return render(request, 'add_income.html')
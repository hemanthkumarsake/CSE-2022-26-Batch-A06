from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from .models import *
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Q, Count
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
import string
from django.core.mail import send_mail
from django.conf import settings
import joblib
import pandas as pd
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods



# Create your views here.

MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'random_forest_model (7).pkl')
model = joblib.load(MODEL_PATH)

def calculate_engineered_features(input_data):
    """
    Calculate all engineered features based on input data
    """
    # Extract basic features
    expense_month = float(input_data.get('expense_month', 1))
    budget_category_limit = float(input_data.get('budget_category_limit', 0))
    monthly_budget = float(input_data.get('monthly_budget', 0))
    is_subscription = float(input_data.get('is_subscription', 0))
    group_expense = float(input_data.get('group_expense', 0))
    savings_goal_linked = float(input_data.get('savings_goal_linked', 0))
    alert_triggered = float(input_data.get('alert_triggered', 0))
    
    # Calculate month-related features
    month_rad = 2 * np.pi * expense_month / 12
    month_cos = np.cos(month_rad)
    month_sin = np.sin(month_rad)
    
    # Calculate quarter
    quarter = ((int(expense_month) - 1) // 3) + 1
    
    # Calculate year/quarter end flags
    is_quarter_end = 1 if int(expense_month) in [3, 6, 9, 12] else 0
    is_year_end = 1 if int(expense_month) in [11, 12] else 0
    
    # Calculate budget ratios
    budget_utilization = float(input_data.get('budget_utilization', 0))
    monthly_budget_utilization = float(input_data.get('monthly_budget_utilization', 0))
    budget_to_monthly_ratio = budget_category_limit / (monthly_budget + 1e-6)
    
    # Calculate interaction features
    subscription_budget_interaction = is_subscription * budget_category_limit
    group_alert_interaction = group_expense * alert_triggered
    savings_alert_interaction = savings_goal_linked * alert_triggered
    
    # Get other features from input
    budget_exceed_flag = float(input_data.get('budget_exceed_flag', 0))
    expense_size = float(input_data.get('expense_size', 1))
    
    # Return all features in correct order
    return {
        'category': input_data.get('category', 'Transportation'),
        'expense_month': expense_month,
        'payment_mode': input_data.get('payment_mode', 'Cash'),
        'is_subscription': is_subscription,
        'budget_category_limit': budget_category_limit,
        'monthly_budget': monthly_budget,
        'budget_exceeded': float(input_data.get('budget_exceeded', 0)),
        'group_expense': group_expense,
        'savings_goal_linked': savings_goal_linked,
        'alert_triggered': alert_triggered,
        'month_cos': month_cos,
        'month_sin': month_sin,
        'is_year_end': is_year_end,
        'budget_exceed_flag': budget_exceed_flag,
        'quarter': quarter,
        'budget_utilization': budget_utilization,
        'budget_to_monthly_ratio': budget_to_monthly_ratio,
        'savings_alert_interaction': savings_alert_interaction,
        'is_quarter_end': is_quarter_end,
        'subscription_budget_interaction': subscription_budget_interaction,
        'monthly_budget_utilization': monthly_budget_utilization,
        'expense_size': expense_size,
        'group_alert_interaction': group_alert_interaction
    }

@require_http_methods(["POST"])
def predict_expense(request):
    """Handle prediction requests from form"""
    try:
        # Get form data
        input_data = request.POST.dict()
        
        # Convert numeric fields
        numeric_fields = [
            'expense_month', 'is_subscription', 'budget_category_limit',
            'monthly_budget', 'budget_exceeded', 'group_expense',
            'savings_goal_linked', 'alert_triggered', 'month_cos',
            'month_sin', 'is_year_end', 'budget_exceed_flag', 'quarter',
            'budget_utilization', 'budget_to_monthly_ratio',
            'savings_alert_interaction', 'is_quarter_end',
            'subscription_budget_interaction', 'monthly_budget_utilization',
            'expense_size', 'group_alert_interaction'
        ]
        
        for field in numeric_fields:
            if field in input_data:
                try:
                    input_data[field] = float(input_data[field])
                except (ValueError, TypeError):
                    input_data[field] = 0.0
        
        # Calculate engineered features
        engineered_data = calculate_engineered_features(input_data)
        
        # Create DataFrame for prediction
        df = pd.DataFrame([engineered_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Prepare response
        response = {
            'success': True,
            'prediction': float(prediction),
            'features_used': list(df.columns),
            'feature_values': engineered_data
        }
        
        return JsonResponse(response)
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def api_predict_expense(request):
    """API endpoint for direct JSON input"""
    try:
        # Parse JSON data
        try:
            input_data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        
        if not input_data:
            return JsonResponse({'error': 'No data provided'}, status=400)
        
        # Convert string values to float where needed
        for key, value in input_data.items():
            if isinstance(value, str):
                try:
                    input_data[key] = float(value)
                except ValueError:
                    pass  # Keep as string for non-numeric fields
        
        # Calculate engineered features
        engineered_data = calculate_engineered_features(input_data)
        
        # Create DataFrame for prediction
        df = pd.DataFrame([engineered_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return JsonResponse({
            'success': True,
            'predicted_amount': float(prediction),
            'features_used': list(df.columns),
            'feature_values': engineered_data
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def expense_predictor_home(request):
    """Render the expense predictor home page"""
    return render(request, 'prediction.html')

def register(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists!')
            return redirect('register')
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists!')
            return redirect('user_login')

        User.objects.create_user(
            username=username,
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password
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
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid Credentials!')
            return redirect('user_login')
    return render(request, 'login.html')

@login_required
def dashboard(request):
    today = timezone.now().date()
    current_month_start = today.replace(day=1)

    username = request.user.username
    
    # Get current month's expenses
    monthly_expenses = Addexpenses.objects.filter(
        user=request.user,
        time_stamp__date__gte=current_month_start,
        time_stamp__date__lte=today
    )
    
    # Calculate total expenses for current month
    total_expenses = monthly_expenses.aggregate(total=Sum('spending_amount'))['total'] or 0
    
    # Get expense count
    expense_count = monthly_expenses.count()
    
    # Get recent expenses (last 5)
    recent_expenses = Addexpenses.objects.filter(
        user=request.user
    ).order_by('-time_stamp')[:5]
    
    # Calculate estimated income (for demo purposes - you can modify this logic)
    # Assuming income is 1.5x expenses or you can create an Income model
    estimated_income = total_expenses * 1.5
    savings = estimated_income - total_expenses
    
    # Get budget goals for current month
    budget_goals = BudgetGoalModel.objects.filter(
        user=request.user,
        month__year=today.year,
        month__month=today.month
    )
    
    total_budget = budget_goals.aggregate(total=Sum('planned_amount'))['total'] or 0
    
    # Calculate budget utilization
    budget_utilization = 0
    if total_budget > 0:
        budget_utilization = (total_expenses / float(total_budget)) * 100
    
    # Get active goals count
    active_goals = BudgetGoalModel.objects.filter(
        user=request.user,
        end_of_month__gte=today
    ).count()
    
    # Get active subscriptions
    active_subscriptions = subscriptionModel.objects.filter(
        user=request.user,
        is_active=True
    )
    
    total_subscriptions = active_subscriptions.count()
    
    # Calculate monthly subscription cost
    subscription_cost = 0
    for sub in active_subscriptions:
        if sub.plan_type == 'monthly':
            subscription_cost += float(sub.price)
        else:  # yearly
            subscription_cost += float(sub.price) / 12
    
    # Get upcoming subscription renewals (next 30 days)
    upcoming_renewals = []
    renewals_count = 0
    
    for sub in active_subscriptions:
        # Calculate next renewal date based on created_at
        days_since_created = (today - sub.created_at.date()).days
        if sub.plan_type == 'monthly':
            # Calculate next monthly renewal
            months_passed = days_since_created // 30
            next_renewal = sub.created_at.date() + timedelta(days=30 * (months_passed + 1))
        else:  # yearly
            # Calculate next yearly renewal
            years_passed = days_since_created // 365
            next_renewal = sub.created_at.date() + timedelta(days=365 * (years_passed + 1))
        
        days_until = (next_renewal - today).days
        if 0 <= days_until <= 30:
            renewals_count += 1
            upcoming_renewals.append({
                'name': sub.name,
                'amount': sub.price,
                'days': days_until,
                'plan_type': sub.plan_type,
                'logo': sub.logo.url if sub.logo else None
            })
    
    upcoming_renewals = sorted(upcoming_renewals, key=lambda x: x['days'])[:3]  # Top 3
    
    # Calculate expense trend (compare with previous month)
    previous_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
    previous_month_end = current_month_start - timedelta(days=1)
    
    previous_month_expenses = Addexpenses.objects.filter(
        user=request.user,
        time_stamp__date__gte=previous_month_start,
        time_stamp__date__lte=previous_month_end
    ).aggregate(total=Sum('spending_amount'))['total'] or 0
    
    expense_trend = 0
    if previous_month_expenses > 0:
        expense_trend = ((total_expenses - previous_month_expenses) / previous_month_expenses) * 100
    
    # Get top spending categories for current month
    category_expenses = monthly_expenses.values('category_name').annotate(
        total=Sum('spending_amount')
    ).order_by('-total')[:3]
    
    top_categories = []
    for item in category_expenses:
        top_categories.append({
            'name': item['category_name'] or 'Uncategorized',
            'amount': item['total'],
            'percentage': (item['total'] / total_expenses * 100) if total_expenses > 0 else 0
        })
    
    # Get budget goal progress
    goal_progress = []
    for goal in budget_goals[:3]:  # Top 3 goals
        spent = Addexpenses.objects.filter(
            user=request.user,
            category_name=goal.category.category if goal.category else None,
            time_stamp__date__gte=goal.month,
            time_stamp__date__lte=goal.end_of_month
        ).aggregate(total=Sum('spending_amount'))['total'] or 0
        
        progress = (spent / float(goal.planned_amount) * 100) if goal.planned_amount > 0 else 0
        goal_progress.append({
            'category': goal.category.category if goal.category else 'Overall',
            'planned': goal.planned_amount,
            'spent': spent,
            'progress': progress,
            'remaining': float(goal.planned_amount) - spent
        })
    
    # Prepare category data for chart
    all_categories = monthly_expenses.values('category_name').annotate(
        total=Sum('spending_amount')
    ).order_by('-total')[:5]
    
    category_labels = []
    category_data = []
    category_colors = ['#0066FF', '#7C3AED', '#F59E0B', '#EC4899', '#10B981']
    
    for item in all_categories:
        category_labels.append(item['category_name'] or 'Uncategorized')
        category_data.append(float(item['total']))
    
    # Prepare daily expense data for the last 7 days
    daily_labels = []
    daily_expenses = []
    
    for i in range(6, -1, -1):
        date = today - timedelta(days=i)
        daily_total = Addexpenses.objects.filter(
            user=request.user,
            time_stamp__date=date
        ).aggregate(total=Sum('spending_amount'))['total'] or 0
        daily_expenses.append(float(daily_total))
        daily_labels.append(date.strftime('%a'))
    
    # Format values for template
    context = {
        # Stats for the cards (matching your template's variable names)
        'monthly_income': round(estimated_income, 2),
        'monthly_expenses': round(total_expenses, 2),
        'monthly_savings': round(savings, 2),
        'budget_utilization': round(budget_utilization, 1),
        
        # Trend percentages
        'income_trend': round(expense_trend * 0.8, 1),  # Sample calculation
        'expense_trend': round(expense_trend, 1),
        'savings_trend': round(expense_trend * 1.2, 1),
        'budget_trend': round(expense_trend * 0.5, 1),
        
        # Trend directions
        'income_trend_direction': 'up' if expense_trend * 0.8 > 0 else 'down',
        'expense_trend_direction': 'up' if expense_trend > 0 else 'down',
        'savings_trend_direction': 'up' if expense_trend * 1.2 > 0 else 'down',
        'budget_trend_direction': 'up' if expense_trend * 0.5 > 0 else 'down',
        
        # Recent transactions
        'recent_transactions': [
            {
                'title': exp.Buyed_Items or 'Expense',
                'category': exp.category_name or 'Uncategorized',
                'amount': exp.spending_amount,
                'date': exp.time_stamp.strftime('%d %b'),
                'icon': get_icon_for_category(exp.category_name)
            } for exp in recent_expenses
        ],
        
        # AI Insights data
        'next_month_prediction': round(total_expenses * 1.02, 2),  # 2% increase prediction
        'prediction_trend': '↓' if expense_trend < 0 else '↑',
        'prediction_percentage': abs(round(expense_trend * 0.3, 1)),
        
        'savings_potential': round(savings * 1.1, 2),
        'savings_potential_trend': '↑',
        'savings_percentage': 18,
        
        'subscription_alert': round(subscription_cost, 2),
        'subscription_savings': round(subscription_cost * 12, 2),
        
        'financial_health': 92,
        'health_rank': 'Top 15%',
        
        # Chart data
        'category_labels': json.dumps(category_labels),
        'category_data': json.dumps(category_data),
        'daily_labels': json.dumps(daily_labels),
        'daily_expenses': json.dumps(daily_expenses),
        
        # Additional data for template
        'expense_count': expense_count,
        'active_goals': active_goals,
        'total_subscriptions': total_subscriptions,
        'subscription_cost': round(subscription_cost, 2),
        'renewals_count': renewals_count,
        'upcoming_renewals': upcoming_renewals,
        'top_categories': top_categories,
        'goal_progress': goal_progress,
        'total_budget': round(total_budget, 2),
        'username':username
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

def user_logout(request):
    request.session.flush()
    logout(request)
    return redirect('index')

def add_expenses(request):
    current_month = datetime.now().month
    total_spending = Addexpenses.objects.filter(time_stamp__month=current_month).aggregate(Sum('spending_amount'))
    total_amount = total_spending['spending_amount__sum'] or 0
    user_categories = Finanace_Category.objects.filter(user=request.user)
    recent_expenses = Addexpenses.objects.filter(user=request.user).order_by('-time_stamp')
    if request.method == 'POST':
        category_name = request.POST.get('category_name')
        spending_amount = request.POST.get('spending_amount')
        Buyed_Items = request.POST.get('Buyed_Items')
        bill = request.FILES.get('bill')
        
        Addexpenses.objects.create(
            user=request.user,
            category_name=category_name,
            spending_amount=spending_amount,
            Buyed_Items=Buyed_Items,
            bill=bill
        )
        messages.success(request, 'Bill details added successfully')
        return redirect('add_expenses')
    # monthly_total = 
    return render(request, 'add_expenses.html', {'user_categories':user_categories, 'recent_expenses':recent_expenses, 'total_amount':total_amount})

@login_required
def budget_goals(request):
    # Get current month and year
    today = timezone.now().date()
    current_month = today.replace(day=1)
    
    # Handle form submission for new budget goal
    if request.method == 'POST':
        category_id = request.POST.get('category')
        month_year = request.POST.get('month')
        planned_amount = request.POST.get('planned_amount')
        
        if month_year and planned_amount:
            # Parse the month input (format: YYYY-MM)
            month_date = datetime.strptime(month_year, '%Y-%m').date()
            
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
    # Get date range from request or default to current month
    today = timezone.now().date()
    
    # Handle filter parameters
    filter_type = request.GET.get('filter', 'month')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    category_filter = request.GET.get('category', '')
    
    # Set date range based on filter
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
            # Get first day of current quarter
            quarter_month = ((today.month - 1) // 3) * 3 + 1
            start_date = today.replace(month=quarter_month, day=1)
            end_date = today
        elif filter_type == 'year':
            start_date = today.replace(month=1, day=1)
            end_date = today
        else:  # week
            start_date = today - timedelta(days=today.weekday())
            end_date = today
    
    # Get all expenses for the user within date range
    expenses = Addexpenses.objects.filter(
        user=request.user,
        time_stamp__date__gte=start_date,
        time_stamp__date__lte=end_date
    ).order_by('-time_stamp')
    
    # Apply category filter if specified
    if category_filter:
        expenses = expenses.filter(category_name=category_filter)
    
    # Get all categories for filter dropdown
    categories = Finanace_Category.objects.filter(user=request.user).values_list('category', flat=True).distinct()
    
    # Calculate summary statistics
    total_expenses = expenses.count()
    total_amount = expenses.aggregate(total=Sum('spending_amount'))['total'] or 0
    avg_expense = expenses.aggregate(avg=Sum('spending_amount'))['avg'] or 0
    if total_expenses > 0:
        avg_expense = total_amount / total_expenses
    
    # Get highest expense
    highest_expense = expenses.order_by('-spending_amount').first()
    
    # Get lowest expense (non-zero)
    lowest_expense = expenses.filter(spending_amount__gt=0).order_by('spending_amount').first()
    
    # Category-wise breakdown
    category_data = expenses.values('category_name').annotate(
        total=Sum('spending_amount'),
        count=Count('id')
    ).order_by('-total')
    
    # Prepare data for charts
    categories_chart_labels = []
    categories_chart_data = []
    categories_colors = []
    
    # Color palette for charts
    color_palette = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD',
        '#D4A5A5', '#9B59B6', '#3498DB', '#E67E22', '#2ECC71',
        '#F1C40F', '#E74C3C', '#1ABC9C', '#34495E', '#7F8C8D'
    ]
    
    for i, item in enumerate(category_data):
        categories_chart_labels.append(item['category_name'] or 'Uncategorized')
        categories_chart_data.append(float(item['total']))
        categories_colors.append(color_palette[i % len(color_palette)])
    
    # Daily trend data
    date_range = (end_date - start_date).days + 1
    daily_data = defaultdict(float)
    
    for exp in expenses:
        date_str = exp.time_stamp.date().strftime('%Y-%m-%d')
        daily_data[date_str] += float(exp.spending_amount)
    
    # Fill in missing dates
    daily_trend_labels = []
    daily_trend_data = []
    for i in range(date_range):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime('%Y-%m-%d')
        daily_trend_labels.append(current_date.strftime('%d %b'))
        daily_trend_data.append(float(daily_data.get(date_str, 0)))
    
    # Monthly comparison (for current year)
    monthly_data = defaultdict(float)
    year_expenses = Addexpenses.objects.filter(
        user=request.user,
        time_stamp__year=today.year
    )
    
    for exp in year_expenses:
        month_str = exp.time_stamp.strftime('%B')
        monthly_data[month_str] += float(exp.spending_amount)
    
    # Get all months
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    monthly_labels = []
    monthly_data_values = []
    for month in months_order:
        if month in monthly_data:
            monthly_labels.append(month[:3])  # Abbreviate month
            monthly_data_values.append(monthly_data[month])
    
    # Budget vs Actual comparison
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
        
        # Get actual spending for this category and month
        actual = Addexpenses.objects.filter(
            user=request.user,
            category_name=category_name if category_name != 'Overall' else None,
            time_stamp__date__gte=goal.month,
            time_stamp__date__lte=goal.end_of_month
        ).aggregate(total=Sum('spending_amount'))['total'] or 0
        budget_actual.append(float(actual))
    
    # Top spending items
    top_items = expenses.values('Buyed_Items').annotate(
        total=Sum('spending_amount'),
        count=Count('id')
    ).filter(~Q(Buyed_Items__isnull=True) & ~Q(Buyed_Items='')
    ).order_by('-total')[:10]
    
    # Spending by day of week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_data = defaultdict(float)
    
    for exp in expenses:
        weekday = exp.time_stamp.weekday()  # 0 = Monday
        weekday_data[weekday] += float(exp.spending_amount)
    
    weekday_labels = weekdays
    weekday_values = [float(weekday_data.get(i, 0)) for i in range(7)]
    
    context = {
        'expenses': expenses[:20],  # Recent 20 for table
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
        
        # Chart data
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
        
        # Summary data
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
            # Get form data
            name = request.POST.get('name')
            plan_type = request.POST.get('plan_type')
            price = request.POST.get('price')
            is_active = request.POST.get('is_active') == 'on'
            logo = request.FILES.get('logo')
            bill_slip = request.FILES.get('bill_slip')
            
            # Validate required fields
            if not name or not price:
                messages.error(request, 'Name and price are required fields!')
                return redirect('add_subscription')
            
            # Create new subscription
            subscription = subscriptionModel(
                user=request.user,
                name=name,
                plan_type=plan_type,
                price=price,
                is_active=is_active
            )
            
            # Add files if provided
            if logo:
                subscription.logo = logo
            if bill_slip:
                subscription.bill_slip = bill_slip
            
            subscription.save()
            
            messages.success(request, f'Subscription "{name}" added successfully!')
            return redirect('subscription_list')
            
        except Exception as e:
            messages.error(request, f'Error adding subscription: {str(e)}')
            return redirect('add_subscription')
    
    # GET request - show form
    context = {
        'today': timezone.now().strftime('%Y-%m-%d'),
    }
    
    return render(request, 'add_subscription.html', context)


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


@login_required
def edit_profile(request):
    user = request.user
    
    if request.method == 'POST':
        # Get form data
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')  # You'll need to add this field to User model or create a Profile model
        currency = request.POST.get('currency')
        notification_email = request.POST.get('notification_email') == 'on'
        notification_push = request.POST.get('notification_push') == 'on'
        
        # Update user
        user.first_name = first_name
        user.last_name = last_name
        user.email = email
        
        # Save to database
        try:
            user.save()
            messages.success(request, 'Profile updated successfully!')
        except Exception as e:
            messages.error(request, f'Error updating profile: {str(e)}')
        
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
                last_name=last_name
            )
            
            # Create AddFamilyMember record
            family_member = AddFamilyMember.objects.create(
                Family_code=family_code,
                Added_by=request.user,
                relation=relation
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
        members_data.append({
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
    """Delete a family member"""
    if request.method == 'POST':
        try:
            member = AddFamilyMember.objects.get(id=member_id, Added_by=request.user)
            member.delete()
            messages.success(request, 'Family member removed successfully!')
        except AddFamilyMember.DoesNotExist:
            messages.error(request, 'Family member not found!')
    return redirect('family_members_list')

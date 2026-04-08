from django.urls import path
from .views import *
from django.views.generic import TemplateView

urlpatterns = [
    
    path('', TemplateView.as_view(template_name='index.html'), name='index'),
    path('register/', register, name='register'),
    path('user_login/', user_login, name='user_login'),
    path('dashboard/', dashboard, name='dashboard'),
    path('add_category/', add_category, name='add_category'),
    path('user_logout/', user_logout, name='user_logout'),
    path('add_expenses/', add_expenses, name='add_expenses'),
    path('budget-goals/', budget_goals, name='budget_goals'),
    path('budget-goals/<int:goal_id>/delete/', delete_budget_goal, name='delete_budget_goal'),
    path('budget-goals/<int:goal_id>/edit/', edit_budget_goal, name='edit_budget_goal'),
    path('expenses/report/', expenses_report, name='expenses_report'),
    path('expense/<int:expense_id>/details/', expense_details, name='expense_details'),
    path('subscriptions/', subscription_list, name='subscription_list'),
    path('subscriptions/add/', add_subscription, name='add_subscription'),
    path('subscriptions/<int:sub_id>/', subscription_detail, name='subscription_detail'),
    path('subscriptions/<int:sub_id>/edit/', edit_subscription, name='edit_subscription'),
    path('subscriptions/<int:sub_id>/delete/', delete_subscription, name='delete_subscription'),
    path('subscriptions/<int:sub_id>/toggle/', toggle_subscription_status, name='toggle_subscription'),
    path('profile/', profile_view, name='profile_view'),
    path('profile/edit/', edit_profile, name='edit_profile'),
    path('profile/change-password/', change_password, name='change_password'),
    path('profile/delete-account/', delete_account, name='delete_account'),
    path('family/add/', add_family_member, name='add_family_member'),
    path('family/members/', family_members_list, name='family_members_list'),
    path('family/member/<int:member_id>/delete/', delete_family_member, name='delete_family_member'),
    path('predict/', predict_expense, name='predict'),
    path('api/predict/', api_predict_expense, name='api_predict'),
    path('expense_predictor_home/', expense_predictor_home, name='expense_predictor_home')

    ]
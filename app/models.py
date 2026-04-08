from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.utils import timezone

import os
import uuid

class SavingsGoal(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    goal_name = models.CharField(max_length=200)
    target_amount = models.DecimalField(max_digits=12, decimal_places=2)
    deadline = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.goal_name} - {self.user.username}"

class GoalTransaction(models.Model):
    goal = models.ForeignKey(SavingsGoal, on_delete=models.CASCADE, related_name='transactions')
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    description = models.CharField(max_length=255, default="Manual Deposit")
    date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"₹{self.amount} to {self.goal.goal_name}"

class monthly_salary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    salary = models.IntegerField(null=True)

    class Meta:
        db_table = 'monthly_salary'

class Finanace_Category(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    category = models.CharField(max_length=50, null=True)
    description = models.CharField(max_length=100, null=True)

    class Meta:
        db_table = 'Finanace_Category'

class Addexpenses(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    category_name = models.CharField(max_length=50, null=True)
    time_stamp = models.DateTimeField(auto_now_add=True)
    spending_amount = models.FloatField(null=True)
    Buyed_Items = models.TextField(null=True)
    bill = models.FileField(upload_to=os.path.join('static', 'Bills'))
    is_anomaly = models.BooleanField(default=False)

    class Meta:
        db_table = 'add_expenses'

class BudgetGoalModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    category = models.ForeignKey('Finanace_Category', on_delete=models.CASCADE, blank=True, null=True)
    month = models.DateField()
    end_of_month = models.DateField()
    planned_amount = models.DecimalField(max_digits=12, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'BudgetGoalModel'

    def __str__(self):
        cat = self.category.category if self.category else "Overall"
        return f"{cat} - {self.month.strftime('%b %Y')}"
    
class subscriptionModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    plans = [
        ('monthly', 'Monthly'),
        ('yearly', 'Yearly'),
    ]
    logo = models.ImageField(upload_to=os.path.join('static', 'Logos'), blank=True)
    name = models.CharField(max_length=20, null=True)
    plan_type = models.CharField(max_length=20, choices=plans, default='monthly')
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    bill_slip = models.FileField(upload_to=os.path.join('static', 'subcription_bills'))

    class Meta:
        db_table = 'subscription_plans'

class AddFamilyMember(models.Model):
    Family_code = models.CharField(max_length=10, null=True)
    Added_by = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True, related_name='family_members_added')
    Added_person = models.OneToOneField(User, on_delete=models.CASCADE, blank=True, null=True, related_name='family_member_profile')
    time_stamp = models.DateTimeField(auto_now_add=True)
    relation = models.CharField(max_length=20, null=True)  
    
    class Meta:
        db_table = 'AddFamilyMember'
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Income(models.Model):
    SOURCE_CHOICES = [
        ('Salary', 'Salary'),
        ('Freelance', 'Freelance'),
        ('Gift', 'Gift'),
        ('Business', 'Business'),
        ('Other', 'Other'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    source = models.CharField(max_length=50, choices=SOURCE_CHOICES)
    description = models.TextField(blank=True, null=True)
    date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.source} - {self.amount}"
    
class Expense(models.Model):
    CATEGORY_CHOICES = [
        ('Food', 'Food'),
        ('Rent', 'Rent'),
        ('Bills', 'Bills'),
        ('Travel', 'Travel'),
        ('Entertainment', 'Entertainment'),
        ('Other', 'Other'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    description = models.TextField(blank=True, null=True)
    date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.category} - {self.amount}"
    
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15, blank=True, null=True)

    def __str__(self):
        return self.user.username
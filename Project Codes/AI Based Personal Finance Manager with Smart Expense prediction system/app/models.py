from django.db import models
from django.contrib.auth.models import User
import os
import os
import uuid
from django.contrib.auth.hashers import make_password


# Create your models here.

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
        cat = self.category.name if self.category else "Overall"
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
    Added_by = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    time_stamp = models.DateTimeField(auto_now_add=True)
    relation = models.CharField(max_length=20, null=True)  
    
    class Meta:
        db_table = 'AddFamilyMember'

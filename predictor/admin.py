from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.core.exceptions import ValidationError
from .models import User, PatientRecord


class CustomUserAdmin(BaseUserAdmin):
    model = User
    list_display = ['username', 'email', 'role', 'phone_number']
    list_filter = ['role']
    search_fields = ['username', 'email']
    ordering = ['username']

    fieldsets = BaseUserAdmin.fieldsets + (
        (None, {'fields': ('phone_number', 'role')}),
    )

    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        (None, {'fields': ('phone_number', 'role')}),
    )

    def save_model(self, request, obj, form, change):
        if obj.role == 'admin':
            existing_admins = User.objects.filter(role='admin').exclude(pk=obj.pk)
            if existing_admins.exists():
                raise ValidationError("Only one admin is allowed.")
        super().save_model(request, obj, form, change)


@admin.register(PatientRecord)
class PatientRecordAdmin(admin.ModelAdmin):
    list_display = ['user', 'age', 'bp', 'sg', 'al', 'ckd_stage', 'classification']
    search_fields = ['user__username']
    list_filter = ['ckd_stage', 'classification', 'smoker']


admin.site.register(User, CustomUserAdmin)

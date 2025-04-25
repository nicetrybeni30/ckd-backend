
from rest_framework import serializers
from predictor.models import PatientRecord, User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'role', 'phone_number']

class PatientRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientRecord
        exclude = ['created_at']

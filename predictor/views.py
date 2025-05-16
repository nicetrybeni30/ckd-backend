from django.conf import settings
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from django.contrib.auth import get_user_model
from predictor.models import PatientRecord, ModelRetrainLog
from predictor.serializers import PatientRecordSerializer, UserSerializer
from django.contrib.auth.hashers import make_password
from rest_framework import status
from django.db.models import Count

import time, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

PROGRESS_FILE = settings.BASE_DIR / 'predictor' / 'ml_model' / 'progress.json'

User = get_user_model()

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def retrain_progress(request):
    if request.user.role != 'admin':
        return Response({"error": "Not allowed"}, status=403)

    try:
        if not PROGRESS_FILE.exists():
            return Response({"epoch": 0, "percent": 0, "seconds_left": 0})

        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)

        return Response(data)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

# 🧠 Admin: List all users
class UserListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != "admin":
            return Response({"error": "Not allowed"}, status=403)
        users = User.objects.all()
        return Response(UserSerializer(users, many=True).data)

# 🧠 Admin: Get user detail / update user detail
@api_view(['GET', 'PUT'])
@permission_classes([IsAuthenticated])
def get_user_details(request, user_id):
    if request.user.role != "admin":
        return Response({"error": "Not allowed"}, status=403)

    try:
        user = User.objects.get(id=user_id)

        if request.method == 'GET':
            serializer = UserSerializer(user)
            return Response(serializer.data)

        if request.method == 'PUT':
            data = request.data
            user.username = data.get('username', user.username)
            user.email = data.get('email', user.email)
            user.phone_number = data.get('phone_number', user.phone_number)

            if data.get('new_password'):
                user.password = make_password(data['new_password'])

            user.save()
            return Response({"message": "User updated successfully."})

    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=404)

# 🧠 Admin: Get all patients with record (new)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def users_with_records(request):
    if request.user.role != "admin":
        return Response({"error": "Not allowed"}, status=403)

    patients = User.objects.filter(role='patient').prefetch_related('patientrecord')

    results = []
    for patient in patients:
        record = None
        try:
            record_obj = patient.patientrecord
            record = PatientRecordSerializer(record_obj).data
        except PatientRecord.DoesNotExist:
            record = None

        results.append({
            "id": patient.id,
            "username": patient.username,
            "email": patient.email,
            "phone_number": patient.phone_number,
            "record": record
        })

    return Response(results)

# 🧠 Patient: Get their own user info
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_logged_in_user_info(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)

# 🧠 Patient: Get their own record
class RecordMeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            record = PatientRecord.objects.get(user=request.user)
            return Response(PatientRecordSerializer(record).data)
        except PatientRecord.DoesNotExist:
            return Response({"error": "No patient record found."}, status=404)

# 🧠 Patient: Create or Update their own record
class RecordCreateUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            if PatientRecord.objects.filter(user=request.user).exists():
                return Response({"error": "Patient record already exists."}, status=400)

            serializer = PatientRecordSerializer(data={**request.data, "user": request.user.id})

            if serializer.is_valid():
                serializer.save(user=request.user)
                return Response(serializer.data)
            return Response(serializer.errors, status=400)

        except Exception as e:
            return Response({"error": str(e)}, status=500)

# 🧠 Admin: Get or Edit any patient's record
class RecordDetailByAdminView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, user_id):
        if request.user.role != "admin":
            return Response({"error": "Not allowed"}, status=403)
        try:
            record = PatientRecord.objects.get(user__id=user_id)
            return Response(PatientRecordSerializer(record).data)
        except PatientRecord.DoesNotExist:
            return Response({"error": "Record not found"}, status=404)

    def put(self, request, user_id):
        if request.user.role != "admin":
            return Response({"error": "Not allowed"}, status=403)
        try:
            record = PatientRecord.objects.get(user__id=user_id)
            serializer = PatientRecordSerializer(record, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=400)
        except PatientRecord.DoesNotExist:
            return Response({"error": "Record not found"}, status=404)

# 🧠 API: CKD Model Info
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_model_info(request):
    try:
        latest_log = ModelRetrainLog.objects.latest('retrained_at')
        accuracy = round(latest_log.accuracy * 100, 2)
        retrained_at = latest_log.retrained_at.strftime('%Y-%m-%d %H:%M:%S')
    except ModelRetrainLog.DoesNotExist:
        accuracy = None
        retrained_at = "Unknown"

    return Response({
        "model_accuracy": accuracy if accuracy is not None else "N/A",
        "retrained_at": retrained_at
    })

# 🧠 Patient Signup (Public)
@api_view(['POST'])
@permission_classes([AllowAny])
def patient_register(request):
    username = request.data.get('username')
    email = request.data.get('email')
    password = request.data.get('password')

    if not username or not email or not password:
        return Response({"error": "All fields are required."}, status=400)

    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already exists."}, status=400)
    if User.objects.filter(email=email).exists():
        return Response({"error": "Email already exists."}, status=400)

    user = User.objects.create(
        username=username,
        email=email,
        password=make_password(password),
        role='patient'
    )
    return Response({"message": "Patient account created successfully!"})

# 🧠 Patient: Update Account Info
class UpdateAccountView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request):
        try:
            user = request.user
            data = request.data

            user.username = data.get('username', user.username)
            user.email = data.get('email', user.email)

            if data.get('password'):
                user.password = make_password(data.get('password'))

            user.save()

            return Response({"message": "Account updated successfully!"})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

# 🆕 ✅ API: Get Retrain History (for LineChart)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_retrain_history(request):
    logs = ModelRetrainLog.objects.order_by('retrained_at')
    data = [
        {"date": log.retrained_at.strftime('%Y-%m-%d'), "accuracy": round(log.accuracy * 100, 2)}
        for log in logs
    ]
    return Response(data)

# 🆕 ✅ API: Get CKD Stage Distribution (for BarChart)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_ckd_distribution(request):
    if request.user.role != 'admin':
        return Response({"error": "Not allowed"}, status=403)

    distribution = PatientRecord.objects.values('classification').annotate(total=Count('id'))

    result = {
        "Chronic Kidney Disease (Registered)": 0,
        "No Chronic Kidney Disease (Registered)": 0
    }

    for item in distribution:
        raw = item['classification'].strip().lower()
        if raw == 'ckd':
            result['Chronic Kidney Disease (Registered)'] += item['total']
        elif raw == 'notckd':
            result['No Chronic Kidney Disease (Registered)'] += item['total']

    formatted = [{"classification": k, "total": v} for k, v in result.items() if v > 0]
    return Response(formatted)

# 🆕 ✅ API: Get Patient Record by USER_ID (NEW)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_record_by_user(request, user_id):
    if request.user.role != "admin":
        return Response({"error": "Not allowed"}, status=403)
    try:
        record = PatientRecord.objects.get(user__id=user_id)
        serializer = PatientRecordSerializer(record)
        return Response(serializer.data)
    except PatientRecord.DoesNotExist:
        return Response({'error': 'Record not found.'}, status=404)

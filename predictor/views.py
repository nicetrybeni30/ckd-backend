from django.conf import settings
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from django.contrib.auth import get_user_model
from predictor.models import PatientRecord
from predictor.serializers import PatientRecordSerializer, UserSerializer

User = get_user_model()


class UserListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != "admin":
            return Response({"error": "Not allowed"}, status=403)
        users = User.objects.all()
        return Response(UserSerializer(users, many=True).data)


class RecordMeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            record = PatientRecord.objects.get(user=request.user)
            return Response(PatientRecordSerializer(record).data)
        except PatientRecord.DoesNotExist:
            return Response({"error": "No patient record found."}, status=404)


class RecordCreateUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            record = PatientRecord.objects.get(user=request.user)
            serializer = PatientRecordSerializer(record, data=request.data)
        except PatientRecord.DoesNotExist:
            serializer = PatientRecordSerializer(data={**request.data, "user": request.user.id})

        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data)
        return Response(serializer.errors, status=400)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_details(request, user_id):
    if request.user.role != "admin":
        return Response({"error": "Not allowed"}, status=403)
    try:
        user = User.objects.get(id=user_id)
        serializer = UserSerializer(user)
        return Response(serializer.data)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=404)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_logged_in_user_info(request):
    """Endpoint for GET /api/patient/me/"""
    serializer = UserSerializer(request.user)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_model_info(request):
    try:
        with open(settings.BASE_DIR / 'ml_model' / 'model_accuracy.txt', 'r') as f:
            accuracy = float(f.read().strip())
    except:
        accuracy = None

    try:
        with open(settings.BASE_DIR / 'ml_model' / 'retrained_at.txt', 'r') as f:
            retrained_at = f.read().strip()
    except:
        retrained_at = "Unknown"

    return Response({
        "model_accuracy": round(accuracy * 100, 2) if accuracy else "N/A",
        "retrained_at": retrained_at
    })
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

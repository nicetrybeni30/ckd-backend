# File: predictor/api/predict.py

import os
import joblib
import traceback
import pandas as pd
from datetime import datetime
from django.utils import timezone
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from predictor.models import PatientRecord

# Absolute paths to model + metadata
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'rf_ckd_model.pkl')
ACCURACY_PATH = os.path.join(BASE_DIR, 'ml_model', 'model_accuracy.txt')
RETRAINED_AT_PATH = os.path.join(BASE_DIR, 'ml_model', 'retrained_at.txt')

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded from:", MODEL_PATH)
except Exception as e:
    model = None
    print(f"⚠️ Model loading failed: {e}")

# Helper: determine CKD stage based on creatinine
def determine_ckd_stage(creatinine):
    if creatinine < 1.5:
        return "Stage 1"
    elif 1.5 <= creatinine < 2.0:
        return "Stage 2"
    elif 2.0 <= creatinine < 5.0:
        return "Stage 3"
    elif 5.0 <= creatinine < 7.0:
        return "Stage 4"
    else:
        return "Stage 5"

# Helper: recommendation text
def generate_recommendation(prediction, stage):
    if prediction == "ckd":
        return f"CKD {stage} detected. Recommend nephrologist consultation, dietary monitoring, and hydration control."
    elif stage != "Stage 1":
        return f"No CKD detected. However, {stage} indicators present. Recommend regular monitoring and healthy lifestyle."
    else:
        return "No CKD detected. Maintain a healthy lifestyle."

# Encode categorical inputs to numeric values
def encode(value):
    return {
        'yes': 1, 'no': 0,
        'present': 1, 'notpresent': 0,
        'good': 1, 'poor': 0,
        'normal': 1, 'abnormal': 0,
        '': 0, None: 0
    }.get(str(value).lower(), 0)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_ckd(request):
    user = request.user

    try:
        record = PatientRecord.objects.get(user=user)

        # Collect all 24 features in correct order
        features = [
            record.age, record.bp, record.sg, record.al, record.su,
            encode(record.rbc), encode(record.pc), encode(record.pcc), encode(record.ba),
            record.bgr, record.bu, record.sc, record.sod, record.pot,
            record.hemo, record.pcv, record.wc, record.rc,
            encode(record.htn), encode(record.dm), encode(record.cad),
            encode(record.appet), encode(record.pe), encode(record.ane),
        ]

        feature_names = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
            'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]

        df = pd.DataFrame([features], columns=feature_names)
        print("📋 Features:", df)

        if model:
            raw_prediction = model.predict(df)[0]
            confidence = max(model.predict_proba(df)[0])
            label_map = {1: "ckd", 0: "notckd"}
            prediction = label_map.get(raw_prediction, "unknown")
        else:
            prediction = "unknown"
            confidence = 0.0

        stage = determine_ckd_stage(record.sc)
        recommendation = generate_recommendation(prediction, stage)

        # Rule-based red flags
        risk_flags = []
        if record.sc >= 5.0:
            risk_flags.append("⚠️ High serum creatinine")
        if record.al >= 3:
            risk_flags.append("⚠️ High albumin level (protein in urine)")
        if record.hemo <= 9.0:
            risk_flags.append("⚠️ Low hemoglobin level")
        if record.pcv <= 30:
            risk_flags.append("⚠️ Low packed cell volume")
        if record.dm.lower() == "yes":
            risk_flags.append("⚠️ Has diabetes")
        if record.htn.lower() == "yes":
            risk_flags.append("⚠️ Has hypertension")

        if prediction == "notckd" and len(risk_flags) >= 3:
            print("⚠️ Overriding model prediction due to multiple red flags")
            prediction = "ckd"
            recommendation = generate_recommendation(prediction, stage)

        # Save prediction results into PatientRecord
        record.last_prediction = prediction
        record.last_recommendation = recommendation
        record.last_confidence = round(confidence * 100, 2)
        record.last_predicted_at = timezone.now()
        record.save()

        # Load accuracy + retrain time
        try:
            with open(ACCURACY_PATH, 'r') as f:
                model_accuracy = float(f.read().strip())
        except:
            model_accuracy = None

        try:
            with open(RETRAINED_AT_PATH, 'r') as f:
                retrained_at = f.read().strip()
        except:
            retrained_at = "Unknown"

        return Response({
            "prediction": prediction,
            "ckd_stage": stage,
            "recommendation": recommendation,
            "confidence": round(confidence * 100, 2),
            "model_accuracy": round(model_accuracy * 100, 2) if model_accuracy else "Unknown",
            "retrained_at": retrained_at,
            "risk_flags": risk_flags
        })

    except PatientRecord.DoesNotExist:
        return Response({"error": "No patient record found."}, status=404)
    except Exception as e:
        traceback.print_exc()
        return Response({"error": f"Prediction failed. {str(e)}"}, status=500)

import os
import traceback
import pandas as pd
from datetime import datetime
from django.utils import timezone
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from predictor.models import PatientRecord
from tensorflow.keras.models import load_model
import numpy as np
import joblib

# Absolute paths to model + metadata
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'deep_learning_ckd_model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'ml_model', 'scaler.pkl')
ACCURACY_PATH = os.path.join(BASE_DIR, 'ml_model', 'model_accuracy.txt')
RETRAINED_AT_PATH = os.path.join(BASE_DIR, 'ml_model', 'retrained_at.txt')

# Load model + scaler
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Keras model loaded from:", MODEL_PATH)
except Exception as e:
    model = None
    scaler = None
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

        # Collect and transform input features
        features = [
            record.age, record.bp, record.sg, record.al, record.su,
            record.bgr, record.bu, record.sc, record.hemo, record.pcv,
            encode(record.htn), encode(record.dm)
        ]
        df = pd.DataFrame([features], columns=[
            'age', 'bp', 'sg', 'al', 'su',
            'bgr', 'bu', 'sc', 'hemo', 'pcv',
            'htn', 'dm'
        ])
        print("📋 Features for model prediction:\n", df)

        if model and scaler:
            scaled = scaler.transform(df)
            prediction_raw = model.predict(scaled)[0][0]
            prediction = "ckd" if prediction_raw >= 0.5 else "notckd"
            confidence = float(prediction_raw) if prediction == "ckd" else 1 - float(prediction_raw)

            print("🔮 Raw model prediction value:", prediction_raw)
            print("🧑‍🧠 Model says:", prediction)
            print("🎯 Confidence score:", round(confidence * 100, 2), "%")
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

        overridden = False
        model_opinion = prediction

        if (prediction == "notckd" or prediction == "unknown") and len(risk_flags) >= 3:
            print("⚠️ Overriding model prediction due to multiple red flags")
            overridden = True
            prediction = "ckd"
            recommendation = generate_recommendation(prediction, stage)

        # Save prediction results
        confidence_pct = float(round(confidence * 100, 2))
        record.last_prediction = prediction
        record.last_recommendation = recommendation
        record.last_confidence = confidence_pct
        record.last_predicted_at = timezone.now()
        record.save()

        # Load accuracy and retrain time
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
            "confidence": confidence_pct,
            "model_accuracy": round(model_accuracy * 100, 2) if model_accuracy else "Unknown",
            "retrained_at": retrained_at,
            "risk_flags": risk_flags,
            "model_opinion": model_opinion,
            "overridden": overridden
        })

    except PatientRecord.DoesNotExist:
        return Response({"error": "No patient record found."}, status=404)
    except Exception as e:
        traceback.print_exc()
        return Response({"error": f"Prediction failed. {str(e)}"}, status=500)

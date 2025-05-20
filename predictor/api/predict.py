# ✅ predict_ckd (with terminal logging for classification and stage)
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from predictor.models import PatientRecord
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib
from django.utils import timezone
from datetime import datetime

# Paths
BASE_DIR = settings.BASE_DIR
MODEL_PATH = BASE_DIR / 'predictor' / 'ml_model' / 'deep_learning_ckd_model.keras'
SCALER_PATH = BASE_DIR / 'predictor' / 'ml_model' / 'scaler.pkl'
ACCURACY_PATH = BASE_DIR / 'predictor' / 'ml_model' / 'model_accuracy.txt'
RETRAINED_AT_PATH = BASE_DIR / 'predictor' / 'ml_model' / 'retrained_at.txt'

# Load model and scaler
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded")
except Exception as e:
    print("❌ Failed to load model or scaler:", e)
    model = None
    scaler = None

def encode(value):
    return 1 if str(value).lower() in ['yes', 'present', 'abnormal', 'poor'] else 0

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

def generate_recommendation(prediction, stage):
    if prediction == "ckd":
        return f"CKD {stage} detected. Recommend nephrologist consultation, dietary monitoring, and hydration control."
    elif stage != "Stage 1":
        return f"No CKD detected. However, {stage} indicators present. Recommend regular monitoring and healthy lifestyle."
    else:
        return "No CKD detected. Maintain a healthy lifestyle."

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_ckd(request):
    user = request.user

    try:
        record = PatientRecord.objects.get(user=user)

        features = [
            record.age, record.bp, record.sg, record.al, record.su,
            encode(record.rbc), encode(record.pc), encode(record.pcc), encode(record.ba),
            record.bgr, record.bu, record.sc, record.sod, record.pot,
            record.hemo, record.pcv, record.wc, record.rc,
            encode(record.htn), encode(record.dm), encode(record.cad),
            encode(record.appet), encode(record.pe), encode(record.ane)
        ]

        columns = [
            'age', 'bp', 'sg', 'al', 'su',
            'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]

        df = pd.DataFrame([features], columns=columns)

        if model and scaler:
            if scaler.n_features_in_ != len(df.columns):
                return Response({"error": f"Model expects {scaler.n_features_in_} features, got {len(df.columns)}."}, status=400)

            scaled = scaler.transform(df)
            prediction_raw = model.predict(scaled)[0][0]
            prediction = "ckd" if prediction_raw >= 0.5 else "notckd"
            confidence = float(prediction_raw) if prediction == "ckd" else 1 - float(prediction_raw)
        else:
            prediction = "unknown"
            confidence = 0.0

        stage = determine_ckd_stage(record.sc)
        recommendation = generate_recommendation(prediction, stage)

        risk_flags = []
        if record.sc >= 5.0:
            risk_flags.append("⚠️ High serum creatinine")
        if record.al >= 3:
            risk_flags.append("⚠️ High albumin level")
        if record.hemo <= 9.0:
            risk_flags.append("⚠️ Low hemoglobin")
        if record.pcv <= 30:
            risk_flags.append("⚠️ Low packed cell volume")
        if str(record.dm).lower() == "yes":
            risk_flags.append("⚠️ Has diabetes")
        if str(record.htn).lower() == "yes":
            risk_flags.append("⚠️ Has hypertension")

        overridden = False
        model_opinion = prediction

        if (prediction == "notckd" or prediction == "unknown") and len(risk_flags) >= 3:
            prediction = "ckd"
            overridden = True
            recommendation = generate_recommendation(prediction, stage)

        # Log prediction to terminal
        print(f"🔍 Prediction: {prediction.upper()}, Stage: {stage}, Confidence: {round(confidence * 100, 2)}%, Overridden: {overridden}")

        record.last_prediction = prediction
        record.last_recommendation = recommendation
        record.last_confidence = float(round(confidence * 100, 2))
        record.last_predicted_at = timezone.now()
        record.save()

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
            "risk_flags": risk_flags,
            "model_opinion": model_opinion,
            "overridden": overridden
        })

    except PatientRecord.DoesNotExist:
        return Response({"error": "No patient record found."}, status=404)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({"error": f"Prediction failed. {str(e)}"}, status=500)

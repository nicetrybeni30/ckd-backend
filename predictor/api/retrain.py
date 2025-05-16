from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from predictor.models import PatientRecord, ModelRetrainLog

import time, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback

PROGRESS_FILE = settings.BASE_DIR / 'predictor' / 'ml_model' / 'progress.json'
User = get_user_model()

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def retrain_model(request):
    if request.user.role != 'admin':
        return Response({"error": "Not allowed"}, status=403)

    model_dir = settings.BASE_DIR / 'predictor' / 'ml_model'
    model_dir.mkdir(parents=True, exist_ok=True)

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    records = PatientRecord.objects.all()
    if not records:
        return Response({"error": "No records to train on."}, status=400)

    data = pd.DataFrame(list(records.values()))
    print(f"📊 Total records in DB: {len(data)}")

    selected = data[[ 
        'age', 'bp', 'sg', 'al', 'su',
        'bgr', 'bu', 'sc', 'hemo', 'pcv',
        'htn', 'dm', 'classification'
    ]].dropna()
    print(f"✅ Records used for training (after dropna): {len(selected)}")

    selected['htn'] = selected['htn'].apply(lambda x: 1 if x == 'yes' else 0)
    selected['dm'] = selected['dm'].apply(lambda x: 1 if x == 'yes' else 0)

    X = selected.drop('classification', axis=1)
    y = selected['classification'].apply(lambda x: 1 if x == 'ckd' else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"🧪 Training samples: {len(X_train)}")
    print(f"🧪 Test samples: {len(X_test)}")

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    total_epochs = 100

    # ✅ Live progress logging per epoch
    start_time = time.time()
    def update_progress(epoch, logs):
        percent = round(((epoch + 1) / total_epochs) * 100, 1)
        elapsed = time.time() - start_time
        estimated_total = elapsed / ((epoch + 1) / total_epochs)
        seconds_left = int(estimated_total - elapsed)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                "epoch": epoch + 1,
                "percent": percent,
                "seconds_left": seconds_left
            }, f)
    progress_callback = LambdaCallback(on_epoch_end=update_progress)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=total_epochs,
        batch_size=16,
        callbacks=[progress_callback],
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    model.save(model_dir / 'deep_learning_ckd_model.keras')
    joblib.dump(scaler, model_dir / 'scaler.pkl')

    log = ModelRetrainLog.objects.create(accuracy=accuracy)
    with open(model_dir / 'model_accuracy.txt', 'w') as f:
        f.write(str(accuracy))
    with open(model_dir / 'retrained_at.txt', 'w') as f:
        f.write(log.retrained_at.strftime('%Y-%m-%d %H:%M:%S'))

    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            "epoch": 100,
            "percent": 100.0,
            "seconds_left": 0
        }, f)

    print(f"✅ Test Accuracy: {accuracy:.2%}")

    return Response({
        "message": "Model retrained successfully.",
        "accuracy": round(accuracy * 100, 2),
        "retrained_at": log.retrained_at.strftime('%Y-%m-%d %H:%M:%S')
    })

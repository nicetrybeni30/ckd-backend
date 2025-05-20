# ✅ retrain_model.py (Terminal command only — NOT the API one)
from django.core.management.base import BaseCommand
from django.conf import settings
from predictor.models import PatientRecord, ModelRetrainLog

import time, json, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback

PROGRESS_FILE = settings.BASE_DIR / 'predictor' / 'ml_model' / 'progress.json'

class Command(BaseCommand):
    help = 'Retrains the CKD model manually with 24 features via terminal (python manage.py retrain_model).'

    def handle(self, *args, **kwargs):
        model_dir = settings.BASE_DIR / 'predictor' / 'ml_model'
        model_dir.mkdir(parents=True, exist_ok=True)

        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()

        records = PatientRecord.objects.all()
        if not records.exists():
            self.stdout.write(self.style.ERROR("❌ No records to train on."))
            return

        data = pd.DataFrame(list(records.values()))
        self.stdout.write(f"📊 Total records in DB: {len(data)}")

        selected = data[[
            'age', 'bp', 'sg', 'al', 'su',
            'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane',
            'classification'
        ]].dropna()
        self.stdout.write(f"✅ Records used for training (after dropna): {len(selected)}")

        for col in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
            selected[col] = selected[col].apply(lambda x: 1 if str(x).lower() in ['yes', 'present', 'abnormal', 'poor'] else 0)

        X = selected.drop('classification', axis=1)
        y = selected['classification'].apply(lambda x: 1 if x == 'ckd' else 0)

        # Balance dataset
        full_data = pd.concat([X, y.rename("classification")], axis=1)
        ckd = full_data[full_data['classification'] == 1]
        not_ckd = full_data[full_data['classification'] == 0]
        ckd_upsampled = resample(ckd, replace=True, n_samples=len(not_ckd), random_state=42)
        balanced = pd.concat([ckd_upsampled, not_ckd])

        X = balanced.drop('classification', axis=1)
        y = balanced['classification']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.stdout.write(f"🧪 Training samples: {len(X_train)}")
        self.stdout.write(f"🧪 Test samples: {len(X_test)}")

        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

        total_epochs = 200
        start_time = time.time()

        def update_progress(epoch, logs):
            percent = round(((epoch + 1) / total_epochs) * 100, 1)
            elapsed = time.time() - start_time
            estimated_total = elapsed / ((epoch + 1) / total_epochs)
            seconds_left = int(estimated_total - elapsed)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"epoch": epoch + 1, "percent": percent, "seconds_left": seconds_left}, f)

        progress_callback = LambdaCallback(on_epoch_end=update_progress)

        model.fit(
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
            json.dump({"epoch": total_epochs, "percent": 100.0, "seconds_left": 0}, f)

        self.stdout.write(self.style.SUCCESS(f"✅ Test Accuracy: {accuracy:.2%}"))
        self.stdout.write(self.style.SUCCESS("🎉 Model retrained successfully."))
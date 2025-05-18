from django.core.management.base import BaseCommand
from django.conf import settings
from predictor.models import PatientRecord, ModelRetrainLog

import pandas as pd
import numpy as np
import time, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback

PROGRESS_FILE = settings.BASE_DIR / 'predictor' / 'ml_model' / 'progress.json'

class Command(BaseCommand):
    help = 'Manually retrain the CKD deep learning model and show progress.'

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
            'bgr', 'bu', 'sc', 'hemo', 'pcv',
            'htn', 'dm', 'classification'
        ]].dropna()
        self.stdout.write(f"✅ Records used for training (after dropna): {len(selected)}")

        selected['htn'] = selected['htn'].apply(lambda x: 1 if x == 'yes' else 0)
        selected['dm'] = selected['dm'].apply(lambda x: 1 if x == 'yes' else 0)
        selected['classification'] = selected['classification'].apply(lambda x: 1 if x == 'ckd' else 0)

        X = selected.drop('classification', axis=1)
        y = selected['classification']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.stdout.write(f"🧪 Training samples: {len(X_train)}")
        self.stdout.write(f"🧪 Test samples: {len(X_test)}")

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
            self.stdout.write(f"🌀 Epoch {epoch+1:03}/{total_epochs} ➜ acc: {logs['accuracy']:.4f}, loss: {logs['loss']:.4f}")

        callback = LambdaCallback(on_epoch_end=update_progress)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=total_epochs,
            batch_size=16,
            callbacks=[callback],
            verbose=0
        )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
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

        self.stdout.write(self.style.SUCCESS(f"✅ Test Accuracy: {accuracy:.2%}"))
        self.stdout.write(self.style.SUCCESS(f"🕒 Finished at: {log.retrained_at.strftime('%Y-%m-%d %H:%M:%S')}"))

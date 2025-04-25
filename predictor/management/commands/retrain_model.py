from django.core.management.base import BaseCommand
import os
import joblib
from datetime import datetime
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

class Command(BaseCommand):
    help = 'Retrain the CKD prediction model using XGBoost and save accuracy + timestamp.'

    def handle(self, *args, **options):
        # Path setup
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(base_dir, 'Cleaned_CKD_Dataset.csv')
        model_path = os.path.join(base_dir, 'ml_model', 'xgb_ckd_model.pkl')
        accuracy_path = os.path.join(base_dir, 'ml_model', 'model_accuracy.txt')
        retrained_at_path = os.path.join(base_dir, 'ml_model', 'retrained_at.txt')

        # Load dataset
        df = pd.read_csv(csv_path)
        df['classification'] = df['classification'].str.strip()

        # Selected features
        features = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
        required_cols = features + ['classification']
        df = df.dropna(subset=required_cols)

        for col in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
            df[col] = LabelEncoder().fit_transform(df[col])

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['classification'])

        X = df[features]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        model.label_encoder = label_encoder
        joblib.dump(model, model_path)
        self.stdout.write(self.style.SUCCESS("✅ XGBoost model retrained and saved."))

        accuracy = model.score(X_test, y_test)
        with open(accuracy_path, 'w') as f:
            f.write(str(round(accuracy, 4)))

        with open(retrained_at_path, 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        self.stdout.write(f"📊 Accuracy: {accuracy:.4f}")

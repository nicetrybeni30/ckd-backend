from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from predictor.models import PatientRecord
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

@api_view(['POST'])
@permission_classes([IsAdminUser])
def retrain_model(request):
    records = PatientRecord.objects.all()
    if not records:
        return Response({"error": "No records to train on."}, status=400)

    data = pd.DataFrame(list(records.values()))
    selected = data[[
        'age', 'bp', 'sg', 'al', 'su',
        'bgr', 'bu', 'sc', 'hemo', 'pcv',
        'htn', 'dm', 'classification'
    ]].dropna()

    selected['htn'] = selected['htn'].apply(lambda x: 1 if x == 'yes' else 0)
    selected['dm'] = selected['dm'].apply(lambda x: 1 if x == 'yes' else 0)

    X = selected.drop('classification', axis=1)
    y = selected['classification']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, 'predictor/ml_model/rf_ckd_model.pkl')
    return Response({"message": "Model retrained successfully."})

from django.urls import path
from predictor.api.predict import predict_ckd
from predictor.api.retrain import retrain_model
from predictor.views import (
    UserListView,
    RecordMeView,
    RecordCreateUpdateView,
    get_user_details,
    get_logged_in_user_info,
    get_model_info,
    RecordDetailByAdminView  # ✅ new import
)

urlpatterns = [
    path('users/', UserListView.as_view()),
    path('users/<int:user_id>/', get_user_details),
    path('records/me', RecordMeView.as_view()),
    path('records/', RecordCreateUpdateView.as_view()),
    path('predict/', predict_ckd),
    path('retrain/', retrain_model),
    path('patient/me/', get_logged_in_user_info),  # 🧑‍⚕️
    path('model-info/', get_model_info),  
    path('records/<int:user_id>/', RecordDetailByAdminView.as_view())     # 📊
]

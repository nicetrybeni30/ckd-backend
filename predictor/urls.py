from django.urls import path
from predictor.api.predict import predict_ckd
from predictor.api.retrain import retrain_model
from predictor.views import retrain_progress
from predictor.views import (
    UserListView,
    RecordMeView,
    RecordCreateUpdateView,
    get_user_details,
    get_logged_in_user_info,
    get_model_info,
    RecordDetailByAdminView,
    patient_register,
    UpdateAccountView,
    get_retrain_history,
    get_ckd_distribution,
    users_with_records,
    get_record_by_user,
    retrain_progress  # ✅ added this
)

urlpatterns = [
    path('users/', UserListView.as_view()),
    path('users/<int:user_id>/', get_user_details, name='user-detail'),
    path('records/me/', RecordMeView.as_view()),
    path('records/', RecordCreateUpdateView.as_view()),
    path('records/<int:user_id>/', RecordDetailByAdminView.as_view()),
    path('records/by_user/<int:user_id>/', get_record_by_user), 
    path('predict/', predict_ckd),
    path('retrain/', retrain_model),
    path('retrain_progress/', retrain_progress), 
    path('patient/me/', get_logged_in_user_info),
    path('model-info/', get_model_info),
    path('register/', patient_register),
    path('account/update/', UpdateAccountView.as_view()),
    path('retrain-history/', get_retrain_history),
    path('ckd-distribution/', get_ckd_distribution),
    path('users-with-records/', users_with_records),
]

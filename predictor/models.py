from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('patient', 'Patient'),
    ]

    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=15, null=True, blank=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='patient')

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']

    def __str__(self):
        return self.username

class PatientRecord(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.FloatField()
    bp = models.FloatField()
    sg = models.FloatField()
    al = models.FloatField()
    su = models.FloatField()
    rbc = models.CharField(max_length=20)
    pc = models.CharField(max_length=20)
    pcc = models.CharField(max_length=20)
    ba = models.CharField(max_length=20)
    bgr = models.FloatField()
    bu = models.FloatField()
    sc = models.FloatField()
    sod = models.FloatField()
    pot = models.FloatField()
    hemo = models.FloatField()
    pcv = models.FloatField()
    wc = models.FloatField()
    rc = models.FloatField()
    htn = models.CharField(max_length=10)
    dm = models.CharField(max_length=10)
    cad = models.CharField(max_length=10)
    appet = models.CharField(max_length=10)
    pe = models.CharField(max_length=10)
    ane = models.CharField(max_length=10)
    classification = models.CharField(max_length=10)
    smoker = models.CharField(max_length=10)
    ckd_stage = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    last_prediction = models.CharField(max_length=10, null=True, blank=True)
    last_recommendation = models.TextField(null=True, blank=True)
    last_confidence = models.FloatField(null=True, blank=True)
    last_predicted_at = models.DateTimeField(null=True, blank=True)

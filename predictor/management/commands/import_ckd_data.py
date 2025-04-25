import csv
from django.core.management.base import BaseCommand
from predictor.models import User, PatientRecord

class Command(BaseCommand):
    help = 'Import CKD data from a CSV file'

    def handle(self, *args, **kwargs):
        file_path = 'CKD_Dataset_With_Users.csv'

        # Delete old data first
        PatientRecord.objects.all().delete()
        User.objects.exclude(is_superuser=True).delete()

        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for idx, row in enumerate(reader):
                username = f"patient{idx+1}"
                email = f"{username}@email.com"
                password = username

                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                    is_admin=False
                )

                PatientRecord.objects.create(
                    user=user,
                    age=float(row.get('age') or 0),
                    bp=float(row.get('bp') or 0),
                    sg=float(row.get('sg') or 0),
                    al=float(row.get('al') or 0),
                    su=float(row.get('su') or 0),
                    rbc=row.get('rbc') or 'normal',
                    pc=row.get('pc') or 'normal',
                    pcc=row.get('pcc') or 'notpresent',
                    ba=row.get('ba') or 'notpresent',
                    bgr=float(row.get('bgr') or 0),
                    bu=float(row.get('bu') or 0),
                    sc=float(row.get('sc') or 0),
                    sod=float(row.get('sod') or 0),
                    pot=float(row.get('pot') or 0),
                    hemo=float(row.get('hemo') or 0),
                    pcv=float(row.get('pcv') or 0),
                    wc=float(row.get('wc') or 0),
                    rc=float(row.get('rc') or 0),
                    htn=row.get('htn') or 'no',
                    dm=row.get('dm') or 'no',
                    cad=row.get('cad') or 'no',
                    appet=row.get('appet') or 'good',
                    pe=row.get('pe') or 'no',
                    ane=row.get('ane') or 'no',
                    classification=row.get('classification') or 'ckd',
                    smoker='no',  # CSV doesn't include this
                    ckd_stage=row.get('ckd_stage') or 'early'
                )

        self.stdout.write(self.style.SUCCESS("✅ Successfully re-imported all patient data"))

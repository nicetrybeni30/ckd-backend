# Chronic Kidney Disease (CKD) Prediction System

This project is composed of:

- **Backend**: Django REST Framework (Python)
- **Frontend**: Vue.js 3 + Tailwind CSS (Vite)

---

## 🛠 Requirements

- Python 3.10+
- Node.js 18+ and NPM
- Virtualenv (`python -m venv venv`)

---

## ⚙️ Backend Setup (ckd-backend)

```bash
Step 1: Clone the Repository
git clone git@github.com:nicetrybeni30/ckd-backend.git
cd ckd-backend

Step 2: Create and Activate Virtual Environment
# Windows:
python -m venv venv
venv\Scripts\activate

# Linux/macOS:
python3 -m venv venv
source venv/bin/activate

Step 3: Install Python Dependencies
pip install -r requirements.txt

Step 4: Apply Database Migrations
python manage.py makemigrations
python manage.py migrate

Step 5: Run the Backend Server
python manage.py runserver

✅ Your Django API will now run on http://localhost:8000

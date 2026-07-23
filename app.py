from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# إنشاء التطبيق
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Predict Breast Cancer using Machine Learning",
    version="1.0"
)

# تحميل الموديل
model = joblib.load("breast_cancer_model.pkl")
# لو اسم الملف model.pkl اكتبي:
# model = joblib.load("model.pkl")


# شكل البيانات اللي المستخدم هيبعتها
class CancerData(BaseModel):
    features: list[float]


# الصفحة الرئيسية
@app.get("/")
def home():
    return {
        "message": "Breast Cancer Prediction API is Running!"
    }


# التنبؤ
@app.post("/predict")
def predict(data: CancerData):

    # التأكد إن عدد الـ Features = 30
    if len(data.features) != 30:
        return {
            "error": "You must send exactly 30 features."
        }

    # تحويل البيانات لـ NumPy Array
    input_data = np.array(data.features).reshape(1, -1)

    # التنبؤ
    prediction = model.predict(input_data)[0]

    # تحويل الرقم إلى نص
    if prediction == 0:
        result = "Malignant"
    else:
        result = "Benign"

    return {
        "prediction": result
    }
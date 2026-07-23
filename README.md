# Breast Cancer Prediction API

## Project Overview

This project uses a Machine Learning model to predict whether a breast tumor is Benign or Malignant.

## Technologies Used

- Python
- Scikit-learn
- FastAPI
- NumPy
- Pandas
- Joblib

## Project Structure

```
Breast-Cancer-API/
│
├── app.py
├── train.py
├── breast_cancer_model.pkl
├── requirements.txt
├── sample_request.json
├── README.md
└── .gitignore
```

## Run the API

```bash
uvicorn app:app --reload
```

## Open Swagger

```
http://127.0.0.1:8000/docs
```

## Model Performance

Accuracy: **96%**
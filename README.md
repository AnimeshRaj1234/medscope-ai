# MedScope AI — Multi-Disease Prediction System

An educational machine learning web application that demonstrates four independent disease-classification models through an interactive Flask interface.

**Live Demo:** https://medscope-ai-r08v.onrender.com/  
**GitHub Repository:** https://github.com/AnimeshRaj1234/medscope-ai

## Overview

MedScope AI brings four machine learning classification systems together in one web application. Users can enter structured medical measurements and receive predictions from the corresponding trained model.

The project demonstrates an end-to-end machine learning workflow, from data preprocessing and model training to backend integration, frontend interaction, and cloud deployment.

> **Disclaimer:** MedScope AI is an educational project, not a medical device. It is not clinically validated and must not be used for diagnosis, treatment, or medical decision-making. Always consult a qualified healthcare professional.

## Supported Prediction Modules

| Module | Dataset / Source | Model |
|---|---|---|
| Heart Disease | Heart disease dataset | Random Forest pipeline |
| Diabetes | Pima Indians Diabetes dataset | Random Forest pipeline |
| Chronic Kidney Disease | UCI CKD dataset | Random Forest pipeline |
| Breast Cancer | Wisconsin Diagnostic Breast Cancer dataset | Random Forest pipeline |

## Technology Stack

**Machine Learning**
- Python
- Scikit-learn
- Pandas
- NumPy
- Random Forest
- Scikit-learn Pipelines

**Backend**
- Flask
- REST API
- Joblib

**Frontend**
- HTML5
- CSS3
- JavaScript
- Fetch API

**Deployment**
- Git and GitHub
- Render
- Gunicorn

## System Architecture

```text
              USER
               |
               v
       Web Interface
       HTML / CSS / JS
               |
               v
         Flask Backend
               |
               v
       Input Validation
               |
               v
       Disease Selection
               |
       +-------+-------+----------------+
       |       |       |                |
       v       v       v                v
     Heart  Diabetes  Kidney       Breast Cancer
     Model   Model    Model           Model
       |       |       |                |
       +-------+-------+----------------+
                       |
                       v
              Prediction Result
                       |
                       v
              Frontend Display
```

## Model Evaluation

The following are recorded evaluation results from the project's model experiments.

| Disease | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|
| Heart Disease | 80.26% | 80.03% | 86.55% |
| Diabetes | 73.44% | 72.33% | 82.89% |
| Kidney Disease | 100% holdout | 99.13% OOF CV | 99.99% OOF CV |
| Breast Cancer | 97.20% | 96.23% | 99.77% |

**Note:** Kidney Disease's cross-validation metrics are reported separately from its holdout results. These are benchmark evaluation results, not evidence of clinical performance or generalization to real-world patients.

## Application Features

- Four independent disease prediction modules.
- Dynamic input forms generated from model feature schemas.
- Backend input validation.
- Pre-trained machine learning pipelines loaded using Joblib.
- Prediction results displayed through an interactive interface.
- Model evaluation metrics displayed alongside predictions.
- Responsive web interface.
- Public deployment using Render.

## Project Structure

```text
DESI AI DOCTOR/
│
├── app.py
├── train_models.py
├── validate_models.py
├── diagnose_data.py
├── smoke_test.py
├── requirements.txt
├── render.yaml
├── README.md
│
├── app/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── models/
│   ├── heart.joblib
│   ├── diabetes.joblib
│   ├── kidney.joblib
│   └── breast_cancer.joblib
│
├── data/
├── datasets/
└── notebooks/
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AnimeshRaj1234/medscope-ai.git
cd medscope-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The repository includes the trained model artifacts, so model training is not required for the normal application startup.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Load the web application |
| `/health` | GET | Application health check |
| `/api/schema/<disease>` | GET | Retrieve the selected model's input schema |
| `/api/predict/<disease>` | POST | Generate a prediction |

## Deployment

The application is deployed on **Render** using Gunicorn as the production WSGI server.

Live application:

https://medscope-ai-r08v.onrender.com/

## Limitations and Future Improvements

- The models are trained on relatively small public benchmark datasets.
- Dataset bias and class imbalance can affect predictions.
- The models have not undergone external clinical validation.
- Model scores should not be interpreted as calibrated personal disease risks.

Possible future improvements:
- Add model explainability using SHAP.
- Improve model evaluation and calibration.
- Add automated testing and continuous integration.
- Improve accessibility and user experience.
- Explore additional validated datasets.

## Author

**Animesh Raj**  
Electrical Engineering Student  
National Institute of Technology, Jalandhar

GitHub: https://github.com/AnimeshRaj1234

---

*Built as an educational machine learning and web development project.*


from flask import Flask, request, jsonify, render_template
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer


# ==================================================
# APPLICATION CONFIGURATION
# ==================================================

app = Flask(
    __name__,
    template_folder='app/templates',
    static_folder='app/static'
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'models'


# ==================================================
# DISEASE CONFIGURATION
# ==================================================

DISEASES = {

    # ---------------- HEART DISEASE ----------------

    'heart': {
        'title': 'Heart Disease',
        'description': 'Clinical measurements from the heart disease benchmark dataset.',
        'fields': [
            ('age', 'Age', 'number', 29, 77, 54),
            ('sex', 'Sex (1 = male, 0 = female)', 'select', [0, 1], None, 1),
            ('cp', 'Chest pain type (0–3)', 'number', 0, 3, 1),
            ('trestbps', 'Resting blood pressure (mm Hg)', 'number', 80, 220, 130),
            ('chol', 'Serum cholesterol (mg/dL)', 'number', 100, 600, 240),
            ('fbs', 'Fasting blood sugar > 120 mg/dL (1 = yes)', 'select', [0, 1], None, 0),
            ('restecg', 'Resting ECG (0–2)', 'number', 0, 2, 1),
            ('thalach', 'Maximum heart rate achieved', 'number', 60, 220, 150),
            ('exang', 'Exercise-induced angina (1 = yes)', 'select', [0, 1], None, 0),
            ('oldpeak', 'ST depression', 'number', 0, 7, 1.0),
            ('slope', 'ST slope (0–2)', 'number', 0, 2, 1),
            ('ca', 'Major vessels (0–4)', 'number', 0, 4, 0),
            ('thal', 'Thal (0–3)', 'number', 0, 3, 2)
        ]
    },

    # ---------------- DIABETES ----------------

    'diabetes': {
        'title': 'Diabetes',
        'description': 'Pima benchmark variables. This dataset has a limited population and is not universally representative.',
        'fields': [
            ('Pregnancies', 'Pregnancies', 'number', 0, 20, 2),
            ('Glucose', 'Glucose (mg/dL)', 'number', 0, 250, 120),
            ('BloodPressure', 'Diastolic blood pressure (mm Hg)', 'number', 0, 150, 70),
            ('SkinThickness', 'Triceps skinfold (mm)', 'number', 0, 100, 20),
            ('Insulin', '2-hour serum insulin (mu U/mL)', 'number', 0, 900, 80),
            ('BMI', 'Body mass index', 'number', 0, 80, 28),
            ('DiabetesPedigreeFunction', 'Diabetes pedigree function', 'number', 0, 3, 0.5),
            ('Age', 'Age (years)', 'number', 18, 100, 35)
        ]
    },

    # ---------------- KIDNEY DISEASE ----------------

    'kidney': {
        'title': 'Chronic Kidney Disease',
        'description': 'UCI CKD benchmark features. Values should come from clinical laboratory testing.',
        'fields': [
            ('age', 'Age (years)', 'number', 1, 100, 45),
            ('bp', 'Blood pressure (mm/Hg)', 'number', 40, 200, 80),
            ('sg', 'Specific gravity', 'select', ['1.005', '1.010', '1.015', '1.020', '1.025'], None, '1.020'),
            ('al', 'Urine albumin (0–5)', 'number', 0, 5, 0),
            ('su', 'Urine sugar (0–5)', 'number', 0, 5, 0),
            ('rbc', 'Red blood cells', 'select', ['normal', 'abnormal'], None, 'normal'),
            ('pc', 'Pus cells', 'select', ['normal', 'abnormal'], None, 'normal'),
            ('pcc', 'Pus cell clumps', 'select', ['notpresent', 'present'], None, 'notpresent'),
            ('ba', 'Bacteria', 'select', ['notpresent', 'present'], None, 'notpresent'),
            ('bgr', 'Random blood glucose (mg/dL)', 'number', 40, 500, 120),
            ('bu', 'Blood urea (mg/dL)', 'number', 1, 400, 35),
            ('sc', 'Serum creatinine (mg/dL)', 'number', 0.1, 30, 1.0),
            ('sod', 'Sodium (mEq/L)', 'number', 80, 180, 140),
            ('pot', 'Potassium (mEq/L)', 'number', 1, 12, 4.5),
            ('hemo', 'Hemoglobin (g/dL)', 'number', 2, 25, 14),
            ('pcv', 'Packed cell volume', 'number', 10, 70, 42),
            ('wc', 'White blood cell count', 'number', 1000, 30000, 8000),
            ('rc', 'Red blood cell count (millions/cmm)', 'number', 1, 10, 5),
            ('htn', 'Hypertension', 'select', ['no', 'yes'], None, 'no'),
            ('dm', 'Diabetes mellitus', 'select', ['no', 'yes'], None, 'no'),
            ('cad', 'Coronary artery disease', 'select', ['no', 'yes'], None, 'no'),
            ('appet', 'Appetite', 'select', ['good', 'poor'], None, 'good'),
            ('pe', 'Pedal edema', 'select', ['no', 'yes'], None, 'no'),
            ('ane', 'Anemia', 'select', ['no', 'yes'], None, 'no')
        ]
    },

    # ---------------- BREAST CANCER ----------------

    'breast_cancer': {
        'title': 'Breast Tumor Classification',
        'description': 'Wisconsin Diagnostic dataset. Enter 30 cell-nucleus measurements; values are not ordinary symptoms.',
        'fields': []
    }

}


# ==================================================
# BREAST CANCER FEATURE GENERATION
# ==================================================

bc = load_breast_cancer(as_frame=True)

for feature in bc.data.columns:

    values = bc.data[feature]

    minimum = float(values.min())
    maximum = float(values.max() * 1.5)
    default = float(values.median())

    label = feature.title()

    DISEASES['breast_cancer']['fields'].append(
        (feature, label, 'number', minimum, maximum, default)
    )


# ==================================================
# HOME ROUTE
# ==================================================

@app.get('/')
def home():

    return render_template(
        'index.html',
        diseases=DISEASES
    )


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get('/health')
def health():

    return jsonify({
        'status': 'ok',
        'models': {
            key: (MODEL_DIR / f'{key}.joblib').exists()
            for key in DISEASES
        }
    })


# ==================================================
# DISEASE SCHEMA API
# ==================================================

@app.get('/api/schema/<disease>')
def schema(disease):

    if disease not in DISEASES:
        return jsonify(error='Unknown disease'), 404

    return jsonify(DISEASES[disease])


# ==================================================
# PREDICTION API
# ==================================================

@app.post('/api/predict/<disease>')
def predict(disease):

    # Check disease name.
    if disease not in DISEASES:
        return jsonify(error='Unknown disease'), 404

    # Check model availability.
    path = MODEL_DIR / f'{disease}.joblib'

    if not path.exists():
        return jsonify(error='Model is not trained yet.'), 503

    # Read JSON payload.
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify(error='Request must contain valid JSON.'), 400

    try:

        # Load trained model artifact.
        bundle = joblib.load(path)

        features = bundle['features']

        vals = {}

        # ------------------------------------------
        # INPUT VALIDATION
        # ------------------------------------------

        for field in DISEASES[disease]['fields']:

            key, label, field_type, minimum, maximum, default = field

            if key not in payload:
                return jsonify(
                    error=f'Missing input: {key}'
                ), 400

            value = payload[key]

            # --------------------------------------
            # NUMERIC INPUT VALIDATION
            # --------------------------------------

            if field_type == 'number':

                try:
                    value = float(value)

                except (ValueError, TypeError):
                    return jsonify(
                        error=f'{key} must be numeric.'
                    ), 400

                if not np.isfinite(value):
                    return jsonify(
                        error=f'{key} must be finite.'
                    ), 400

                if value < minimum or value > maximum:
                    return jsonify(
                        error=f'{key} must be between {minimum} and {maximum}.'
                    ), 400

            # --------------------------------------
            # SELECT INPUT VALIDATION
            # --------------------------------------

            elif field_type == 'select':

                # Numeric dropdowns:
                # Heart disease: sex, fbs, exang.

                if all(
                    isinstance(option, (int, float))
                    for option in minimum
                ):

                    try:
                        numeric_value = float(value)

                    except (ValueError, TypeError):
                        return jsonify(
                            error=f'Invalid value for {key}.'
                        ), 400

                    if (
                        not np.isfinite(numeric_value)
                        or numeric_value not in minimum
                    ):
                        return jsonify(
                            error=f'Invalid value for {key}.'
                        ), 400

                    value = (
                        int(numeric_value)
                        if numeric_value.is_integer()
                        else numeric_value
                    )

                # Categorical dropdowns:
                # Kidney disease: normal, abnormal, yes, no, etc.

                else:

                    value = str(value)

                    allowed_values = [
                        str(option)
                        for option in minimum
                    ]

                    if value not in allowed_values:
                        return jsonify(
                            error=f'Invalid value for {key}.'
                        ), 400

            # Store validated value.
            vals[key] = value

        # ------------------------------------------
        # FEATURE VALIDATION
        # ------------------------------------------

        if set(features) != set(vals):

            return jsonify(
                error='Input fields do not match the trained model.'
            ), 400

        # Preserve exact training feature order.
        X = pd.DataFrame(
            [[vals[key] for key in features]],
            columns=features
        )

        # ------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------

        pipeline = bundle['pipeline']

        pred = int(
            pipeline.predict(X)[0]
        )

        # Get probability for predicted class.
        probs = pipeline.predict_proba(X)[0]

        class_index = list(pipeline.classes_).index(pred)

        model_score = float(probs[class_index]) * 100

        # ------------------------------------------
        # PREDICTION LABELS
        # ------------------------------------------

        labels = {

            'heart': {
                0: 'Negative-class pattern',
                1: 'Heart-disease-class pattern'
            },

            'diabetes': {
                0: 'Negative-class pattern',
                1: 'Diabetes-class pattern'
            },

            'kidney': {
                0: 'Not-CKD-class pattern',
                1: 'CKD-class pattern'
            },

            'breast_cancer': {
                0: 'Benign-class pattern',
                1: 'Malignant-class pattern'
            }

        }

        # ------------------------------------------
        # API RESPONSE
        # ------------------------------------------

        return jsonify(

            disease=disease,

            prediction=pred,

            label=labels[disease][pred],

            model_score=round(model_score, 1),

            metrics=bundle['metrics'],

            disclaimer=(
                'Educational demonstration only. This output is not a '
                'diagnosis or clinical risk estimate. Consult a qualified '
                'healthcare professional for medical advice.'
            )

        )

    except Exception:

        app.logger.exception(
            'Prediction failed for %s',
            disease
        )

        return jsonify(
            error='Unable to process this prediction. Check the submitted inputs.'
        ), 400


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

# MedScope AI — Multi-Disease Prediction (Educational Demo)

Flask application with five independent scikit-learn pipelines: heart disease, diabetes, chronic kidney disease, Wisconsin breast-tumor classification, and Parkinson's voice-measurement classification.

## Local run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
python app.py
```
Open http://127.0.0.1:5000.

## Public deployment on Render

1. Push this repository to a public GitHub repository.
2. In Render, choose **New + → Blueprint**, connect the repository, and approve the `render.yaml` configuration.
3. Render installs dependencies and trains the five models during the build. This can take several minutes.
4. Open the generated `https://....onrender.com` URL.

The free service may sleep when idle; its first request after sleeping can be slow. Model files are generated during the build and stored in the deployed service image.

## Data sources

- Heart: uploaded project dataset (`data/heart.csv`), based on the original project.
- Diabetes: Pima Indians Diabetes benchmark CSV from Hugging Face dataset `khoaguin/pima-indians-diabetes-database`.
- Kidney: CKD benchmark CSV mirror of the UCI Chronic Kidney Disease dataset.
- Breast tumor: Wisconsin Diagnostic Breast Cancer dataset distributed by scikit-learn.
- Parkinson's: Oxford Parkinson's voice dataset from UCI, mirrored on Hugging Face.

See the respective dataset pages and original UCI records for provenance and citation. Small benchmark datasets have important sampling and generalization limitations.

## Important medical limitation

This is a student educational demonstration. It is not clinically validated, does not diagnose disease, and must not guide treatment or care decisions. The displayed confidence is the classifier's maximum class probability, not a calibrated personal disease risk. The small Pima and Parkinson's datasets are particularly limited; the Parkinson's benchmark contains repeated recordings from a small number of subjects. The project uses a subject-grouped holdout split for Parkinson's evaluation.

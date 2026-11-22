# Dashboard guide

The Streamlit dashboard (`dashboard/`) explores the dataset and serves the
saved models. Run it from the repo root:

```bash
streamlit run dashboard/app.py
```

## Pages

- **Home** (`home.py`) - the workspace landing view: subject performance at a
  glance and pointers into the analysis pages.
- **Overview** (`pages/1_Overview.py`) - dataset at a glance: academic
  performance, the final-grade (G3) distribution, and a preview of the raw
  rows.
- **Student Analysis** (`pages/2_Student_Analysis.py`) - per-student view:
  academic snapshot and performance, student characteristics, and attendance
  analysis.
- **Performance Prediction** (`pages/3_Performance_Prediction.py`) - enter
  student information and get the regression model's estimate of the final
  grade (G3, 0-20), with the pass/fail message derived from that estimate.
- **Risk Analysis** (`pages/4_Risk_Analysis.py`) - enter student information
  and get the classification model's at-risk probability (G3 below 10), plus
  a dataset-wide risk overview.
- **Model Performance** (`pages/5_Model_Performance.py`) - regression and
  classification evaluation numbers, evaluation visualizations, and SHAP
  explanations of what drives the models.

## How predictions are wired

Both prediction pages call the shared module `src/models/prediction.py`:

- `predict_final_grade` - regression estimate of the final grade
- `predict_at_risk` - classification at-risk flag and probability

so the app and the training pipeline can never drift apart.

## Modes

Every task runs in two modes:

- **early** - background and behavior data only, before any exam grades exist
- **later** - adds the period grades G1 and G2

The mode matters: later-mode models are far more reliable (see the results
tables in the README).

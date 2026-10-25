# Customer Churn Prediction

### From customer data to an interactive churn prediction app

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-Logistic_Regression-F7931E?logo=scikitlearn&logoColor=white)
![Interface](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

An educational machine learning project that estimates whether a telecom customer will leave a service. It brings together exploratory analysis, data preparation, logistic regression, model evaluation, and an interactive **Streamlit** interface.

Enter a customer's account, service, and billing details to receive a predicted churn probability and a binary prediction.

**[Quick start](#quick-start) · [Explore the notebook](#explore-the-notebook) · [Results](#baseline-results) · [Project files](#project-files)**

## Overview

Customer churn means a customer stops using a company's service. Predicting churn can help teams explore retention strategies and understand which account characteristics are associated with customers leaving.

This repository demonstrates the complete workflow on the Telco Customer Churn dataset:

1. Inspect the dataset and explore the target distribution.
2. Clean numeric values and encode categorical features.
3. Train a logistic regression classifier with a stratified train/test split.
4. Evaluate discrimination and classification performance.
5. Save the model together with its feature metadata.
6. Generate predictions from Python or a browser-based form.

## Features

- **Exploratory notebook:** dataset inspection, churn distribution, ROC curve, and coefficient analysis.
- **Repeatable training entry point:** train and export the baseline with one command.
- **Interactive predictions:** enter customer features through a Streamlit form.
- **Reusable inference functions:** validate input with Pydantic and align it to the training columns.
- **Portable file paths:** run the scripts on macOS, Windows, or Linux.
- **Included model and dataset:** try the demonstration without first training a model.

## Quick start

Use **Python 3.11 or later**. The commands below are for macOS or Linux; run them from Terminal or the VS Code terminal.

### 1. Get the project

After this repository is published:

```bash
git clone https://github.com/KingaTshering10/customer-churn-prediction.git
cd customer-churn-prediction
```

If you downloaded a ZIP instead, extract it and open the `customer-churn-prediction` folder in VS Code. Run the remaining commands from that folder.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use `py -m venv .venv` followed by `.venv\Scripts\Activate.ps1`.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch the app

```bash
python -m streamlit run churn_streamlit_app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Enter customer details and click **Generate prediction**.

| Output | Meaning |
| --- | --- |
| Churn probability | Model estimate of the probability that the customer belongs to the churn class |
| Label `1` | Predicted to churn |
| Label `0` | Predicted to stay |

The default logistic regression decision threshold is 0.5. The estimate is a model output, rather than a guarantee about an individual customer.

## Train the model

```bash
python train_churn.py
```

The script loads the included CSV, applies the baseline preprocessing, prints evaluation metrics, and replaces `churn_logistic_model.pkl` with a newly trained artifact. It uses an 80/20 stratified split with `random_state=42` and `LogisticRegression(max_iter=1000, solver="lbfgs")`.

The saved artifact contains:

| Key | Contents |
| --- | --- |
| `model` | Fitted logistic regression estimator |
| `feature_names` | Ordered encoded feature names used during training |
| `cat_cols` | Original categorical columns used for one-hot encoding |

The supplied model records **scikit-learn 1.9.1**, which is pinned in `requirements.txt`. If you use a different scikit-learn version, retrain in that environment before inference. Model compatibility across scikit-learn versions is not supported; see the [model persistence documentation](https://scikit-learn.org/stable/model_persistence.html).

Restart Streamlit after retraining so its cached model is refreshed.

## Run a prediction from Python

```bash
python predict_churn.py
```

This runs the example customer included in the script and prints the predicted label and probability. To use the functions in another program:

```python
from predict_churn import load_artifact, predict_from_dict

model, feature_names, cat_cols = load_artifact()

# customer is a dictionary containing the fields shown in predict_churn.py.
label, probability = predict_from_dict(
    customer, model, feature_names, cat_cols
)
```

For a single customer, categorical values are encoded without dropping the observed category, then aligned to the saved training features. This preserves the customer's categorical information while excluding the training reference columns.

## Explore the notebook

Open `churn_logistic.ipynb` in VS Code with the Python and Jupyter extensions, and select the `.venv` interpreter as the notebook kernel. Install the kernel package if prompted:

```bash
python -m pip install ipykernel
```

Alternatively, use JupyterLab:

```bash
python -m pip install jupyterlab
python -m jupyterlab
```

Launch JupyterLab from the project folder. Run the notebook from top to bottom to reproduce the analysis, inspect ROC and coefficient plots, and save the model. Saved outputs were cleared in this repository copy so example predictions are regenerated with the corrected encoding.

## Dataset

The included file is `WA_Fn-UseC_-Telco-Customer-Churn.csv`, a copy of the widely used **IBM Telco Customer Churn** sample dataset. See [IBM's reference repository](https://github.com/IBM/telco-customer-churn-on-icp4d) for the upstream example and dataset context.

| Property | Value in the supplied CSV |
| --- | --- |
| Original records | 7,043 |
| Original columns | 21 |
| Prediction target | `Churn` (`Yes` / `No`) |
| Records after cleaning | 7,032 |
| Encoded input features | 30 |
| Training / test records | 5,625 / 1,407 |

Preprocessing converts `TotalCharges` to numeric, removes 11 rows with missing values after conversion, drops `customerID`, maps the target to 0/1, and one-hot encodes categorical inputs.

Inputs describe customer demographics, service subscriptions, tenure, contract type, payment method, and monthly and total charges. The baseline derives the categorical feature vocabulary before splitting the data; a stricter evaluation should fit preprocessing only on the training subset.

## Baseline results

The following values are **historical results recorded in the supplied notebook**, rather than a new benchmark for the cleaned repository. Re-running with different dependency versions may produce different values.

| Metric | Recorded value |
| --- | ---: |
| Accuracy | 80.31% |
| Precision — churn class | 64.56% |
| Recall — churn class | 57.49% |
| F1 — churn class | 60.82% |
| ROC AUC | 0.8365 |

Recorded confusion matrix:

| Actual / predicted | Stay (`0`) | Churn (`1`) |
| --- | ---: | ---: |
| Stay (`0`) | 915 | 118 |
| Churn (`1`) | 159 | 215 |

The original training output includes a convergence warning at 1,000 iterations. These numbers describe that baseline run, not a fully optimized model. The correction to single-customer encoding affects the example and app predictions; the recorded batch evaluation used the full encoded feature matrix.

## Project files

| File | Purpose |
| --- | --- |
| `churn_logistic.ipynb` | Analysis, training, evaluation plots, and model export |
| `train_churn.py` | Command-line baseline training and model export |
| `predict_churn.py` | Input schema, model loading, preprocessing, and inference |
| `churn_streamlit_app.py` | Interactive prediction interface |
| `churn_logistic_model.pkl` | Supplied model and feature metadata |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset used by the demonstration |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes environments, caches, temporary files, and local secrets |
| `docs/summary.docx` | Original training/lab summary; its environment commands are Windows-oriented |

## Limitations and next steps

This is a learning project. Its evaluation uses one holdout split, and the class distribution is imbalanced. The form validates numeric types but does not enforce all categorical choices or consistency between service selections. Inputs outside the dataset's categories need additional handling.

Useful extensions include fitting preprocessing inside a scikit-learn pipeline, scaling numeric features, resolving convergence, cross-validation, threshold selection, probability calibration, and comparison with other classifiers. Feature coefficients describe associations and should not be interpreted as causal effects.

## Acknowledgments

Maintained by **[Kinga Tshering](https://github.com/KingaTshering10)**. Prepared from the supplied customer-churn training materials, retaining the original analysis structure, dataset, model artifact, and lab summary. Dataset credit belongs to its original providers.

The repository copy adds portable model paths, corrected single-customer encoding, a command-line training script, and documentation. No project license was included in the supplied archive.

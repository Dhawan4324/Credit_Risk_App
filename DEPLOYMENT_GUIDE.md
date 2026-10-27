# Explainable Credit Risk Assessment

## Start locally

Install requirements:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Streamlit Community Cloud

1. Upload this application folder to a GitHub repository.
2. Open https://share.streamlit.io/
3. Select the repository and main file app.py.
4. Deploy and check the application logs.

## Research configuration

Paper: Paper_4_Credit_Risk

Input features: 20

Target classes: 2

Decision threshold: 0.3

## Safety and interpretation

This interface is a research demonstration.

The model is frozen and is not retrained on user inputs.

Model files are trusted, previously trained joblib artifacts.
Never load untrusted joblib files.

Model predictions are not causal explanations.

The credit risk application is not designed for
automated real-world lending decisions.

## Validation

See the root-level Step18_Validation_Details.csv for the
exact tests performed, their outcomes, and any failures.

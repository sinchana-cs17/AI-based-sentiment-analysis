# Sentiment Analysis for Customer Feedback

## Features
- NLP pipeline using TF-IDF and Logistic Regression
- Positive, negative, and neutral sentiment labels
- Interactive Streamlit dashboard
- CSV upload and classified CSV download

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run app.py`.
6. Open the local URL shown in the terminal.

## CSV format
Upload a CSV with a column named `feedback`.

## Important
The built-in training dataset is tiny and intended only for demonstration. Model predictions can be wrong and must not be treated as reliable business conclusions without evaluation on representative data.

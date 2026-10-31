import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
st.title("💬 Customer Feedback Sentiment Dashboard")
st.caption("Demo project using sample data. Predictions are model estimates, not ground truth.")

@st.cache_resource
def train_model():
    examples = [
        ("I love this product, it works perfectly", "positive"),
        ("Excellent service and quick delivery", "positive"),
        ("Very happy with my purchase", "positive"),
        ("The quality is fantastic", "positive"),
        ("Support was helpful and friendly", "positive"),
        ("This is wonderful and easy to use", "positive"),
        ("I am disappointed with this product", "negative"),
        ("Terrible service, very slow delivery", "negative"),
        ("It stopped working after one day", "negative"),
        ("Poor quality and bad experience", "negative"),
        ("Customer support did not help me", "negative"),
        ("I want a refund, this is awful", "negative"),
        ("The product arrived today", "neutral"),
        ("It is an ordinary product", "neutral"),
        ("My order has been received", "neutral"),
        ("The package contains two items", "neutral"),
        ("I contacted support about my order", "neutral"),
        ("The delivery was scheduled for Monday", "neutral"),
    ]
    x, y = zip(*examples)
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2))), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(x, y)
    return model, pd.DataFrame(examples, columns=["feedback", "sentiment"])
model, training_data = train_model()
st.sidebar.header("Analyze feedback")
feedback = st.sidebar.text_area("Enter customer feedback", "The delivery was fast and the product is excellent.")
if st.sidebar.button("Analyze", type="primary"):
    pred = model.predict([feedback])[0]
    probs = model.predict_proba([feedback])[0]
    st.sidebar.success(f"Predicted sentiment: {pred.title()}")
    st.sidebar.write({c: f"{p:.1%}" for c,p in zip(model.classes_, probs)})

uploaded = st.file_uploader("Upload a CSV with a 'feedback' column (optional)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    if "feedback" not in df.columns:
        st.error("CSV must contain a 'feedback' column.")
        st.stop()
else:
    df = pd.DataFrame({"feedback": [
        "Amazing product and helpful support", "I am unhappy with the delayed delivery",
        "My order arrived this morning", "The quality is excellent", "This was a terrible experience",
        "The package has two items", "Very friendly customer service", "The app keeps crashing",
        "Delivery is scheduled for Friday", "I am satisfied with the purchase"
    ]})
df["feedback"] = df["feedback"].fillna("").astype(str)
df["predicted_sentiment"] = model.predict(df["feedback"])
a,b,c = st.columns(3)
counts = df["predicted_sentiment"].value_counts()
a.metric("Feedback records", len(df))
b.metric("Positive", int(counts.get("positive",0)))
c.metric("Negative", int(counts.get("negative",0)))
left,right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    st.bar_chart(counts)
with right:
    st.subheader("Classified feedback")
    st.dataframe(df, use_container_width=True)
st.download_button("Download classified CSV", df.to_csv(index=False).encode(), "classified_feedback.csv", "text/csv")
with st.expander("Training sample and limitations"):
    st.dataframe(training_data)
    st.write("This compact model is for learning/demo purposes. Real deployment needs a larger, representative, reviewed dataset.")

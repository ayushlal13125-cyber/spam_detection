import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-header{
    text-align:center;
    padding:10px;
}

.result-spam{
    background-color:#ffebee;
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#d32f2f;
}

.result-ham{
    background-color:#e8f5e9;
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#2e7d32;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    return df

try:
    df = load_data()

except Exception as e:
    st.error(f"Error loading spam.csv : {e}")
    st.stop()

# ==========================================
# CLEAN FUNCTION
# ==========================================

def clean(text):
    text = str(text).lower()

    text = ''.join(
        ch for ch in text
        if ch not in string.punctuation
    )

    text = ''.join(
        ch for ch in text
        if not ch.isdigit()
    )

    return text

df["clean_text"] = df["text"].apply(clean)

# ==========================================
# MODEL TRAINING
# ==========================================

vector = TfidfVectorizer()

X = vector.fit_transform(df["clean_text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📊 Project Dashboard")

st.sidebar.markdown("""
### Technologies Used

- Python
- Streamlit
- TF-IDF Vectorizer
- Naive Bayes
- NLP

### Purpose

Detect whether a message is:

✅ Ham (Safe)

🚨 Spam (Unwanted)
""")

st.sidebar.metric(
    "Total Messages",
    len(df)
)

st.sidebar.metric(
    "Spam Messages",
    len(df[df["label"] == "spam"])
)

st.sidebar.metric(
    "Ham Messages",
    len(df[df["label"] == "ham"])
)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="main-header">
<h1>📧 Spam Detection System</h1>
<p>Machine Learning Based SMS & Email Classifier</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# USER INPUT
# ==========================================

message = st.text_area(
    "Enter Message",
    height=150,
    placeholder="Example: Congratulations! You won ₹50,000. Click here to claim..."
)

predict = st.button(
    "🔍 Detect Spam",
    use_container_width=True
)

# ==========================================
# PREDICTION
# ==========================================

if predict:

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        # User input stored in list
        new = [message]

        cleaned = [clean(text) for text in new]

        X_new = vector.transform(cleaned)

        prediction = model.predict(X_new)[0]

        probability = model.predict_proba(X_new)

        confidence = round(probability.max() * 100, 2)

        st.markdown("## Result")

        if prediction.lower() == "spam":

            st.markdown(
                """
                <div class="result-spam">
                🚨 SPAM MESSAGE DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-ham">
                ✅ SAFE / HAM MESSAGE
                </div>
                """,
                unsafe_allow_html=True
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                prediction.upper()
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence}%"
            )

        st.markdown("### Message Entered")

        st.info(message)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Built using Streamlit, TF-IDF Vectorization and Naive Bayes Classification."
)
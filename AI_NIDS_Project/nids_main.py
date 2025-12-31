    
# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# App Title
# -----------------------------
st.title("AI-Based Network Intrusion Detection System")
st.write("Using NSL-KDD Binary Dataset for Real Traffic Detection")

# -----------------------------
# Load NSL-KDD Dataset
# -----------------------------
@st.cache_data
def load_real_data():
    # Load NSL-KDD CSV file (replace with your path)
    df = pd.read_csv("NSL_Binary.csv")

    # Assign column names (NSL-KDD has 42 columns)
    df.columns = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
    ]

    # Select only numeric columns for simplicity
    numeric_cols = ["duration","src_bytes","dst_bytes","count","srv_count"]
    X = df[numeric_cols]

    # Convert label to binary: normal=0, attack=1
    y = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Fill missing values with 0
    X = X.fillna(0)

    return X, y

# Load data
X, y = load_real_data()

# -----------------------------
# Train Model
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = None

if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.session_state.model = model
    st.success("Model trained successfully using NSL-KDD binary data!")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Test Network Traffic")

duration = st.number_input("Duration")
src_bytes = st.number_input("Source Bytes")
dst_bytes = st.number_input("Destination Bytes")
count = st.number_input("Count")
srv_count = st.number_input("Srv Count")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Check Traffic"):
    if st.session_state.model is None:
        st.warning("Please train the model first.")
    else:
        input_data = np.array([[duration, src_bytes, dst_bytes, count, srv_count]])
        prediction = st.session_state.model.predict(input_data)

        if prediction[0] == 0:
            st.success("✅ Normal Traffic")
        else:
            st.error("🚨 Intrusion Detected!")

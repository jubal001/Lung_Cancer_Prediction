import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset (for column names)
df = pd.read_csv("Lung Cancer Dataset.csv")
columns = df.drop('PULMONARY_DISEASE', axis=1).columns

st.set_page_config(page_title="AI Lung Health Assistant")

st.title("🩺 AI Lung Health Assistant")
st.write("This AI helps predict lung disease risk and gives health advice.")

# --- USER INPUT ---
st.subheader("Enter Your Details")

user_inputs = []

for col in columns:
    val = st.selectbox(f"{col}", [0, 1])
    user_inputs.append(val)

# --- PREDICTION ---
def predict(data):
    prediction = model.predict([data])[0]
    return prediction

# --- EXPLANATION ---
def explain(data):
    reasons = []
    for i, val in enumerate(data):
        if val == 1:
            reasons.append(columns[i])
    return reasons

# --- CHATBOT LOGIC ---
def chatbot(msg):
    msg = msg.lower()
    
    if "smoke" in msg:
        return "Smoking is a major risk factor for lung disease. Consider reducing or quitting."
    
    elif "chest" in msg:
        return "Chest discomfort can be serious. Please run a risk check above."
    
    elif "help" in msg:
        return "Fill the form and click 'Check Risk' to get your result."
    
    else:
        return "I'm here to help with lung health. Ask me anything!"

# --- BUTTON ---
if st.button("Check Risk"):
    result = predict(user_inputs)
    reasons = explain(user_inputs)
    
    if result == 1:
        st.error("⚠️ High Risk of Lung Disease")
    else:
        st.success("✅ Low Risk")
    
    st.write("🧠 Explanation")
    st.write("Risk factors detected:", reasons)
    
    st.write("💡 Advice")
    if result == 1:
        st.write("- Reduce smoking")
        st.write("- Improve air quality")
        st.write("- Consult a doctor")
    else:
        st.write("- Maintain healthy lifestyle")
        st.write("- Regular checkups")

# --- CHAT SECTION ---
st.subheader("💬 Chat with Assistant")

user_msg = st.text_input("Ask a question")

if user_msg:
    response = chatbot(user_msg)
    st.write("🤖:", response)
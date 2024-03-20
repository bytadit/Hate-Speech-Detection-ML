import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

def main():
    st.title("Hate Speech Detection")
    text_input = st.text_area("Enter your text here:", "")
    
    if st.button("Predict"):
        response = requests.post(API_URL, json={"text": text_input})
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probability = round(result["probability"] * 100, 2)  # Convert probability to percentage
            if prediction == "Hate Speech":
                prediction_text = "😡 " + prediction
                color = "red"
            else:
                prediction_text = "😊 " + prediction
                color = "green"
            st.markdown(f'<p style="color:{color}; font-size:32px;">{prediction_text}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:24px; animation: percentage-grow 1s ease-in-out infinite alternate;">Probability of Hate Speech: {probability}%</p>', unsafe_allow_html=True)
            st.markdown(
                """
                <style>
                @keyframes percentage-grow {
                    0% { font-size: 24px; }
                    100% { font-size: 36px; }
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Failed to get prediction from the server.")
        
if __name__ == "__main__":
    main()

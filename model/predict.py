import joblib
from model.preprocessing import preProcessText

def predictNewData(text):
    model = joblib.load('model\[LR]Hate-Speech-Classifier.joblib')
    vectorizer = joblib.load('model\Hate-Speech-TFIDF.joblib')
    preprocessed_text = preProcessText(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    input_prediction = model.predict(vectorized_text)
    probability_estimates = model.predict_proba(vectorized_text)
    probability_hate_speech = probability_estimates[0][1]
    if input_prediction == 1:
        prediction = "Hate Speech"
    else:
        prediction = "Not Hate Speech"
        
    return {
        "prediction": prediction, 
        "probability": probability_hate_speech
        }

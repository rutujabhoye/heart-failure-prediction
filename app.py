from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("heart_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        'Age': int(request.form['age']),
        'Sex': request.form['sex'],
        'ChestPainType': request.form['cp'],
        'RestingBP': int(request.form['restingbp']),
        'Cholesterol': int(request.form['cholesterol']),
        'FastingBS': int(request.form['fastingbs']),
        'RestingECG': request.form['restecg'],
        'MaxHR': int(request.form['maxhr']),
        'ExerciseAngina': request.form['exang'],
        'Oldpeak': float(request.form['oldpeak']),
        'ST_Slope': request.form['slope']
    }

    df = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df_encoded)[0]
    result = "High Risk" if prediction == 1 else "Low Risk"

    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)

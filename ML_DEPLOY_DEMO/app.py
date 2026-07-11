from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)
model = joblib.load('tree_clf.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

#This is called the Dunder (Double Under) function in Python.
#It ensures that the code inside the block runs only when the file is executed directly, 
# and not when it is imported as a module into another Python file.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
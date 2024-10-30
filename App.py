from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

# Load the saved model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    MSZoning = request.form['MSZoning']
    LotArea = float(request.form['LotArea'])
    OverallQual = int(request.form['OverallQual'])
    YearBuilt = int(request.form['YearBuilt'])
    GrLivArea = float(request.form['GrLivArea'])
    BedroomAbvGr = int(request.form['BedroomAbvGr'])
    TotRmsAbvGrd = int(request.form['TotRmsAbvGrd'])
    GarageCars = int(request.form['GarageCars'])
    GarageArea = float(request.form['GarageArea'])

    # Chuẩn bị dữ liệu đầu vào dưới dạng DataFrame
    input_data = pd.DataFrame({
        'MSZoning': [MSZoning],
        'LotArea': [LotArea],
        'OverallQual': [OverallQual],
        'YearBuilt': [YearBuilt],
        'GrLivArea': [GrLivArea],
        'BedroomAbvGr': [BedroomAbvGr],
        'TotRmsAbvGrd': [TotRmsAbvGrd],
        'GarageCars': [GarageCars],
        'GarageArea': [GarageArea]
    })

    # Dự đoán giá nhà
    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)  # Đảo ngược log transformation

    # Hiển thị kết quả dự đoán
    return render_template('index.html', prediction_text=f'Giá nhà dự đoán: ${prediction:,.2f}')

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))  # Mặc định là 8000 nếu không có biến môi trường PORT
    app.run(host="0.0.0.0", port=port)

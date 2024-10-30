import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Bước 1: Đọc dữ liệu
data = pd.read_csv('train.csv')  # Thay bằng đường dẫn file dữ liệu của bạn

# Bước 2: Tách biến mục tiêu và các đặc trưng
data = data.drop(columns=['Id'])
X = data[['MSZoning', 'LotArea', 'OverallQual', 'YearBuilt', 'GrLivArea', 
          'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']]
y = data['SalePrice']

# Áp dụng log transformation cho biến mục tiêu để giảm độ lệch
y_log = np.log1p(y)

# Xác định các cột dữ liệu số và dữ liệu phân loại
num_cols = X.select_dtypes(exclude='object').columns
cat_cols = X.select_dtypes(include='object').columns

# Tạo pipeline xử lý cho dữ liệu số
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Xử lý giá trị thiếu bằng trung bình
    ('scaler', StandardScaler())                  # Chuẩn hóa dữ liệu số
])

# Tạo pipeline xử lý cho dữ liệu phân loại
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Xử lý giá trị thiếu bằng giá trị phổ biến nhất
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Mã hóa one-hot
])

# Kết hợp các pipeline cho dữ liệu số và phân loại
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Bước 3: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Bước 4: Tạo và huấn luyện mô hình
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Huấn luyện mô hình
model_pipeline.fit(X_train, y_train)

# Bước 5: Đánh giá mô hình trên tập kiểm tra
# Dự đoán trên tập kiểm tra
y_pred_log = model_pipeline.predict(X_test)

# Chuyển đổi ngược từ log để tính MSE và R2 cho giá trị thực
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

# Tính toán MSE và R2
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Bước 6: Lưu mô hình đã huấn luyện vào file .pkl
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model_pipeline, file)

print("Model training completed, evaluated, and saved to 'house_price_model.pkl'.")

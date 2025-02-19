import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Đọc dữ liệu Titanic
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Tải dữ liệu và xử lý
def load_data():
    df = pd.read_csv(DATA_URL)
    
    # Loại bỏ các cột không cần thiết
    columns_to_drop = ['Name', 'Ticket', 'Cabin', 'Embarked']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Xử lý giá trị thiếu
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Điền giá trị trung bình cho các cột số
    df.fillna("Unknown", inplace=True)  # Điền "Unknown" cho các cột chuỗi
    
    # Chuyển đổi kiểu dữ liệu
    df['Pclass'] = df['Pclass'].astype(str)  # Chuyển đổi Pclass thành chuỗi
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Chuyển đổi giới tính thành số
    
    return df

# Tiền xử lý dữ liệu (chuẩn hóa và chuẩn bị cho mô hình)
def preprocess_data(df):
    # Lựa chọn các cột số để chuẩn hóa
    numeric_cols = ['Age', 'Fare']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Chuyển đổi các cột còn lại thành các giá trị số
    df = pd.get_dummies(df, drop_first=True)  # Sử dụng one-hot encoding cho các cột chuỗi
    
    return df

# Chia dữ liệu thành các tập train/valid/test theo tỷ lệ người dùng tùy chọn
def split_data(df, train_size=0.7, valid_size=0.15, test_size=0.15):
    train, temp = train_test_split(df, test_size=1 - train_size, random_state=42)
    valid, test = train_test_split(temp, test_size=test_size / (test_size + valid_size), random_state=42)
    return train, valid, test

# Huấn luyện mô hình Linear hoặc Polynomial Regression và log kết quả với MLFlow
def train_model(train_data, valid_data, degree=1):
    X_train = train_data.drop(columns='Survived')
    y_train = train_data['Survived']
    
    X_valid = valid_data.drop(columns='Survived')
    y_valid = valid_data['Survived']
    
    # Chọn mô hình Linear Regression hoặc Polynomial Regression
    if degree == 1:  # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
    else:  # Polynomial Regression
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_valid_poly = poly.transform(X_valid)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
    # Cross-validation trên tập training và validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_cv_score = np.mean(cv_scores)
    
    # Đánh giá mô hình trên tập validation
    if degree == 1:
        y_pred = model.predict(X_valid)
    else:
        y_pred = model.predict(X_valid_poly)
    
    mse = mean_squared_error(y_valid, y_pred)
    
    # Logging kết quả với MLFlow
    with mlflow.start_run():
        mlflow.log_param("degree", degree)
        mlflow.log_param("cv_mean_mse", mean_cv_score)
        mlflow.log_metric("mse", mse)
        
        # Log mô hình đã huấn luyện
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Cross-validation MSE: {mean_cv_score}")
        print(f"Validation MSE: {mse}")
        
    return model

if __name__ == "__main__":
    # Tải và tiền xử lý dữ liệu
    df = load_data()
    df = preprocess_data(df)
    
    # Chia dữ liệu theo tỷ lệ người dùng tùy chọn
    train, valid, test = split_data(df, train_size=0.7, valid_size=0.15, test_size=0.15)
    
    # Huấn luyện mô hình với Linear Regression (hoặc Polynomial Regression)
    train_model(train, valid, degree=1)  # Chọn degree=1 cho Linear Regression

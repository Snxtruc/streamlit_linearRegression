import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

# Tải dữ liệu Titanic
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

# Tiền xử lý dữ liệu
def preprocess_data(df):
    st.markdown("## 🔍 Tiền xử lý dữ liệu")
    st.markdown("### 📌 10 dòng đầu của dữ liệu gốc")
    st.dataframe(df.head(10))
    
    # Loại bỏ cột không cần thiết
    df.drop(columns=["Cabin", "Ticket", "Name"], inplace=True)

    st.markdown("### ⚙️ Các bước chính trong tiền xử lý dữ liệu")
    st.markdown("### 1️⃣ Loại bỏ các cột không cần thiết")
    st.markdown(
        """
        Một số cột trong dữ liệu không có giá trị quan trọng đối với mô hình dự đoán hoặc chứa quá nhiều giá trị bị thiếu. 
        Vì vậy, chúng ta sẽ loại bỏ các cột sau:
    
        - **Cabin**: Chứa nhiều giá trị bị thiếu.
        - **Ticket**: Mã vé không cung cấp nhiều thông tin hữu ích.
        - **Name**: Không ảnh hưởng đến kết quả dự đoán.
        """
    )

    st.markdown("#### 📜 Code xử lý")
    st.code(
        """
        columns_to_drop = ["Cabin", "Ticket", "Name"]
        df.drop(columns=columns_to_drop, inplace=True)
        """,
        language="python"
    )
    
    # Xử lý giá trị thiếu
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df.dropna(subset=["Embarked"], inplace=True)

    st.markdown("""
        ### 2️⃣ Xử lý giá trị thiếu
        Dữ liệu thực tế thường có giá trị bị thiếu. Ta cần xử lý để tránh ảnh hưởng đến mô hình.

        - **Cột "Age"**: Điền giá trị trung bình vì đây là dữ liệu số.
        - **Cột "Fare"**: Điền giá trị trung vị để giảm ảnh hưởng của ngoại lai.
        - **Cột "Embarked"**: Xóa các dòng bị thiếu vì số lượng ít.
        """) 

    st.markdown("#### 📜 Code xử lý")
    # Hiển thị code xử lý
    txt_code = '''
    df["Age"].fillna(df["Age"].mean(), inplace=True)  # Điền giá trị trung bình cho "Age"
    df["Fare"].fillna(df["Fare"].median(), inplace=True)  # Điền giá trị trung vị cho "Fare"
    df.dropna(subset=["Embarked"], inplace=True)  # Xóa dòng thiếu "Embarked"
    '''
    st.code(txt_code, language="python")

    
    # Mã hóa dữ liệu
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    st.markdown("""
    - **Cột `"Sex"`**: Chuyển đổi giá trị `"male"` thành `1`, `"female"` thành `0`.
    - **Cột `"Embarked"`**: Sử dụng **One-Hot Encoding** để tạo các cột mới cho từng giá trị ("S", "C", "Q").
    """)

    st.markdown("#### 📜 Code xử lý")
    code = '''
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})  # Mã hóa giới tính (Nam: 1, Nữ: 0)
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)  # One-Hot Encoding cột "Embarked"
    '''
    st.code(code, language="python")
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]]) 

    st.markdown("### 🔄 Chuẩn hóa dữ liệu số")
    st.markdown("""
    - **Vấn đề**: Các giá trị số có thể có khoảng giá trị khác nhau, làm ảnh hưởng đến mô hình.
    - **Giải pháp**: Dùng `StandardScaler()` để đưa dữ liệu **"Age"** và **"Fare"** về cùng một thang đo.
    """)

    code = '''
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
    '''
    st.markdown("#### 📜 Code xử lý")
    st.code(code, language="python") 
    
    st.subheader("✅ Dữ liệu sau tiền xử lý")
    st.dataframe(df.head()) 
    
    # Vẽ biểu đồ phân phối dữ liệu trước và sau chuẩn hóa
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['Age'], bins=30, kde=True, ax=ax[0])
    ax[0].set_title("Phân phối Age sau chuẩn hóa")
    sns.histplot(df['Fare'], bins=30, kde=True, ax=ax[1])
    ax[1].set_title("Phân phối Fare sau chuẩn hóa")
    st.pyplot(fig)
    
    st.success("Dữ liệu đã được tiền xử lý xong!")
    
    return df

# Chia dữ liệu
def split_data(df, train_size=0.7, valid_size=0.15, test_size=0.15):
    train, temp = train_test_split(df, test_size=(1 - train_size), random_state=42)
    valid, test = train_test_split(temp, test_size=(test_size / (valid_size + test_size)), random_state=42)
    
    # Vẽ biểu đồ tỷ lệ dữ liệu
    labels = ['Train', 'Validation', 'Test']
    sizes = [len(train), len(valid), len(test)]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
    ax.set_title("Tỷ lệ dữ liệu")
    st.pyplot(fig) 

    st.markdown(" ### 5️⃣ Chia dữ liệu thành tập Train, Validation, và Test")

    # 📝 Mô tả
    st.markdown("""
    Dữ liệu được chia thành ba phần để đảm bảo mô hình tổng quát tốt:
    - **70%**: để train mô hình.
    - **15%**: để validation, dùng để điều chỉnh tham số.
    - **15%**: để test, đánh giá hiệu suất thực tế.
    """)

    # Hiển thị code mẫu
    st.code("""
    # Chia dữ liệu theo tỷ lệ 70% và 30% (train - temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Chia tiếp 30% thành 15% validation và 15% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    """, language="python") 

    # 📊 Thống kê số lượng mẫu
    st.markdown("### 📊 Số lượng mẫu trong từng tập dữ liệu:")
    st.markdown(f"👉 **Train**: {622} mẫu")
    st.markdown(f"👉 **Validation**: {133} mẫu")
    st.markdown(f"👉 **Test**: {134} mẫu")
    
    return train, valid, test

# Huấn luyện mô hình
def train_model(train, valid, degree):
    st.markdown("## 🚀 Huấn luyện mô hình") 
    
    st.markdown("### 🔥 Mô hình Linear/Polynomial Regression")

    # 📌 Giới thiệu
    st.header("1️⃣ Giới thiệu về Polynomial Regression")
    st.write(
        "Polynomial Regression (Hồi quy đa thức) là một phương pháp mở rộng của "
        "Linear Regression, giúp mô hình có thể nắm bắt được mối quan hệ phi tuyến tính "
        "giữa biến độc lập (X) và biến phụ thuộc (y)."
    )

    st.subheader("📌 Công thức tổng quát của Hồi quy bậc k:")
    st.latex(r"""
    y = \theta_0 + \theta_1 X + \theta_2 X^2 + \dots + \theta_k X^k + \epsilon
    """)

    st.write("Trong đó:")
    st.markdown("- \( k \): Bậc của đa thức (degree)")
    st.markdown("- \( \theta \): Hệ số hồi quy cần học")
    st.markdown("- \( \epsilon \): Nhiễu trong dữ liệu") 

    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Polyreg_scheffe.svg/488px-Polyreg_scheffe.svg.png", caption = "Mô hình Polynomail Regression", use_container_width=True)

    st.success("🚀 Polynomial Regression giúp mô hình bắt được xu hướng phi tuyến của dữ liệu!") 

    # Tiêu đề
    st.markdown("### 2️⃣ Khi nào sử dụng Polynomial Regression?")

    # Nội dung
    st.markdown("""
    ### 📌 Sử dụng Polynomial Regression khi:
    - ✅ **Dữ liệu có quan hệ phi tuyến** giữa X và y.
    - ✅ **Linear Regression có hiệu suất kém** do không thể nắm bắt được mối quan hệ cong của dữ liệu.
    - ✅ **Cần thử nghiệm một mô hình đơn giản** trước khi sử dụng các mô hình phức tạp hơn như **Random Forest hoặc Neural Networks**.
    """)

    st.markdown("### 🚀 Các bước huấn luyện mô hình Polynomial Regression") 
    st.markdown("""
    ### 🔹 Bước 1: Chuẩn bị dữ liệu 
    - ✅ Tải và làm sạch dữ liệu.
    - ✅ Xử lý giá trị thiếu, chuẩn hóa dữ liệu nếu cần.
    - ✅ Chia dữ liệu thành train, validation, test theo tỷ lệ thích hợp (VD: 70%-15%-15%).
    """)

    st.markdown("""
    ### 🔹 Bước 2: Biến đổi dữ liệu thành đa thức 
    - ✅  Dùng **PolynomialFeatures**  để tạo ra các đặc trưng mới với bậc k tùy chọn.
    - ✅  Ví dụ, nếu chọn **bậc 2**, một đặc trưng x sẽ được chuyển thành **[x, x^2]** 
    """)

    st.markdown("### 📜 Mã nguồn Python:")
    st.code("""
    from sklearn.preprocessing import PolynomialFeatures

    degree = 2  # Chọn bậc của đa thức
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_valid_poly = poly.transform(X_valid)
    X_test_poly = poly.transform(X_test)
    """, language="python") 

    st.markdown("""
    ### 🔹 Bước 3: Huấn luyện mô hình  
    - ✅  Dùng **LinearRegression** để huấn luyện trên tập dữ liệu đã biến đổi.
    - ✅  Tính toán **MSE (Mean Squared Error)** để đánh giá mô hình.
    """) 

    st.code("""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_valid_poly)
    mse = mean_squared_error(y_valid, y_pred)
    """, language="python") 

    st.markdown("### 🎯 Đánh giá mô hình bằng Cross-Validation") 
    # Giải thích về Cross-Validation
    st.markdown("""
    ### 📌 Cross-Validation là gì?
    Cross-Validation (CV) là một kỹ thuật đánh giá mô hình giúp kiểm tra hiệu suất của mô hình trên nhiều tập dữ liệu khác nhau. Mục tiêu chính của CV là **giảm thiểu overfitting** và đánh giá mô hình một cách tổng quát hơn.

    ### 🚀 Cách hoạt động của Cross-Validation:
    1️⃣ Chia dữ liệu thành **n phần (folds)**.  
    2️⃣ Sử dụng **(n-1) folds** để train, và **1 fold** để validate.  
    3️⃣ Thực hiện huấn luyện **n lần**, mỗi lần chọn một fold khác nhau để validate.  
    4️⃣ Cuối cùng, lấy **trung bình các kết quả** để đánh giá hiệu suất mô hình.
    """)


    X_train, y_train = train.drop(columns='Survived'), train['Survived']
    X_valid, y_valid = valid.drop(columns='Survived'), valid['Survived']
    
    if degree == 1:
        model = LinearRegression()
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_valid, model.predict(X_valid))
        poly = None
    else:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_valid_poly = poly.transform(X_valid)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        mse = mean_squared_error(y_valid, model.predict(X_valid_poly))
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    with mlflow.start_run():
        mlflow.log_param("degree", degree)
        mlflow.log_metric("cv_mean_mse", np.mean(cv_scores))
        mlflow.log_metric("valid_mse", mse)
        mlflow.sklearn.log_model(model, "model")
    
    return model, mse, poly
# Dự đoán sống sót
def predict_survival(model, poly, user_input):
    input_df = pd.DataFrame([user_input])
    
    # Chuyển đổi dữ liệu
    input_df["Sex"] = input_df["Sex"].map({"male": 1, "female": 0})
    input_df = pd.get_dummies(input_df, columns=["Embarked"], drop_first=True)

    # Đảm bảo đầy đủ cột đặc trưng bao gồm PassengerId
    expected_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    for col in expected_cols:
        if col not in input_df:
            input_df[col] = 0  

    input_df = input_df[expected_cols]
    
    # Chuẩn hóa dữ liệu (trừ PassengerId)
    scaler = StandardScaler()
    input_df[["Age", "Fare"]] = scaler.fit_transform(input_df[["Age", "Fare"]])
    
    # Áp dụng PolynomialFeatures nếu có
    if poly:
        input_df = poly.transform(input_df)
    
    # Dự đoán
    prediction = model.predict(input_df)[0]
    
    return prediction


# Giao diện Streamlit
def main():
    st.title("🔍 Phân tích Titanic với Regression")
    
    df = load_data()
    df = preprocess_data(df)
    train, valid, test = split_data(df)
    
    degree = st.slider("Chọn bậc của Polynomial Regression", 1, 5, 1)
    model, mse, poly = train_model(train, valid, degree)
    
    st.subheader("📊 Kết quả") 

    st.subheader("🛳️ Dự đoán khả năng sống sót")
    user_input = {
        "Pclass": st.selectbox("Hạng vé", [1, 2, 3]),
        "Sex": st.radio("Giới tính", ["male", "female"]),
        "Age": st.slider("Tuổi", 1, 80, 30),
        "SibSp": st.slider("Anh chị em / Vợ chồng đi cùng", 0, 8, 0),
        "Parch": st.slider("Cha mẹ / Con cái đi cùng", 0, 6, 0),
        "Fare": st.slider("Giá vé", 0.0, 500.0, 50.0),
        "Embarked": st.selectbox("Cảng đi", ["C", "Q", "S"])
    }
    
    if st.button("Dự đoán sống sót"):
        prediction = predict_survival(model, poly, user_input)
        st.write(f"🔮 Xác suất sống sót: {prediction:.2f}")
        
        fig, ax = plt.subplots()
        ax.bar(["Không sống sót", "Sống sót"], [1 - prediction, prediction], color=["red", "green"])
        st.pyplot(fig)

if __name__ == "__main__":
    main()

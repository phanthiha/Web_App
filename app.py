import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os
import io

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Retail Data App", layout="wide")
st.title("Retail Dashboard & Revenue Forecast")

# --- KHỞI TẠO CACHE CHO PYSPARK VÀ MÔ HÌNH ---
@st.cache_resource
def load_environment():
    spark = SparkSession.builder.appName("Streamlit_Retail").getOrCreate()
    # Tải mô hình tốt nhất đã lưu
    model = PipelineModel.load("best_pipeline_model")
    return spark, model

spark, model = load_environment()

# --- TẠO CÁC TAB GIAO DIỆN ---
tab1, tab2 = st.tabs(["Dashboard & EDA", "Prediction Forecast"])

# ==========================================
# TAB 1: DASHBOARD METRICS & INTERACTIVE EDA
# ==========================================
with tab1:
    st.header("1. Dashboard Metrics")
    if os.path.exists("eda_artifacts/descriptive_stats.parquet"):
        stats_df = pd.read_parquet("eda_artifacts/descriptive_stats.parquet")
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.warning("Không tìm thấy file descriptive_stats.parquet")

    st.header("2. Interactive EDA Charts")
    if os.path.exists("eda_artifacts/revenue_by_product.parquet"):
        rev_df = pd.read_parquet("eda_artifacts/revenue_by_product.parquet")
        fig = px.bar(
            rev_df, 
            x="Product_line", 
            y="Total_Revenue", 
            color="Product_line",
            title="Tổng doanh thu theo từng Dòng sản phẩm",
            labels={"Total_Revenue": "Tổng Doanh Thu ($)", "Product_line": "Dòng Sản Phẩm"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Không tìm thấy file revenue_by_product.parquet")

# ==========================================
# TAB 2: PREDICTION REVENUE FORECAST
# ==========================================
with tab2:
    st.header("Dự đoán Doanh thu (Sales Forecast)")
    
    # Chia làm 2 phần nhỏ trong Tab 2
    sub_tab1, sub_tab2 = st.tabs(["Dự đoán Đơn lẻ (Single)", "Dự đoán Hàng loạt (Upload Excel)"])
    
    # ------------------------------------
    # PHẦN 1: DỰ ĐOÁN ĐƠN LẺ (Như cũ)
    # ------------------------------------
    with sub_tab1:
        st.write("Nhập thông tin giả định để dự đoán một đơn hàng:")
        col1, col2 = st.columns(2)
        with col1:
            branch = st.selectbox("Chi nhánh (Branch)", ["A", "B", "C"])
            customer_type = st.selectbox("Loại khách hàng (Customer type)", ["Member", "Normal"])
            gender = st.selectbox("Giới tính (Gender)", ["Female", "Male"])
            product_line = st.selectbox("Dòng sản phẩm", [
                "Health and beauty", "Electronic accessories", "Home and lifestyle", 
                "Sports and travel", "Food and beverages", "Fashion accessories"
            ])
        with col2:
            unit_price = st.number_input("Đơn giá (Unit price)", min_value=1.0, value=50.0, step=1.0)
            quantity = st.number_input("Số lượng (Quantity)", min_value=1, max_value=100, value=5, step=1)
            
        if st.button("Dự đoán Đơn hàng", type="primary", key="btn_single"):
            input_data = [(branch, customer_type, gender, product_line, float(unit_price), float(quantity))]
            columns = ["Branch", "Customer_type", "Gender", "Product_line", "Unit_price", "Quantity"]
            input_df = spark.createDataFrame(input_data, columns)
            
            with st.spinner('Đang tính toán...'):
                prediction_df = model.transform(input_df)
                predicted_sales = prediction_df.select("prediction").collect()[0][0]
            st.success(f"Doanh thu dự đoán (Sales): **${predicted_sales:,.2f}**")

    # ------------------------------------
    # PHẦN 2: DỰ ĐOÁN TỪ FILE EXCEL
    # ------------------------------------
    with sub_tab2:
        st.write("Tải lên file Excel (`.xlsx`) chứa danh sách các đơn hàng. File cần có các cột:")
        st.info("`Branch`, `Customer_type`, `Gender`, `Product_line`, `Unit_price`, `Quantity`")
        
        uploaded_file = st.file_uploader("Kéo thả hoặc chọn file Excel của bạn", type=["xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Đọc dữ liệu từ file Excel
                batch_pdf = pd.read_excel(uploaded_file)
                st.write("**Dữ liệu đầu vào (Preview):**")
                st.dataframe(batch_pdf.head(10), use_container_width=True) # Hiển thị 10 dòng đầu
                
                if st.button("Chạy Dự đoán Hàng loạt", type="primary", key="btn_batch"):
                    with st.spinner('PySpark đang xử lý dữ liệu hàng loạt...'):
                        # Đảm bảo kiểu dữ liệu an toàn trước khi chuyển sang PySpark
                        batch_pdf['Unit_price'] = batch_pdf['Unit_price'].astype(float)
                        batch_pdf['Quantity'] = batch_pdf['Quantity'].astype(float)
                        
                        # Chuyển Pandas DF -> PySpark DF
                        spark_batch_df = spark.createDataFrame(batch_pdf)
                        
                        # Dự đoán bằng mô hình
                        batch_predictions = model.transform(spark_batch_df)
                        
                        # Rút trích các cột quan trọng và cột dự đoán (prediction) ra Pandas
                        result_pdf = batch_predictions.select(
                            "Branch", "Product_line", "Unit_price", "Quantity", "prediction"
                        ).toPandas()
                        
                        # Làm tròn số và đổi tên cột
                        result_pdf['prediction'] = result_pdf['prediction'].round(2)
                        result_pdf.rename(columns={"prediction": "Predicted_Sales_($)"}, inplace=True)
                        
                        st.success("Dự đoán thành công!")
                        st.dataframe(result_pdf, use_container_width=True)
                        
                        # Tính năng: Tải xuống kết quả dạng Excel
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            result_pdf.to_excel(writer, index=False, sheet_name='Predictions')
                        
                        st.download_button(
                            label="Tải xuống kết quả (.xlsx)",
                            data=buffer.getvalue(),
                            file_name="predicted_sales_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi đọc hoặc xử lý file: {e}")
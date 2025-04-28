import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Data Sweeper", layout="wide")

# Function to load data
def load_data(file):
    if file is not None:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            st.error("Invalid file format. Please upload a CSV or Excel file.")
            return None
    return None

# Function for cleaning data
def clean_data(df):
    # Drop missing values
    st.write("### Clean Missing Data")
    st.write(f"Total missing values: \n{df.isnull().sum()}")
    drop_missing = st.checkbox("Remove rows with missing values", value=False)
    if drop_missing:
        df = df.dropna()
        st.success("Missing values removed.")
    
    # Drop duplicates
    st.write("### Remove Duplicates")
    duplicate_rows = df.duplicated().sum()
    st.write(f"Total duplicate rows: {duplicate_rows}")
    drop_duplicates = st.checkbox("Remove duplicate rows", value=False)
    if drop_duplicates:
        df = df.drop_duplicates()
        st.success("Duplicate rows removed.")
    
    return df

# Function for visualizing data
def visualize_data(df):
    st.write("### Data Visualizations")
    chart_type = st.selectbox("Select Chart Type", ["Histogram", "Correlation Heatmap", "Box Plot"])
    
    if chart_type == "Histogram":
        column = st.selectbox("Select Column for Histogram", df.columns)
        st.write(f"### Histogram of {column}")
        fig = px.histogram(df, x=column, nbins=20, title=f"Histogram of {column}")
        st.plotly_chart(fig)
    
    elif chart_type == "Correlation Heatmap":
        st.write("### Correlation Heatmap")
        correlation_matrix = df.corr()
        fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='Blues', title="Correlation Heatmap")
        st.plotly_chart(fig)
    
    elif chart_type == "Box Plot":
        column = st.selectbox("Select Column for Box Plot", df.columns)
        st.write(f"### Box Plot of {column}")
        fig = px.box(df, y=column, title=f"Box Plot of {column}")
        st.plotly_chart(fig)

# Function for encoding categorical columns
def encode_categorical(df):
    st.write("### Encode Categorical Data")
    categorical_columns = df.select_dtypes(include=["object"]).columns
    st.write(f"Categorical Columns: {list(categorical_columns)}")
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        st.write(f"Encoded {col} column.")
    return df

# Streamlit UI
st.title("🔍 Data Sweeper - Data Cleaning & Visualization Tool")
st.write("""
    This application allows you to upload a dataset (CSV/Excel), clean it by removing missing values and duplicates,
    and visualize key insights like distributions and correlations.
""")

# Upload Data
uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load the data
    df = load_data(uploaded_file)
    if df is not None:
        # Display Data Preview
        st.write("### Data Preview")
        st.write(df.head())

        # Clean Data Section
        st.sidebar.title("Data Cleaning")
        df = clean_data(df)

        # Data Encoding
        df = encode_categorical(df)

        # Visualization Section
        visualize_data(df)

        # Show cleaned data preview
        st.write("### Cleaned Data Preview")
        st.write(df.head())

        # Allow user to download cleaned data
        st.write("### Download Cleaned Data")
        cleaned_data = df.to_csv(index=False)
        st.download_button("Download Cleaned Data", cleaned_data, file_name="cleaned_data.csv")


import streamlit as st
import pandas as pd
import  os
#noman
# profiling libraries

from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
# noman
# ML libraries
from pycaret.classification import setup as setup_class, compare_models as compare_models_class, create_model as create_model_class, save_model as save_model_class,pull as  pull_class
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, create_model as create_model_reg, save_model as save_model_reg, pull as pull_reg

st.write("Hello World This is a test streamlit app_123")

with st.sidebar:
    #st.image("https://static.streamlit.io/examples/cat.jpg")
    st.write("This is a sidebar")
    st.title("This is a sidebar title")
    choice = st.radio("What do you like to do: ", ["Upload", "Profile", "ML", "Download"])
    st.write(choice)

# Ensure df is defined
if "df" not in globals():
    df = pd.DataFrame()

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv")
    st.dataframe(df)

if choice == "Upload":
    st.write("You chose upload")
    file = st.file_uploader("Choose a file")
    #st.write(file)
    if file:
        #st.write(file.getvalue())
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv")
        st.dataframe(df)


if choice == "Profile":
    st.title("Automated EDA")
    if 'df' in locals():
        profile = ProfileReport(df, title="Profiling Report")
        st_profile_report(profile)
    else:
        st.write("No data available for profiling.")

# noman - just test
if choice == "ML":
    st.write("You chose ML")
    if 'df' in locals():
        # Check for missing values in the target column
        target = st.selectbox("What is the target column", df.columns)
        missing_values_count = df[target].isnull().sum()
        if missing_values_count > 0:
            st.error(f"There were {missing_values_count} missing values in {target}. Rows with missing values are discarded.")
            df = df.dropna(subset=[target])
        
        ml_task = st.selectbox("Choose ML task", ["Regression", "Classification"])
        
        if st.button("Train Model"):
            if ml_task == "Classification":
                setup_class(data=df, target=target)
                best_model = compare_models_class()
                compare_df = pull_class()
                st.info("Comparison is complete")
                st.dataframe(compare_df)
            elif ml_task == "Regression":
                setup_reg(data=df, target=target)
                best_model = compare_models_reg()
                compare_df = pull_reg()
                st.info("Comparison is complete")
                st.dataframe(compare_df)


            st.write(f"Best Model: {best_model}")
            if ml_task == "Classification":
                save_model_class(best_model, "best_model_classification")
             
            elif ml_task == "Regression":
                save_model_reg(best_model, "best_model_regression")
                


    else:
        st.write("No data available for ML.")


if choice == "Download":
    st.write("You chose download")
    model_file = st.selectbox("Choose model to download", ["best_model_classification.pkl", "best_model_regression.pkl"])
    if st.button("Download Model"):
        with open(model_file, "rb") as f:
            st.download_button("Download the model", f, model_file)



    


import streamlit as st
import pandas as pd
import  os
#noman
# profiling libraries
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# ML libraries
from pycaret.classification import setup, compare_models, pull, create_model, tune_model, finalize_model, predict_model, save_model, load_model


st.write("Hello World")

with st.sidebar:
    #st.image("https://static.streamlit.io/examples/cat.jpg")
    st.write("This is a sidebar")
    st.title("This is a sidebar title")
    choice = st.radio("What is your favorite animal", ["Upload", "Profile", "ML", "Download"])
    st.write(choice)

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
    profile_report = ydata_profiling.ProfileReport(df)
    st_profile_report(profile_report)



if choice == "ML":
    st.write("You chose ML")
    target = st.selectbox("What is the target column", df.columns)
    if st.button("train model"):
        setup(df, target = target)
        setup_df = pull()
        st.info("Setup is complete")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("Comparison is complete")
        st.dataframe(compare_df)
        best_model = create_model(best_model)
        save_model(best_model, "best_model")



if choice == "Download":
    st.write("You chose download")
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the model", f, "trained_model.pkl")

    

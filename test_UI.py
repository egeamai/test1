import streamlit as st
from functions_ui import *

# STREAMLIT PAGE
tabs = ["Data Analyst", "Researcher", "ML Engineer", "AM Engineer", "Final Query", "Nano", "EGE AMAI LIFT-UP TEAM Hakkında"]
page = st.sidebar.radio("Sayfalar", tabs)


# Data Analys Ekranı
if page == "Data Analyst":
    st.markdown("<h1 style='text-align: center; color: black;'>EGE AMAI LIFT-UP Data Analyst Part</h1>", unsafe_allow_html=True)
    st.image("./banner.png", width=700)
    result_df = DataAnalystRun()

    ground_truth = st.text_input("What is your ground_truth?: ")
    if st.button("Evaluate"):
        result_df['ground_truth'] = [ground_truth]
        DA_Eval_Run(result_df)
        st.snow()


# Researcher Ekranıs
if page == "Researcher":
    st.markdown("<h1 style='text-align: center; color: black;'>EGE AMAI LIFT-UP Researcher Part</h1>", unsafe_allow_html=True)
    st.image("./banner.png", width=200)
    #st.success("EGE AMAI LIFT-UP Researcher Part")


# ML Engineer Ekranı
if page == "ML Engineer":
    st.markdown("<h1 style='text-align: center; color: black;'>EGE AMAI LIFT-UP ML Engineer Part</h1>", unsafe_allow_html=True)
    st.image("./banner.png", width=200)
    #st.success("EGE AMAI LIFT-UP ML Engineer Part")


# AM Engineer Ekranı
if page == "AM Engineer":
    st.markdown("<h1 style='text-align: center; color: black;'>EGE AMAI LIFT-UP AM Engineer Part</h1>", unsafe_allow_html=True)
    st.image("./banner.png", width=200)
    result_df = AMEngineerRun()

    ground_truth = st.text_input("What is your ground_truth?: ")
    if st.button("Evaluate"):
        result_df['ground_truth'] = [ground_truth]
        AM_Eval_Run(result_df)
        st.snow()


if page=="Final Query":
    st.markdown("<h2 style='text-align: center; color: black;'>EGE AMAI LIFT-UP VIRTUAL TEAM PART</h2>", unsafe_allow_html=True)
    st.image("./banner.png", width=700)
    result_df = DataAnalystRun()
    essay_no = int(result_df['essay_no'])

    st.write("Researcher Part")
    st.spinner("Loading...")
    ResearcherRun(essay_no)

    st.write("....................")
    st.write("ML Engineer Part")
    MLEngineerRun(essay_no)

if page == "Nano":
    st.markdown("<h1 style='text-align: center; color: black;'>Nano Part</h1>", unsafe_allow_html=True)
    st.image("./banner.png", width=200)
    result_df = NanoRun()
    


if page == "EGE AMAI LIFT-UP TEAM Hakkında":
    st.success("EGE AMAI LIFT-UP TEAM")
    st.write("................................")

"""
Created on Sat Juni 02 15:54:49 2023

@author: Vladimir Burlay
"""
import streamlit as st
import plotly.express as px
import urllib
import pandas as pd


def main():
    dl_to_file = 'https://raw.githubusercontent.com/vburlay/uci-bank-marketing/main/data/df_cnn_res.csv'
    dl_pred = pd.read_csv(dl_to_file)
    return dl_pred

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/vburlay/uci-bank-marketing/main/data/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


st.sidebar.title("Control Panel")

with st.sidebar:
    add_selectbox = st.selectbox("App-Mode", ["Application start", "Show the source code"])
    add_radio = st.radio("Choose a model", ("Keras - CNN"))

if add_selectbox == "Application start":
    dl_pred = main()
    data_dl = pd.DataFrame(data=dl_pred)
    st.title("Caravan insurance")

    tab1, tab2, tab3 = st.tabs(["Countplot of the results", "Result Tabular", "Individual results"])
    with tab1:
        if add_radio == "Keras - CNN":
            st.bar_chart(data=data_dl.loc[:, ['Yes_Prob', 'predicted']], x='predicted', width=1000, height=500)
    with tab2:
        if add_radio == "Keras - CNN":
            fig = px.scatter(data_dl.loc[:, ['Yes_Prob', 'predicted']], width=1000, height=650)
            st.plotly_chart(fig)
    with tab3:
        if add_radio == "Keras - CNN":
            st.dataframe(data_dl.drop(columns=['ID', 'Yes']), width=1200, height=600)


elif add_selectbox == "Show the source code":
    readme_text = st.markdown(get_file_content_as_string("streamlit.md"))

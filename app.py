import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

st.set_page_config(page_title="World Happiness Report 2021", page_icon=":smile:", layout="wide")

st.title("World Happiness Report Analytics")
st.markdown("Welcome! To narrow your search, please choose from the filters in the sidebar on the left of your screen.")
st.markdown("##")
data = pd.read_csv(filepath_or_buffer = "world-happiness-report-2021.csv")

median_ladderScore = np.median(data['Ladder score'])
howMany_Ladders = "\U0001fa9c" * int(median_ladderScore)
mean_lifeExpectancy = np.mean(data['Healthy life expectancy'])
happiest_country = data[data['Ladder score'] == np.max(data['Ladder score'])]['Country name'].item()

left_column, middle_column, right_column = st.columns(3)

with left_column:
    st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">Median Ladder Score:</p>', unsafe_allow_html=True)
    st.subheader(body=f"{median_ladderScore}")
    # {howMany_Ladders}
with middle_column:
    st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">Average Life Expectancy:</p>', unsafe_allow_html=True)
    st.subheader(body =  f"{mean_lifeExpectancy: .2f} Years old")
with right_column:
    st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">Happiest Country:</p>', unsafe_allow_html=True)
    st.subheader(body=f"{happiest_country}")

st.markdown("---")

st.sidebar.header("Filter:")

st.markdown(body = '<p style="font-family:sans-serif; font-size: 24px;">Data by Regions</p>', unsafe_allow_html=True)

region_selected = st.sidebar.multiselect(label = "select region:",
                                         options = data['Regional indicator'].unique(),
                                         default = data['Regional indicator'].unique())

data_regions = data.query("`Regional indicator` == @region_selected")

numer_of_results = data_regions.shape[0]

st.markdown(f"*Number of observations: {numer_of_results}*")
st.dataframe(data=data_regions)


#Merdiven Skoru Kaydırıcısı (Slider)
st.markdown(body = '<p style="font-family:sans-serif; font-size: 24px;">Data by Ladder Score</p>', unsafe_allow_html=True)

score = st.sidebar.slider(label = 'Minimum Ladder score:', min_value=5, max_value=10, value = 10)

data_ladderScore = data[data['Ladder score'] <= score] 

numer_of_results2 = data_ladderScore.shape[0]

st.markdown(f"*Number of observations: {numer_of_results2}*")
st.dataframe(data=data_ladderScore)

st.markdown("----")


fig_gdp_lifeExpect = px.scatter(data_frame = data_regions,
x="Logged GDP per capita",
y="Healthy life expectancy",
size="Ladder score",
color="Regional indicator",
hover_name="Country name",
size_max=10,
title='GDP per capita and Life expectancy',
template='ggplot2')

fig_gdp_lifeExpect.update_layout(
    xaxis=(dict(showgrid=False))
)

fig_ladderByCountry = px.bar(data_frame= data_regions, x='Ladder score', y='Country name', orientation='h', title='Ladder scores by country', template='ggplot2')

fig_ladderByCountry.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_gdp_lifeExpect, use_container = True)
right_column.plotly_chart(fig_ladderByCountry, use_container = True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

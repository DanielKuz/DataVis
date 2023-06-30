import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np


st.set_page_config(page_title="World Happiness Report 2021", page_icon=":smile:", layout="wide")

st.title("The happy world - analyzing the beauty behind the happiness")
st.markdown("Welcome to all the information you will ever need about what is behind happiness, in this dashboard you "
            "will get quick information about the happiest countries in the world and you can also dive deeper into "
            "the connections of the factors that affect world happiness, as the dashboard is working in a dynamic way "
            "you can even add more relevant information and get results by region and date")
st.markdown("##")
data =pd.read_csv(filepath_or_buffer = "happiness_data_dash.csv")
data['Healthy life expectancy'] = data['Healthy life expectancy']*100
data['Logged GDP per capita'] = data['Logged GDP per capita']*10
aggregated_data = data.groupby('Country name').agg({
        'Country name': 'first',
        'Regional indicator': 'first',
        'Ladder score': 'mean',
        'Logged GDP per capita': 'mean',
        'Social support': 'mean',
        'Healthy life expectancy': 'mean',
        'Freedom to make life choices': 'mean',
        'Generosity': 'mean',
        'Perceptions of corruption': 'mean'})
# aggregated_data = data.groupby('Country name')['Ladder score'].mean().reset_index()
st.sidebar.header("Welcome to the project filter section, here you will choose the year you are interested in, "
                  "the regions and the ladder score:")

year_opt = ["All Averaged",2015, 2016, 2017, 2018, 2019, 2020]
year_selected = st.sidebar.selectbox(label="Select year:",
                                     options=year_opt  ,
                                     index=0,
                                     key="year_select"
                                     )

if year_selected == "All Averaged":
    data_year = data.groupby('Country name').agg({
        'Country name': 'first',
        'Regional indicator': 'first',
        'Year':'first',
        'Ladder score': 'mean',
        'Logged GDP per capita': 'mean',
        'Social support': 'mean',
        'Healthy life expectancy': 'mean',
        'Freedom to make life choices': 'mean',
        'Generosity': 'mean',
        'Perceptions of corruption': 'mean'
        # 'Dystopia Residual': 'mean'
        # 'lat': 'mean',
        # 'long': 'mean'
    })
    data_year = data_year.sort_values(by='Ladder score', ascending=False)
else:
    data_year = data[data['Year'] == year_selected]
region_selected = st.sidebar.multiselect(label = "select region:",
                                         options = data_year['Regional indicator'].unique(),
                                         default = data_year['Regional indicator'].unique())
if len(region_selected) == 0:
    region_selected = ["Middle East and Northern Africa"]
data_regions = data_year.query("`Regional indicator` == @region_selected")

median_ladderScore = np.mean(data_regions['Ladder score'])
howMany_Ladders = "\U0001fa9c" * int(median_ladderScore)
mean_lifeExpectancy = np.mean(data_regions['Healthy life expectancy'])
happiest_country = data_regions[data_regions['Ladder score'] == np.max(data_regions['Ladder score'])]['Country name'].item()
st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">Quick world happiness information</p>', unsafe_allow_html=True)
lifexpectancy = data_regions[data_regions['Country name'] == happiest_country]['Healthy life expectancy'].item()
ladder_country = data_regions[data_regions['Country name'] == happiest_country]['Ladder score'].item()

left_column, middle_column = st.columns(2)

with left_column:
    st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">Average Ladder Score for regions:</p>', unsafe_allow_html=True)
    st.subheader(body=f"{median_ladderScore: .3f}")

with middle_column:
    st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">Average Life Expectancy for regions:</p>', unsafe_allow_html=True)
    st.subheader(body =  f"{mean_lifeExpectancy: .2f} Years old")
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.markdown(body = '<p style="font-size: 20px">Happiest Country by year and region:</p>', unsafe_allow_html=True)
    st.subheader(body=f"{happiest_country}")
with middle_column:
    st.markdown(body='<p style="font-size: 20px">Ladder score for happiest Country:</p>', unsafe_allow_html=True)
    st.subheader(body=f"{round(ladder_country, 3)}")
with right_column:
    st.markdown(body='<p style="font-size: 20px">Life Expectancy for happiest Country:</p>', unsafe_allow_html=True)
    st.subheader(body=f"{round(lifexpectancy, 3)} Years old")

st.markdown("---")
# Create the choropleth map
fig = px.choropleth(data_regions, locations='Country name', locationmode='country names',
                    color='Ladder score', hover_name='Country name',
                    title='Ladder Score for country by year',
                    color_continuous_scale='YlOrRd')

# Configure the layout
fig.update_layout(geo=dict(showframe=False, showcoastlines=False))

# Display the heatmap
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.markdown(body = '<p style="font-family:sans-serif; font-size: 24px;">Insights by regions and year, filter on the '
                   'left</p>', unsafe_allow_html=True)

numer_of_results = data_regions.shape[0]
slider_range = st.sidebar.slider(label = 'Choose Ladder score:',value = (2,9),min_value=2,max_value=9)
data_regions = data_regions[data_regions['Ladder score'].between(slider_range[0],slider_range[1])]
numer_of_results2 = data_regions.shape[0]
feature1 = st.selectbox("Select Feature for both plots",
                        options=['Ladder score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                                 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])

left_column, middle_column = st.columns(2)
with left_column:
    feature2 = st.selectbox("Select Feature for scatterplot", options=['Logged GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption'])
fig_gdp_lifeExpect = px.scatter(data_frame=data_regions,
                                x=f"{feature1}",
                                y=f"{feature2}",
                                size=data_regions["Ladder score"] - 2,  # Adjusted size
                                color="Regional indicator",
                                hover_name="Country name",
                                size_max=10,
                                title=f'{feature1} and {feature2}',
                                template='ggplot2')
if feature1 == "Ladder score":
    range = [data_regions[feature1].min(), data_regions[feature1].max() + 2]
else:
    range = [data_regions[feature1].min(), data_regions[feature1].max()]
fig_gdp_lifeExpect.update_layout(
    xaxis=dict(
        showgrid=False,
        range=range  # Set the range to match the data range
    ),
    annotations=[
        dict(
            x=-0.14,
            y=1.1,
            text="Press twice on the plot to get the whole range of countries \n "
                 "or select the part you find interesting",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=10)
        )
    ]
)
data_bar = data_regions.sort_values(by=feature1, ascending=False)
fig_ladderByCountry = px.bar(data_frame=data_bar, x=feature1, y='Country name', orientation='h', title= f'{feature1} by country', template='ggplot2')

fig_ladderByCountry.update_layout(
    xaxis=dict(
        tickmode="linear",
        showgrid=False,
        showticklabels=False  # Hide tick labels on the x-axis
    ),
    yaxis=dict(showgrid=False),
    height=400,  # Adjust the height of the chart as per your preference
    xaxis_title=''
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_gdp_lifeExpect, use_container = True)
right_column.plotly_chart(fig_ladderByCountry, use_container = True)

st.markdown("---")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# # Select the features you want to use for clustering
# features = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
#
# # Perform t-SNE dimensionality reduction
# tsne = TSNE(n_components=2, random_state=42)
# tsne_data = tsne.fit_transform(data_regions[features])
#
# # Create a DataFrame with t-SNE coordinates and country names
# tsne_df = pd.DataFrame(data=tsne_data, columns=['TSNE1', 'TSNE2'])
# tsne_df['Country'] = data_regions['Country name']
#
# # Perform Agglomerative Clustering
# n_clusters = 5
# clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
# tsne_df['Cluster'] = clustering.fit_predict(tsne_df[['TSNE1', 'TSNE2']])
#
# # Define the cluster labels based on your criteria
# cluster_labels = {
#     0: 'Bad',
#     1: 'Barely Ok',
#     2: 'Ok',
#     3: 'Good',
#     4: 'Very Good'
# }
#
# # Assign cluster labels to the data
# tsne_df['Cluster Label'] = tsne_df['Cluster'].map(cluster_labels)
#
# # Create a scatter plot of the clusters
# fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', color='Cluster Label', hover_data='Country')
#
# fig.update_layout(title='Country Clusters (t-SNE)')
#
# # Display the scatter plot in Streamlit
# st.plotly_chart(fig)
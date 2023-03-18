import streamlit as st
import  pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from folium import plugins
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import altair as alt
from math import pi
from PIL import Image
import pickle
from sklearn.ensemble import RandomForestRegressor
import time

# ------------------------------------------------------
st.set_page_config(layout="centered",page_icon="⚽",page_title="FiFA-19")



st.title("FIFA-19")
tab1, tab2= st.tabs(["Data Analysis & Visualization", "Prediction","About Me"])

df = pd.read_csv('fifa19.csv')

with tab1:
    st.header("Data Analysis & Visualization")
    if st.checkbox('Show Data'):
        st.subheader('Data')
        st.write(df)
    if st.checkbox('Describe Data'):
        st.subheader('The Description')
        st.write(df.describe())
# ----------------------------------------------------------------------------------------

    st.subheader("1-Distribution of Age and Value")

    plt.figure(figsize=(20,10))
    st.pyplot(sns.displot(df, x="Age", kind="kde", fill=True))
# ----------------------------------------------------------------------------------------
    st.subheader("2-The Average top 5 Values For Nationality")

    average=df.groupby("Nationality")["Value"].mean().sort_values(ascending=False).head()
    Nation=df.loc[df['Nationality'].isin(['United Arab Emirates','Dominican Republic','Central African Rep.','Egypt','Gabon'])]
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(20,10))
    sns.boxplot(x=Nation['Nationality'],y=Nation['Value'])
    st.pyplot(fig)
# ----------------------------------------------------------------------------------------
    st.subheader("3-Most Expensive Clubs")

    Clubs=df.groupby("Club")["Value"].sum().sort_values(ascending=False).head(10)
    top_clubs=pd.DataFrame(Clubs)
    top_clubs=top_clubs.reset_index(level=0)

    chart1 = px.pie(top_clubs, values='Value', names='Club')
    # Plot!
    st.plotly_chart(chart1, use_container_width=True)
# ----------------------------------------------------------------------------------------
    st.subheader("4-The Top 5 Most Common Nationality")

    top_nations=df['Nationality'].value_counts().head(5)
    Nationality_clubs=pd.DataFrame(top_nations)
    Nationality_clubs=Nationality_clubs.reset_index(level=0)
    Nationality_clubs.columns=["Nationality","Counts"]

    sns.set_theme(style="white")

    # Plot miles per gallon against horsepower with other semantics
    g=sns.relplot(x=Nationality_clubs.Nationality, y=Nationality_clubs.Counts, hue="Nationality", size="Counts",
            sizes=(40, 400), alpha=.7, palette="muted",
            height=6, data=Nationality_clubs)

    st.pyplot(g)
# ----------------------------------------------------------------------------------------
    st.subheader("5-Players Counts per Origin In Map")

    full_nations=pd.read_csv('Map.csv')
    m = folium.Map(tiles="OpenStreetMap", zoom_start=6,control_scale=True)

    for i in range(0,len(full_nations)):
        folium.Circle(
            location=[full_nations.iloc[i]['Latitude'], full_nations.iloc[i]['Longitude']],
            popup=f"{full_nations.iloc[i]['Counts']} Players From {full_nations.iloc[i]['Nationality']}",
            radius=float(full_nations.iloc[i]['Counts'])*500,
            color='crimson',
            fill=True,
            fill_color='crimson'
        ).add_to(m)

    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.TileLayer('Stamen Toner').add_to(m)
    folium.TileLayer('cartodbpositron').add_to(m)
    folium.LayerControl().add_to(m)
    st_data = st_folium(m, width = 725)
#------------------------------------------------
    st.markdown('**Heatmap**.')

    heat_map = folium.Map(tiles="OpenStreetMap",control_scale=True)

    full_nations['Latitude'] = full_nations['Latitude'].astype(float)
    full_nations['Longitude'] = full_nations['Longitude'].astype(float)


    map_values1 = full_nations[['Latitude','Longitude','Counts']]

    dataa = map_values1.values.tolist()

    HeatMap(dataa,min_opacity=0.5, max_opacity=0.9, radius=20,
                    use_local_extrema=False).add_to(heat_map)

    st_data2 = st_folium(heat_map, width = 725)
# ----------------------------------------------------------------------------------------
    st.subheader("6-Top 10 In Each Position By Values")
    lst=df.Position.unique()
    df.Position.nunique()
    RF=df.loc[df.Position=="RF"].head(10).sort_values(by="Value",ascending=False)
    ST=df.loc[df.Position=="ST"].head(10).sort_values(by="Value",ascending=False)
    LW=df.loc[df.Position=="LW"].head(10).sort_values(by="Value",ascending=False)
    GK=df.loc[df.Position=="GK"].head(10).sort_values(by="Value",ascending=False)
    RCM=df.loc[df.Position=="RCM"].head(10).sort_values(by="Value",ascending=False)
    LF=df.loc[df.Position=="LF"].head(10).sort_values(by="Value",ascending=False)
    RS=df.loc[df.Position=="RS"].head(10).sort_values(by="Value",ascending=False)
    RCB=df.loc[df.Position=="RCB"].head(10).sort_values(by="Value",ascending=False)
    LCM=df.loc[df.Position=="LCM"].head(10).sort_values(by="Value",ascending=False)
    CB=df.loc[df.Position=="CB"].head(10).sort_values(by="Value",ascending=False)
    LDM=df.loc[df.Position=="LDM"].head(10).sort_values(by="Value",ascending=False)
    CAM=df.loc[df.Position=="CAM"].head(10).sort_values(by="Value",ascending=False)
    CDM=df.loc[df.Position=="CDM"].head(10).sort_values(by="Value",ascending=False)
    LS=df.loc[df.Position=="LS"].head(10).sort_values(by="Value",ascending=False)
    LCB=df.loc[df.Position=="LCB"].head(10).sort_values(by="Value",ascending=False)
    RM=df.loc[df.Position=="RM"].head(10).sort_values(by="Value",ascending=False)
    LAM=df.loc[df.Position=="LAM"].head(10).sort_values(by="Value",ascending=False)
    LM=df.loc[df.Position=="LM"].head(10).sort_values(by="Value",ascending=False)
    LB=df.loc[df.Position=="LB"].head(10).sort_values(by="Value",ascending=False)
    RDM=df.loc[df.Position=="RDM"].head(10).sort_values(by="Value",ascending=False)
    RW=df.loc[df.Position=="RW"].head(10).sort_values(by="Value",ascending=False)
    CM=df.loc[df.Position=="CM"].head(10).sort_values(by="Value",ascending=False)
    RB=df.loc[df.Position=="RB"].head(10).sort_values(by="Value",ascending=False)
    RAM=df.loc[df.Position=="RAM"].head(10).sort_values(by="Value",ascending=False)
    CF=df.loc[df.Position=="CF"].head(10).sort_values(by="Value",ascending=False)
    RWB=df.loc[df.Position=="RWB"].head(10).sort_values(by="Value",ascending=False)
    LWB=df.loc[df.Position=="LWB"].head(10).sort_values(by="Value",ascending=False)
    #-----------RF------------
    if st.checkbox('Show Most expensive in RF Postion'):
        chart_RF = px.bar(RF, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RF.update_traces(textposition='outside')
        chart_RF.update_layout(title_text='Most expensive in RF Postion')
        st.plotly_chart(chart_RF, use_container_width=True)
    #-----------ST------------
    if st.checkbox('Show Most expensive in ST Postion'):
        chart_ST = px.bar(ST, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_ST.update_traces(textposition='outside')
        chart_ST.update_layout(title_text='Most expensive in ST Postion')
        st.plotly_chart(chart_ST, use_container_width=True)
    #-----------LW------------
    if st.checkbox('Show Most expensive in LW Postion'):
        chart_LW = px.bar(LW, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LW.update_traces(textposition='outside')
        chart_LW.update_layout(title_text='Most expensive in LW Postion')
        st.plotly_chart(chart_LW, use_container_width=True)
    #-----------GK------------
    if st.checkbox('Show Most expensive in GK Postion'):
        chart_GK = px.bar(GK, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_GK.update_traces(textposition='outside')
        chart_GK.update_layout(title_text='Most expensive in GK Postion')
        st.plotly_chart(chart_GK, use_container_width=True)
    #-----------RCM------------
    if st.checkbox('Show Most expensive in RCM Postion'):
        chart_RCM = px.bar(RCM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RCM.update_traces(textposition='outside')
        chart_RCM.update_layout(title_text='Most expensive in RCM Postion')
        st.plotly_chart(chart_RCM, use_container_width=True)
    #-----------LF------------
    if st.checkbox('Show Most expensive in LF Postion'):
        chart_LF = px.bar(LF, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LF.update_traces(textposition='outside')
        chart_LF.update_layout(title_text='Most expensive in LF Postion')
        st.plotly_chart(chart_LF, use_container_width=True)
    #-----------RS------------
    if st.checkbox('Show Most expensive in RS Postion'):
        chart_RS = px.bar(RS, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RS.update_traces(textposition='outside')
        chart_RS.update_layout(title_text='Most expensive in RS Postion')
        st.plotly_chart(chart_RS, use_container_width=True)
    #-----------RCB------------
    if st.checkbox('Show Most expensive in RCB Postion'):
        chart_RCB = px.bar(RCB, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RCB.update_traces(textposition='outside')
        chart_RCB.update_layout(title_text='Most expensive in RCB Postion')
        st.plotly_chart(chart_RCB, use_container_width=True)
    #-----------LCM------------
    if st.checkbox('Show Most expensive in LCM Postion'):
        chart_LCM = px.bar(LCM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LCM.update_traces(textposition='outside')
        chart_LCM.update_layout(title_text='Most expensive in LCM Postion')
        st.plotly_chart(chart_LCM, use_container_width=True)
    #-----------CB------------
    if st.checkbox('Show Most expensive in CB Postion'):
        chart_CB = px.bar(CB, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_CB.update_traces(textposition='outside')
        chart_CB.update_layout(title_text='Most expensive in CB Postion')
        st.plotly_chart(chart_CB, use_container_width=True)
    #-----------LDM------------
    if st.checkbox('Show Most expensive in LDM Postion'):
        chart_LDM = px.bar(LDM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LDM.update_traces(textposition='outside')
        chart_LDM.update_layout(title_text='Most expensive in LDM Postion')
        st.plotly_chart(chart_LDM, use_container_width=True)
    #-----------CAM------------
    if st.checkbox('Show Most expensive in CAM Postion'):
        chart_CAM = px.bar(CAM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_CAM.update_traces(textposition='outside')
        chart_CAM.update_layout(title_text='Most expensive in CAM Postion')
        st.plotly_chart(chart_CAM, use_container_width=True)
    #-----------CDM------------
    if st.checkbox('Show Most expensive in CDM Postion'):
        chart_CDM = px.bar(CDM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_CDM.update_traces(textposition='outside')
        chart_CDM.update_layout(title_text='Most expensive in CDM Postion')
        st.plotly_chart(chart_CDM, use_container_width=True)
    #-----------LS------------
    if st.checkbox('Show Most expensive in LS Postion'):
        chart_LS = px.bar(CDM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LS.update_traces(textposition='outside')
        chart_LS.update_layout(title_text='Most expensive in LS Postion')
        st.plotly_chart(chart_LS, use_container_width=True)
    #-----------LCB------------
    if st.checkbox('Show Most expensive in LCB Postion'):
        chart_LCB = px.bar(LCB, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LCB.update_traces(textposition='outside')
        chart_LCB.update_layout(title_text='Most expensive in LCB Postion')
        st.plotly_chart(chart_LCB, use_container_width=True)
    #-----------RM------------
    if st.checkbox('Show Most expensive in RM Postion'):
        chart_RM = px.bar(RM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RM.update_traces(textposition='outside')
        chart_RM.update_layout(title_text='Most expensive in RM Postion')
        st.plotly_chart(chart_RM, use_container_width=True)
    #-----------LAM------------
    if st.checkbox('Show Most expensive in LAM Postion'):
        chart_LAM = px.bar(LAM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LAM.update_traces(textposition='outside')
        chart_LAM.update_layout(title_text='Most expensive in LAM Postion')
        st.plotly_chart(chart_LAM, use_container_width=True)
    #-----------LM------------
    if st.checkbox('Show Most expensive in LM Postion'):
        chart_LM = px.bar(LM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LM.update_traces(textposition='outside')
        chart_LM.update_layout(title_text='Most expensive in LM Postion')
        st.plotly_chart(chart_LM, use_container_width=True)
    #-----------LB------------
    if st.checkbox('Show Most expensive in LB Postion'):
        chart_LB = px.bar(LB, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LB.update_traces(textposition='outside')
        chart_LB.update_layout(title_text='Most expensive in LB Postion')
        st.plotly_chart(chart_LB, use_container_width=True)
    #-----------RDM------------
    if st.checkbox('Show Most expensive in RDM Postion'):
        chart_RDM = px.bar(RDM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RDM.update_traces(textposition='outside')
        chart_RDM.update_layout(title_text='Most expensive in RDM Postion')
        st.plotly_chart(chart_RDM, use_container_width=True)
    #-----------RW------------
    if st.checkbox('Show Most expensive in RW Postion'):
        chart_RW = px.bar(RW, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RW.update_traces(textposition='outside')
        chart_RW.update_layout(title_text='Most expensive in RW Postion')
        st.plotly_chart(chart_RW, use_container_width=True)
    #-----------CM------------
    if st.checkbox('Show Most expensive in CM Postion'):
        chart_CM = px.bar(CM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_CM.update_traces(textposition='outside')
        chart_CM.update_layout(title_text='Most expensive in CM Postion')
        st.plotly_chart(chart_CM, use_container_width=True)
    #-----------RB------------
    if st.checkbox('Show Most expensive in RB Postion'):
        chart_RB = px.bar(RB, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RB.update_traces(textposition='outside')
        chart_RB.update_layout(title_text='Most expensive in RB Postion')
        st.plotly_chart(chart_RB, use_container_width=True)
    #-----------RAM------------
    if st.checkbox('Show Most expensive in RAM Postion'):
        chart_RAM = px.bar(RAM, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RAM.update_traces(textposition='outside')
        chart_RAM.update_layout(title_text='Most expensive in RAM Postion')
        st.plotly_chart(chart_RAM, use_container_width=True)
    #-----------CF------------
    if st.checkbox('Show Most expensive in CF Postion'):
        chart_CF = px.bar(CF, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_CF.update_traces(textposition='outside')
        chart_CF.update_layout(title_text='Most expensive in CF Postion')
        st.plotly_chart(chart_CF, use_container_width=True)
    #-----------RWB------------
    if st.checkbox('Show Most expensive in RWB Postion'):
        chart_RWB = px.bar(RWB, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_RWB.update_traces(textposition='outside')
        chart_RWB.update_layout(title_text='Most expensive in RWB Postion')
        st.plotly_chart(chart_RWB, use_container_width=True)
    #-----------LWB------------
    if st.checkbox('Show Most expensive in LWB Postion'):
        chart_LWB = px.bar(LWB, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
        chart_LWB.update_traces(textposition='outside')
        chart_LWB.update_layout(title_text='Most expensive in LWB Postion')
        st.plotly_chart(chart_LWB, use_container_width=True)

# ----------------------------------------------------------------------------------------
    st.subheader("7-Top 10 by Values in All Positions")
    top=df.head(10).sort_values(by="Value",ascending=False)
    topiest = px.bar(top, y='Name', x='Value',color="Nationality", labels={'Value':'Value in Euro (€)'},
             text_auto=True,orientation="h")
    topiest.update_traces(textposition='outside')
    topiest.update_layout(title_text='Most expensive in All Postion')

    st.plotly_chart(topiest, use_container_width=True)
# ----------------------------------------------------------------------------------------
    st.subheader("8- Most expensive in All Position by Age")
    chart_data = top[["Name","Nationality","Age"]]

    chart = (
    alt.Chart(chart_data)
    .mark_bar()
    .encode(
        alt.X("Name"),
        alt.Y("Age"),
        alt.Color("Nationality",scale=alt.Scale(scheme='dark2')),
        alt.Tooltip(["Name", "Age"]),
    )
    .interactive()
    )
    st.altair_chart(chart,use_container_width=True)

# ----------------------------------------------------------------------------------------
    st.subheader("9- Top Players Overall")
    Top_Overall=df.head(10).sort_values(by="Overall",ascending=False)
    data = dict(
    number=Top_Overall.Overall,
    stage=Top_Overall.Name)
    Top_Overalls = px.funnel(data, x='number', y='stage',color=Top_Overall.Nationality,
                title="Top Players Overall",labels={"stage":"Players"},)

    st.plotly_chart(Top_Overalls, use_container_width=True)
# ----------------------------------------------------------------------------------------
    st.subheader("10-Radar for the top 10 players overall")
    image = Image.open('download.png')
    st.image(image, caption='top 10 players overall')

# ----------------------------------------------------------------------------------------
    st.subheader("11-The most expensive player for Positions")
    positions=[RF,ST,LW,GK,RCM,LF,RS,RCB,LCM,CB,LDM,CAM,CDM,LS,LCB,RM,LAM,LM,LB,RDM,RW,CM,RB,RAM,CF,RWB,LWB]
    keys=[]
    values=[]
    for i in positions:
        keys.append(" ".join(str(i.loc[i.Value==max(i.Value)]["Name"]).split()[1:3]))
        values.append(max(i.Value))

    players=['L.Messi','H.Kane','Neymar Jr','De Gea','K.De Bruyne','E. Hazard','L.Suárez','Sergio Ramos','T.Kroos','S.Umtiti',
    'N.Kanté','A.Griezmann','Casemiro','E.Cavani','K.Koulibaly','K. Mbappé','J.Rodríguez','P.Aubameyang','Marcelo',
    'P.Pogba','Bernardo Silva','S.Milinković-Savić','Azpilicueta','H.Ziyech','Luis Alberto','M.Ginter',
    'N.Schulz']
    Position_index=['RF','ST','LW','GK','RCM','LF','RS','RCB','LCM',
           'CB','LDM','CAM','CDM','LS','LCB','RM','LAM','LM',
           'LB','RDM','RW','CM','RB','RAM','CF','RWB','LWB']
    expensive_players=pd.DataFrame({"Players":players,"Values(€)":values,"Position":Position_index})

    final = px.sunburst(expensive_players, path=['Players', 'Position'], values='Values(€)', color='Position')
    final.update_layout(
                    title_font_size = 25,
                    font_size = 18,
                    title_font_family = 'Arial')

    st.plotly_chart(final, use_container_width=True)


with tab2:
    st.header("Prediction")
    st.markdown("After trying the algorithms to determine the best algorithm to predict the player's price we find the Best algorithm is Random Forest algorithm.")
    image2 = Image.open('download (1).png')
    st.image(image2, caption='Random Forest algorithm')
    st.text('- Random Forest Accuracy : 98.16%')
    st.subheader("How does Prediction Work?")
    st.text('1- Make sure that the entries are written correctly')
    st.text('2- press enter in last input field then wait the result')
    st.write("-----------------------------------------")
    try:
        Age = st.text_input('Age of player', placeholder='input here age of player ex: 22')
        overall_player = st.text_input('Overall of player', placeholder='input here Overall of player ex: 90')
        position_player = st.text_input('position of player', placeholder='input here position of player ex: RW ').upper()


        position_dict={'RF':0,'ST':1,'LW':2,'GK':3,'RCM':4,'LF':5,'RS':6,'RCB':7,'LCM':8,
            'CB':9,'LDM':10,'CAM':11,'CDM':12,'LS':13,'LCB':14,'RM':15,'LAM':16,'LM':17,
            'LB':18,'RDM':19,'RW':20,'CM':21,'RB':22,'RAM':23,'CF':24,'RWB':25,'LWB':26}

        with open('model_pickle.pkl', 'rb') as f:
            model_RF = pickle.load(f)

    

        time.sleep(1.1)
        results=model_RF.predict([[Age,overall_player,position_dict[position_player]]])
        st.write('Predicted Price is : '+ str(results[0].round())+"€")
    except :
        st.write('Check From Your Inputs')


# with tab3:
#     st.header("How I Am ?")
#     st.text('My Name is Abdelhamid Adel, Data scientist')
#     st.text('I am From Cairo, Egypt')
#     st.text('Github : https://github.com/AbdelhamidADel')
#     image3 = Image.open('qrcode.png')
#     st.image(image3, caption='MY Github QR Code')

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
st.markdown(hide_st_style, unsafe_allow_html=True)


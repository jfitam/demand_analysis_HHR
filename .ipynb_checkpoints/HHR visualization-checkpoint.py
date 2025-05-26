######################################################################################################################################
## IMPORT ##
import streamlit as st
import scripts.render as pages 
import pandas as pd
from scripts.features_utils import get_data


# format some fields
def format_df(df):
    df['train_departure_date_short'] = pd.to_datetime(df['train_departure_date_short'])
    df['passengers'] = df['passengers'].astype(float)
    df['number_of_services'] = df['number_of_services'].astype(int)
    df['revenue'] = df['revenue'].astype(float)
    df['revenue_without_promotion'] = df['revenue_without_promotion'].astype(float)
    df['passengers_km'] = df['passengers_km'].astype(float)
    df['price_mix_ratio'] = df['price_mix_ratio'].astype(float)

    return df

####################################################################################################################################################
#MAIN PROGRAM

# get the data from the database
df = get_data()

# format 
df = format_df(df)

# layout
st.set_page_config(layout="wide")
st.title("Passenger Analysis")

# filters
date_range = []
with st.sidebar:
    st.header("Filters")
    date_range = st.date_input("Date Range", [])
    plot_selections = st.multiselect(
        "Select the Models to Plot", 
        ["Original Data", "Linear Regression", "Holt Winters", "Arima", "LightGBM"],
        default=["Original Data", "Linear Regression", "Holt Winters", "Arima", "LightGBM"])
    show_past = st.checkbox("Show past predictions for models", value=False)
    forecast_days = st.sidebar.slider("Number of Days for Forecasting", min_value=1, max_value=200, value=14)


    
# different tabs of the app
tab_dashboard, tab_forecast, tab_evaluation = st.tabs(['Dashboard', "Daily Forecast", "Evaluation of the Models"])

with tab_dashboard:
    pages.render_dashboard(df, date_range)

with tab_forecast:
    pages.render_forecast(df, date_range, plot_selections, forecast_days, show_past)

with tab_evaluation:
    pages.render_evaluation(df, date_range, plot_selections, forecast_days, show_past)

    
 
        
    
import streamlit as st
from pathlib import Path
import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import altair as alt

    
from scripts.features_utils import get_features_arima, select_features_arima, get_data

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

DATE_COLUMN = 'train_departure_date_short'
LAST_DATE_DATA = pd.to_datetime('2024-08-06')

####################################################################################################################################################
# FUNCTION TO GET THE MODELS ALREADY TRAINED FROM THE SOURCE FILES. PYTHON OBJECTS OF THE FINAL MODELS WERE SAVED TO FILES USING THE JOBLIB 

def load_models():
    pipeline_lgbm   = joblib.load(os.path.join(ROOT_DIR, "models", "pipeline_lgbm.pkl"))
    pipeline_lr = joblib.load(os.path.join(ROOT_DIR, "models", "pipeline_linear_regression.pkl"))
    model_hw    = joblib.load(os.path.join(ROOT_DIR, "models", "final_model_holt_winters.pkl"))
    model_arima = joblib.load(os.path.join(ROOT_DIR, "models", "model_arima.pkl"))
    
    return pipeline_lgbm, pipeline_lr, model_hw, model_arima

###################################################################################################################################################
# FILTERING

# Auxiliar function to handle the parameters pass for the dates
# could be none for no filter, 1 for date from only, and 2 elements for both dates
# return None for any date missing
def get_filter_dates(date_range):
    if len(date_range) == 2:
        date_from, date_to = date_range
    elif len(date_range) == 1:
        date_from = date_range[0]
        date_to = None  
    else:
        date_from = date_to = None

    return date_from, date_to

# Filter the df according to the parameters passed
def filter_df(df_to_filter, date_range):
    #breakdown
    date_from, date_to = get_filter_dates(date_range)

    if date_from:
        df_to_filter = df_to_filter[df_to_filter[DATE_COLUMN] >= pd.to_datetime(date_from)]
        
    if date_to:
        df_to_filter = df_to_filter[df_to_filter[DATE_COLUMN] <= pd.to_datetime(date_to)]

    return df_to_filter
    

###################################################################################################################################################
# GROUP OF FUNCTIONS THAT RENDER THE DIFFERENT PAGES OF THE LAYOUT

# RENDER THE DASHBOARD FOR THE ORIGINAL DATA
def render_dashboard(df, date_range):
    # to not modify the original df
    df_original = df.copy()
    
    #remove bookings in incomplete days
    df_to_render = filter_df(df_original, [False, LAST_DATE_DATA])
    df_to_render = filter_df(df_to_render, date_range)
    
    #check if there is data after the filter
    if df_to_render.empty:
        st.warning("No data to show. Please, select different filters.")
        return
            
    # passengers chart
    passengers_chart = alt.Chart(df_to_render).mark_line().encode(
        x=alt.X("train_departure_date_short:T", title="Date"),
        y=alt.Y("passengers:Q", title="Passengers"),
        color=alt.value("red"),
        tooltip=["train_departure_date_short", "passengers"]
    ).properties(height=150).interactive()

    #load factor chart
    df_to_render['load_factor'] = round(df_to_render['passengers_km'].astype(float) / df_to_render['seats_km'].astype(float), 2)
    load_factor_chart = alt.Chart(df_to_render).mark_line().encode(
        x=alt.X("train_departure_date_short:T", title="Date"),
        y=alt.Y("load_factor", title="Load Factor", axis=alt.Axis(title="Load Factor", format='%')),
        color=alt.value("blue"),
        tooltip=["train_departure_date_short", "load_factor"]
    ).properties(height=150).interactive()
    
    # chart for weekdays
    df_to_render['week_day'] = df_to_render['train_departure_date_short'].dt.day_name()
    day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    
    weekday_chart = alt.Chart(df_to_render).mark_boxplot(color="white").encode(
        x=alt.X('week_day:N', title="Week Day", sort=day_order),
        y=alt.Y('passengers:Q', title="Passengers"),
        color=alt.Color('week_day:N', title="Week Day", legend=None),
        tooltip=[alt.Tooltip("train_departure_date_short:T", title="Date"), "passengers"]
    ).properties(title="Passenger Distribution by Weekday")

    # chart for nationalities
    nationalities_cols = [
        "tickets_saudi",
        "tickets_egyptian",
        "tickets_pakistani",
        "tickets_indian",
        "tickets_yemeni",
        "tickets_indonesian",
        "tickets_jordanian",
        "tickets_USA", 
        "tickets_britain"]
    df_to_render['other_nationalities'] = df_to_render['passengers'] - df_to_render[nationalities_cols].sum(axis=1)
    
    df_nationalities = df_to_render.melt(
        id_vars='train_departure_date_short',
        value_vars=nationalities_cols,
        var_name='nationality',
        value_name='passengers_nationality'
    )

    df_nationalities["month"] = df_nationalities["train_departure_date_short"].dt.to_period("M").dt.to_timestamp()
    df_nationalities = df_nationalities[['month', 'nationality', 'passengers_nationality']].groupby(['month','nationality']).sum().reset_index()

    nationalities_chart = alt.Chart(df_nationalities).mark_line().encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("passengers_nationality:Q", title="Passengers"),
        color=alt.Color("nationality:N", title="Nationality"),
        tooltip=[alt.Tooltip("month:T", title="Month"), "nationality:N", "passengers_nationality:Q"]
    ).properties(title="Passenger Volume by Nationality").interactive()

    
    # first line: combine together passengers and load factor
    st.altair_chart(alt.vconcat(passengers_chart, load_factor_chart).resolve_scale(x='shared'))
    
    #the seecond part is divided by columns: first for weekday, second for nationalities
    col1, col2 = st.columns([0.3,0.7])
    with col1:
        st.altair_chart(weekday_chart)
    with col2:
        st.altair_chart(nationalities_chart)





# chart with the forecasts for the different models
def render_forecast(df, date_range, plot_selections, forecast_days, show_past):

    # to not modify the original df
    df_original = df.copy()
    
    # last forecast day
    last_day_forecasted = LAST_DATE_DATA + pd.Timedelta(days=forecast_days)
    
    # load the previously trained models to show
    pipeline_lgbm, pipeline_lr, model_hw, model_arima = load_models()
    
    # prepare the data for the plot based on the selection
    df_plot = pd.DataFrame()
    
    # Original data
    if "Original Data" in plot_selections:
        df_temp = pd.DataFrame()
        df_temp = df_original[["train_departure_date_short", "passengers"]].copy()
        df_temp["Model"] = "Original Data"
        df_temp.rename(columns={"passengers": "value"}, inplace=True)
        df_temp = filter_df(df_temp, [False, LAST_DATE_DATA]) # remove bokings on days in the future
        df_plot = pd.concat([df_temp, df_plot])
    
    # Linear Regression
    if "Linear Regression" in plot_selections:
        #predict
        df_temp = pd.DataFrame()
        df_temp = df_original[["train_departure_date_short"]].copy()
        df_temp["value"] = pipeline_lr.predict(df_original) 
        df_temp["Model"] = "Linear Regression"

        # remove the past values if not requested
        if not show_past:
            df_temp = filter_df(df_temp, [LAST_DATE_DATA, False])
        #merge
        df_plot = pd.concat([df_temp, df_plot])
    
    # Holt Winters
    if "Holt Winters" in plot_selections:        
        # prepare the dataframe
        df_temp = pd.DataFrame()
    
        # get future forecast
        forecast = model_hw.forecast(steps=forecast_days)
        df_temp["value"] = forecast.values
        df_temp["train_departure_date_short"] = forecast.index
        df_temp["Model"] = "Holt Winters"
    
        df_plot = pd.concat([df_plot, df_temp])
    
        
    # Arima
    if "Arima" in plot_selections:
        # get the future dates
        is_future = df_original["train_departure_date_short"] > LAST_DATE_DATA
        
        # prepare the dataframe
        df_temp = df_original[is_future][["train_departure_date_short"]][:forecast_days].copy()
    
        # get future forecast
        df_exog = get_features_arima(df_original)
        exog_future = select_features_arima(df_exog[is_future])
        forecast = model_arima.forecast(steps=forecast_days, exog=exog_future[:forecast_days])
        df_temp["value"] = forecast.values
        df_temp["Model"] = "Arima"

    
        df_plot = pd.concat([df_plot, df_temp])
    
    
    # Gradient Boosting
    if "LightGBM" in plot_selections:
        df_temp = pd.DataFrame()
        df_temp = df_original[["train_departure_date_short"]].copy()
        df_temp["value"] = pipeline_lgbm.predict(df_original) 
        df_temp["Model"] = "LightGBM"
        
        # remove the past values if not requested
        if not show_past:
            df_temp = filter_df(df_temp, [LAST_DATE_DATA, False])
            
        #merge
        df_plot = pd.concat([df_temp, df_plot])

    if plot_selections:
        # filt data
        if len(date_range) > 1 and last_day_forecasted.date() > date_range[1]:
            st.warning("Some forecasted data is not shown due to filters.")
        else:
            df_plot = filter_df(df_plot, [False, last_day_forecasted])
        
        df_plot = filter_df(df_plot, date_range)
        
        #check if there is data after the filter
        if df_plot.empty:
            st.warning("No data to show. Please, select different filters.")
            return

        #check if there is any missing models due to the filters
        for model in plot_selections:
            if model not in df_plot['Model'].unique():
                st.warning(f"The data for {model} is not appearing in the chart due to filter restrictions")
        
        st.divider()
        chart = alt.Chart(df_plot).mark_line().encode(
            x=alt.X('train_departure_date_short:T', title="Date"),
            y=alt.Y('value:Q', 
                    #scale=alt.Scale(domain=[0, 24000]), 
                    title="Passengers"),
            color=alt.Color('Model:N', title="Legend") 
        ).properties(height=800).interactive(bind_y=False)
        
        st.altair_chart(chart, use_container_width=True)
        st.write(f"Last real data available: {LAST_DATE_DATA}")
        st.write(f"Forecasted until: {last_day_forecasted}")
    else:
        st.info("Please, select data to plot at sidebar menu.")

##########################################################################################################################################
# function for the evaluation page 
def render_evaluation(df, date_range, plot_selections, forecast_days, show_past):

    # keep the original data
    df_original = df.copy()

    # last forecast day
    last_day_forecasted = LAST_DATE_DATA + pd.Timedelta(days=forecast_days)
    
    # load the previously trained models to show
    pipeline_lgbm, pipeline_lr, model_hw, model_arima = load_models()
    
    # prepare the data for the plot based on the selection
    df_plot = pd.DataFrame()
    
    # Linear Regression
    if "Linear Regression" in plot_selections:
        df_temp = df_original[["train_departure_date_short","passengers"]].copy()
        df_temp["residuals"] = pipeline_lr.predict(df_original) - df_temp["passengers"].values
        df_temp["Model"] = "Linear Regression"
        df_plot = pd.concat([df_temp, df_plot])
    
    # Holt Winters
    if "Holt Winters" in plot_selections:
        
        # prepare the dataframe
        df_temp = pd.DataFrame()
    
        # get future forecast
        forecast = model_hw.forecast(steps=forecast_days)
        df_temp["train_departure_date_short"] = forecast.index
        df_temp = df_temp.merge(df_original[["train_departure_date_short", "passengers"]], on="train_departure_date_short", how="left")
        df_temp["residuals"] = forecast.values  - df_temp["passengers"]
        df_temp["Model"] = "Holt Winters"
    
        df_plot = pd.concat([df_plot, df_temp])
    
        
    # Arima
    if "Arima" in plot_selections:
        # get the future dates
        is_future = df_original["train_departure_date_short"] > LAST_DATE_DATA
        
        # prepare the dataframe
        df_temp = df_original[is_future][["train_departure_date_short", "passengers"]][:forecast_days].copy()
    
        # get the features
        df_exog = get_features_arima(df_original)
        exog_future = select_features_arima(df_exog[is_future][:forecast_days])

        #prediction
        forecast = model_arima.forecast(steps=forecast_days, exog=exog_future)  
        df_temp["residuals"] = forecast.values - df_temp["passengers"]
        df_temp["Model"] = "Arima"
    
        df_plot = pd.concat([df_plot, df_temp])
    
    
    # Gradient Boosting
    if "LightGBM" in plot_selections:
        df_temp = pd.DataFrame()
        df_temp = df_original[["train_departure_date_short", "passengers"]].copy()
        df_temp["residuals"] = pipeline_lgbm.predict(df_original)  - df_temp["passengers"]
        df_temp["Model"] = "LightGBM"
        df_plot = pd.concat([df_temp, df_plot])

    if plot_selections:
        # remove the undesired entries in the df
        df_plot = filter_df(df_plot, date_range)
        df_plot = filter_df(df_plot, [False, last_day_forecasted])

        # remove the past values if required
        if not show_past:
            df_plot = filter_df(df_plot, [LAST_DATE_DATA, False])
        
        #check if there is data after the filtering
        if df_plot.empty:
            st.warning("No data to show. Please, select different filters.")
            return
    
        # evaluation table and metrics bar chart
        st.subheader("Metrics")
        st.divider()
        metrics = []

        # reconstruct the predictions
        df_eval = df_plot
        df_eval['value'] = df_eval['passengers'] + df_eval['residuals']

        # group by model to get the metrics for each one of them
        for model, group in df_eval.groupby("Model"):
            y_true = group["passengers"]
            y_pred = group["value"]
        
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        
            metrics.append({
                "Model": model,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R²": round(r2, 3)
            })

        # show
        df_metrics = pd.DataFrame(metrics)
        st.dataframe(metrics)

        st.markdown("###") 

        # bar chart with the mtrics
        df_melt = df_metrics.melt(
            id_vars="Model",
            value_vars=["RMSE", "MAE", "R²"],
            var_name="Metric",
            value_name="Value"
        )

        metrics_bars = alt.Chart(df_melt).mark_bar().encode(
            x=alt.X("Model:N"),
            y=alt.Y("Value:Q"),
            color=alt.Color("Model:N"),
            column=alt.Column("Metric:N", title=None),
            tooltip=["Model", "Metric", "Value"]
        ).properties(height=300)

        metrics_bars = metrics_bars.resolve_scale(y='independent')

        st.altair_chart(metrics_bars)


        # chart for residuals
        st.subheader("Residuals")
        st.divider()
        residuals_chart = alt.Chart(df_plot).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X('train_departure_date_short:T', title="Date"),
            y=alt.Y('residuals:Q', 
                    #scale=alt.Scale(domain=[0, 24000]), 
                    title="Residuals of the model"),
            color=alt.Color('Model:N', title="Legend") 
        ).properties(height=800).interactive(bind_y=False)
        
        st.altair_chart(residuals_chart, use_container_width=True)
        
    else:
        st.info("Please, select data to plot at sidebar menu.")
######################################################################################################################################
## IMPORT ##
import streamlit as st
#import psycopg2 as ps
#from sqlalchemy import create_engine
#from sqlalchemy import text
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import altair as alt
from pathlib import Path
import sys
import os

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from scripts.features_utils import get_features_arima, select_features_arima, get_data


####################################################################################################################################################
# FUNCTION TO GET THE MODELS ALREADY TRAINED FROM THE SOURCE FILES. PYTHON OBJECTS OF THE FINAL MODELS WERE SAVED TO FILES USING THE JOBLIB 

def load_models():
    pipeline_lgbm   = joblib.load(os.path.join(ROOT_DIR, "models", "pipeline_lgbm.pkl"))
    pipeline_lr = joblib.load(os.path.join(ROOT_DIR, "models", "pipeline_linear_regression.pkl"))
    model_hw    = joblib.load(os.path.join(ROOT_DIR, "models", "final_model_holt_winters.pkl"))
    model_arima = joblib.load(os.path.join(ROOT_DIR, "models", "model_arima.pkl"))
    
    return pipeline_lgbm, pipeline_lr, model_hw, model_arima


####################################################################################################################################################
#MAIN PROGRAM

# get the data from the database
df = get_data()

# format 
df['train_departure_date_short'] = pd.to_datetime(df['train_departure_date_short'])
df['passengers'] = df['passengers'].astype(float)
df['number_of_services'] = df['number_of_services'].astype(int)
df['revenue'] = df['revenue'].astype(float)
df['revenue_without_promotion'] = df['revenue_without_promotion'].astype(float)
df['passengers_km'] = df['passengers_km'].astype(float)
df['price_mix_ratio'] = df['price_mix_ratio'].astype(float)

# load the previously trained models to show
pipeline_lgbm, pipeline_lr, model_hw, model_arima = load_models()

# layout
st.set_page_config(layout="wide")

# different tabs of the app
page = st.sidebar.radio(
    "Select an option:",
    ['Original data Table', "Forecasts", "Evaluation"]
)

if page == 'Original data Table':
    st.title("Original Data")
    st.divider()

    #write the table
    st.dataframe(df)

    # plot the original dataframe
    df_plot = df.iloc[:, 3:]
    y_cols = st.multiselect("Select Y variables to plot:", df_plot.columns.drop("train_departure_date_short"))

    # Show chart
    if y_cols:
        df_melt = df.melt(id_vars="train_departure_date_short", value_vars=y_cols, var_name="variable", value_name="value")
    
        chart = alt.Chart(df_melt).mark_line().encode(
            x="train_departure_date_short:T",
            y="value:Q",
            color="variable:N"
        ).properties(height=400).interactive()
    
        st.altair_chart(chart, use_container_width=True)

elif page == 'Forecasts':
    # view page
    st.title("Daily Passengers Forecast")
    st.divider()
    
    #selector
    plot_selections = st.multiselect("Select the Models to Plot", ["Original Data", "Linear Regression", "Holt Winters", "Arima", "LightGBM"])
    
    # prepare the data for the plot based on the selection
    df_plot = pd.DataFrame()
    
    # Original data
    if "Original Data" in plot_selections:
        df_temp = pd.DataFrame()
        df_temp = df[["train_departure_date_short", "passengers"]].copy()
        df_temp["Model"] = "Original Data"
        df_temp.rename(columns={"passengers": "value"}, inplace=True)
        df_plot = pd.concat([df_temp, df_plot])
    
    # Linear Regression
    if "Linear Regression" in plot_selections:
        df_temp = pd.DataFrame()
        df_temp = df[["train_departure_date_short"]].copy()
        df_temp["value"] = pipeline_lr.predict(df) 
        df_temp["Model"] = "Linear Regression"
        df_plot = pd.concat([df_temp, df_plot])
    
    # Holt Winters
    if "Holt Winters" in plot_selections:
        # get the future dates
        fitted_end = pd.to_datetime(model_hw.fittedvalues.index).max()
        is_future = df["train_departure_date_short"] > fitted_end
        
        # prepare the dataframe
        df_temp = pd.DataFrame()
    
        # get future forecast
        steps = sum(is_future)
        forecast = model_hw.forecast(steps=steps)
        df_temp["value"] = forecast.values
        df_temp["train_departure_date_short"] = forecast.index
        df_temp["Model"] = "Holt Winters"
    
        df_plot = pd.concat([df_plot, df_temp])
    
        
    # Arima
    if "Arima" in plot_selections:
        # get the future dates
        fitted_end = pd.to_datetime(model_arima.fittedvalues.index).max()
        is_future = df["train_departure_date_short"] > fitted_end
        
        # prepare the dataframe
        df_temp = df[is_future][["train_departure_date_short"]].copy()
    
        # get future forecast
        df_exog = get_features_arima(df)
        exog_future = select_features_arima(df_exog[is_future])
        steps = exog_future.shape[0]
        forecast = model_arima.forecast(steps=steps, exog=exog_future)
        df_temp["value"] = forecast.values
        df_temp["Model"] = "Arima"
    
        df_plot = pd.concat([df_plot, df_temp])
    
    
    # Gradient Boosting
    if "LightGBM" in plot_selections:
        df_temp = pd.DataFrame()
        df_temp = df[["train_departure_date_short"]].copy()
        df_temp["value"] = pipeline_lgbm.predict(df) 
        df_temp["Model"] = "LightGBM"
        df_plot = pd.concat([df_temp, df_plot])

    if plot_selections:
        st.divider()
        chart = alt.Chart(df_plot).mark_line().encode(
            x=alt.X('train_departure_date_short:T', title="Date"),
            y=alt.Y('value:Q', 
                    #scale=alt.Scale(domain=[0, 24000]), 
                    title="Passengers"),
            color=alt.Color('Model:N', title="Legend") 
        ).properties(height=800).interactive(bind_y=False)
        
        st.altair_chart(chart, use_container_width=True)

elif page == 'Evaluation':
    # evaluation page
    
    #selector
    plot_selections = st.multiselect("Select the Models to Plot", ["Linear Regression", "Holt Winters", "Arima", "LightGBM"])
    
    # prepare the data for the plot based on the selection
    df_plot = pd.DataFrame()

    # Original data
    df_original = df.set_index('train_departure_date_short')[['passengers']]
    
    # Linear Regression
    if "Linear Regression" in plot_selections:
        df_temp = pd.DataFrame()
        df_temp = df[["train_departure_date_short"]].copy()
        df_temp["residuals"] = pipeline_lr.predict(df) - df_original["passengers"].reindex(df_temp["train_departure_date_short"].values).values
        df_temp["Model"] = "Linear Regression"
        df_plot = pd.concat([df_temp, df_plot])
    
    # Holt Winters
    if "Holt Winters" in plot_selections:
        # get the future dates
        fitted_end = pd.to_datetime(model_hw.fittedvalues.index).max()
        is_future = df["train_departure_date_short"] > fitted_end
        
        # prepare the dataframe
        df_temp = pd.DataFrame()
    
        # get future forecast
        steps = sum(is_future)
        forecast = model_hw.forecast(steps=steps)
        df_temp["train_departure_date_short"] = forecast.index
        df_temp["residuals"] = forecast.values  - df_original["passengers"].reindex(df_temp["train_departure_date_short"].values).values
        df_temp["Model"] = "Holt Winters"
    
        df_plot = pd.concat([df_plot, df_temp])
    
        
    # Arima
    if "Arima" in plot_selections:
        # get the future dates
        fitted_end = pd.to_datetime(model_arima.fittedvalues.index).max()
        is_future = df["train_departure_date_short"] > fitted_end
        
        # prepare the dataframe
        df_temp = df[is_future][["train_departure_date_short"]].copy()
    
        # get future forecast
        df_exog = get_features_arima(df)
        exog_future = select_features_arima(df_exog[is_future])
        steps = exog_future.shape[0]
        forecast = model_arima.forecast(steps=steps, exog=exog_future)  - df_original["passengers"].reindex(df_temp["train_departure_date_short"].values).values
        df_temp["residuals"] = forecast.values
        df_temp["Model"] = "Arima"
    
        df_plot = pd.concat([df_plot, df_temp])
    
    
    # Gradient Boosting
    if "LightGBM" in plot_selections:
        df_temp = pd.DataFrame()
        df_temp = df[["train_departure_date_short"]].copy()
        df_temp["residuals"] = pipeline_lgbm.predict(df)  - df_original["passengers"].reindex(df_temp["train_departure_date_short"].values).values
        df_temp["Model"] = "LightGBM"
        df_plot = pd.concat([df_temp, df_plot])

    if plot_selections:
        # evaluation table and metrics bar chart
        st.title("Metrics")
        st.divider()
        metrics = []

        # reconstruct the predictions
        df_eval = df_plot.merge(df_original[['passengers']].reset_index(), on='train_departure_date_short', how='left')
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
        st.title("Residuals")
        st.divider()
        residuals_chart = alt.Chart(df_plot).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X('train_departure_date_short:T', title="Date"),
            y=alt.Y('residuals:Q', 
                    #scale=alt.Scale(domain=[0, 24000]), 
                    title="Residuals of the model"),
            color=alt.Color('Model:N', title="Legend") 
        ).properties(height=800).interactive(bind_y=False)
        
        st.altair_chart(residuals_chart, use_container_width=True)

        
    
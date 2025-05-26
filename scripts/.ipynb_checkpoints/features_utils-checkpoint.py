from sklearn.preprocessing import StandardScaler
import pandas as pd
from hijri_converter import Gregorian
import os
import json
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent


######################################################################################################################################################

# get the connection string to establish connection
def get_connection_string():
    import streamlit as st
    try:
        return st.secrets["connection_string"]
    except:
        # Fallback to local secrets file
        import tomllib
        

        secrets_path = os.path.join(Path(__file__).resolve().parent.parent, ".streamlit", "secrets.toml")
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)

        return secrets["connection_string"]


#function to get the data from database. this is shared by all the model files. Returns a pandas dataframe with the data fetched from the database, or taken from a saved file (if any), or None otherwise.
def get_data():
    # establish the connection
    try:
        conn.close()
    except NameError:
        pass
    
    last_refreshed_day = '2025-03-31'
    fpath = os.path.join(BASE_DIR, 'data','data.csv')
    
    conexion_string = get_connection_string()

    data = None
    try:
        from sqlalchemy import create_engine
        from sqlalchemy import text
        
        # pull the data. the aggregation data is per day
        engine = create_engine(conexion_string)
        conn = engine.connect()

        query = text("""
            SELECT corridor_name, train_year, train_week_num, train_departure_date_short,
                   sum(total_occupancy) as passengers,
                   (sum(composition) / 2)::integer as number_of_services,
                   sum(total_amount_wo_vat) as revenue,
                   sum(
                       coalesce(minimum_standard_price,0)
                       +coalesce(intermediate_standard_price,0)
                       +coalesce(maximum_standard_price,0)
                       +coalesce(no_level_standard_price,0)
                    ) as revenue_without_promotion,
                   sum(passengers_km) as passengers_km,
                   sum(seats_km) as seats_km,
                   day_category,
                   ( 
                       (sum(intermediate_tickets) + sum(intermediate_tickets_promo))*0.3
                       + (sum(maximum_tickets) + sum(maximum_tickets_promo)) * 0.6
                   ) / sum(total_occupancy) + 1 as price_mix_ratio,
                   sum(tickets_saudi) as tickets_saudi, 
                   sum(tickets_egyptian) as tickets_egyptian, 
                   sum(tickets_pakistani) as tickets_pakistani, 
                   sum(tickets_indian) as tickets_indian, 
                   sum(tickets_yemeni) as tickets_yemeni, 
                   sum(tickets_indonesian) as tickets_indonesian, 
                   sum(tickets_jordanian) as tickets_jordanian, 
                   sum("tickets_USA") as "tickets_USA", 
                   sum(tickets_britain) as tickets_britain 
            FROM analytics.metrics_class
            WHERE train_departure_date_short <= :last_day AND corridor_name = 'MAK-MAD'
            GROUP BY corridor_name, train_year, train_week_num, train_departure_date_short, day_category
        """).bindparams(last_day=last_refreshed_day)
        
        cursor = conn.execute(query)
        data = pd.DataFrame(cursor.all())
        data.to_csv(fpath)
    except Exception as e:
        print("Error while fetching the data: ", e)
    
        if os.path.isfile(fpath):
            data = pd.read_csv(fpath)
            print("data loaded from file.")
            
    #returning
    try:
        conn.rollback()
    except Exception as e:
        pass
        
    try:
        conn.close()
    except Exception as e:
        pass
             
    return data
    
######################################################################################################################################################        
# auxiliary function to get the hijri information from a gregorian date
def get_hijri_info(greg_date):
    hijri_date = Gregorian(greg_date.year, greg_date.month, greg_date.day).to_hijri()
    if hijri_date.month == 9:  # Ramadan
        return f"ramadan_{hijri_date.day}"
    elif hijri_date.month == 12:  # Dhu al-Hijjah
        return f"dul_hijja_{hijri_date.day}"
    else:
        return "None"

######################################################################################################################################################
# open a file and get the json inside
def get_features_from_file(file_name):
    json_path = os.path.join(BASE_DIR, "features", file_name)

    if not os.path.exists(json_path):
        raise RuntimeError(f"Archivo no encontrado: {json_path}")
    
    with open(json_path) as f:
        selected_features = json.load(f)
    return selected_features


######################################################################################################################################################
# function that filter a df with the saved features used by the linear regression model
def fill_missing_features(df, model_columns):
    #fill in any dummy column that was removed due to filter
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    return df
    
def select_features_lr(df):
    cols = get_features_from_file("features_lr.json")
    df = fill_missing_features(df, cols)

    return df[cols]

# function that filter a df with the saved features used by arima
def select_features_arima(df):
    cols = get_features_from_file("exog_arima.json")
    df = fill_missing_features(df, cols)

    return df[cols]

# function that filter a df with the saved features used by the xgradient boosting model
def select_features_lgbm(df):
    cols = get_features_from_file("features_lgbm.json")
    df = fill_missing_features(df, cols)

    return df[cols]
    


######################################################################################################################################################
# create the needed features for the linear regression model
    
def get_features_lr(data):
    # scale to standarize the features
    scaler = StandardScaler()
    
    #select the route
    df_r1 = data[data['corridor_name']=='MAK-MAD']
    df_r1.sort_values('train_departure_date_short', inplace=True, ignore_index=True)
    
    
    
    # reset the year
    df_r1['train_year'] = df_r1['train_year'] - min(df_r1['train_year'])
    
    # create yield column to use it as a general indicator of the prices
    # note: this paramter was removed from the regression since it showed positive paramter, most likely due to multicolinearity
    df_r1['yield'] = df_r1['revenue'] / df_r1['passengers_km']
    
    # standarize
    df_r1['yield_std'] = scaler.fit_transform(df_r1[['yield']])
    df_r1['number_of_services_std'] = scaler.fit_transform(df_r1[["number_of_services"]])
    
    #create dummies for the monthes
    df_date = pd.to_datetime(df_r1['train_departure_date_short'])
    #period_dummies = pd.get_dummies(df_date.dt.month, prefix="Month", drop_first=True) # by months
    period_dummies = pd.get_dummies(df_r1['train_week_num'], prefix="Week", drop_first=True) # by weeks
    
    # create dummies for the days of the week
    df_r1['DayOfWeek'] = df_date.dt.dayofweek
    weekday_dummies = pd.get_dummies(df_date.dt.dayofweek, prefix="Weekday", drop_first=True)
    
    #create dummies for the relevant hijjri dates (ramadan and hajj)    
    df_r1["hijri_day_tag"] = df_r1["train_departure_date_short"].apply(get_hijri_info)
    hijri_dummies = pd.get_dummies(df_r1["hijri_day_tag"], prefix="", prefix_sep="")
    hijri_dummies = hijri_dummies.drop(columns=["None"], errors="ignore")

    #get the price ratio
    df_r1['price_ratio'] = (df_r1['revenue'] / df_r1['revenue_without_promotion']).astype(float)

    #join dummies
    df_r1 = pd.concat([df_r1, period_dummies, weekday_dummies, hijri_dummies], axis=1)
    
    # create polynomial features (discarted due to lack of model improvement)
    
    #poly = PolynomialFeatures(degree=2, include_bias=False)
    #poly_features = poly.fit_transform(df_r1[["number_of_services_std", "train_year"]])
    #poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(["number_of_services_std", "train_year"]))
    # Add all to your dataset
    #df_r1 = pd.concat([df_r1, poly_df], axis=1)

    return df_r1

######################################################################################################################################################
# get features for the arima model
def get_features_arima(df):
    df_r1 = df.copy()
    df_r1["date"] = df['train_departure_date_short']
    df_r1["Y"] = df['passengers']
    
    #prices
    df_r1['ratio_prices'] = df['revenue'] / df['revenue_without_promotion']
    elasticity = -.7661 # the coefficient in the log log model
    df_r1['Y_adjusted'] = df_r1['Y'].astype(float) * (1/df_r1['ratio_prices']).astype(float) ** elasticity

    # add the lag 364 and 354 as exogoneous features
    df_r1["lag_364"] = df_r1["Y_adjusted"].shift(364).astype(float)
    df_r1["lag_354"] = df_r1["Y_adjusted"].shift(354).astype(float)

    # 
    df_r1['num_services'] = df_r1[['number_of_services']]
    df_r1['year'] = df['train_departure_date_short'].dt.year
    df_r1['time'] = ((df['train_departure_date_short'] - min(df['train_departure_date_short'])) / pd.Timedelta(weeks=1)).astype(int)

    #dropna
    return df_r1
    
######################################################################################################################################################
# final function used with the ajustment of dummies to the number of services. to be used by the final model
def get_features_lgbm(data):
    # get the first element of the tuple
    df, month_dummies, week_dummies, kind_date_dummies, weekday_dummies, hijri_dummies = get_features_and_dummies_lgbm(data)

    #adjust dummies to the number of services
    df[month_dummies.columns.tolist()] = df[month_dummies.columns.tolist()].values * df['number_of_services'].values.reshape(-1,1)
    df[weekday_dummies.columns.tolist()] = df[weekday_dummies.columns.tolist()].values * df['number_of_services'].values.reshape(-1,1)
    df[kind_date_dummies.columns.tolist()] = df[kind_date_dummies.columns.tolist()].values * df['number_of_services'].values.reshape(-1,1) 
    df[week_dummies.columns.tolist()] = df[week_dummies.columns.tolist()].values  * df['number_of_services'].values.reshape(-1,1) 
    df[hijri_dummies.columns.tolist()] = df[hijri_dummies.columns.tolist()].values  * df['number_of_services'].values.reshape(-1,1) 
    
    return df


# original function used during the analysis of the model
def get_features_and_dummies_lgbm(data):
    # avoiding standarization for this model
    # scaler = StandardScaler()
    
    #select the route
    df_r1 = data[data['corridor_name']=='MAK-MAD']
    df_r1.sort_values('train_departure_date_short', inplace=True, ignore_index=True)
    
    # reset the year
    df_r1['train_year'] = df_r1['train_year'] - min(df_r1['train_year'])
    
    # instead of the yield column of the first model, the prices will be represented by the price ratio (average prices / standard prices)
    df_r1['price_ratio'] = (df_r1['revenue'] / df_r1['revenue_without_promotion']).astype(float)
    df_r1['price_mix_ratio'] = df_r1['price_mix_ratio'].astype(float)
    
    # standarize
    # df_r1['number_of_services_std'] = scaler.fit_transform(df_r1[["number_of_services"]])
    
    #create dummies for the monthes
    df_date = pd.to_datetime(df_r1['train_departure_date_short'])
    month_dummies = pd.get_dummies(df_date.dt.month, prefix="Month", drop_first=True) # by months
    week_dummies = pd.get_dummies(df_r1['train_week_num'], prefix="Week", drop_first=True) # by weeks
    
    #create dummies for the categorization of days
    kind_date_dummies = pd.get_dummies(df_r1['day_category'], drop_first=False)
    for col in ['Weekend National Day', 'National Day', 'Post Ramadan']:
        if col in kind_date_dummies.columns:
            kind_date_dummies.drop(col, axis=1, inplace=True)
    
    # create dummies for the days of the week
    df_r1['DayOfWeek'] = df_date.dt.dayofweek
    weekday_dummies = pd.get_dummies(df_date.dt.dayofweek, prefix="Weekday", drop_first=True)
    
    
    #create dummies for the relevant hijjri dates (ramadan and hajj) 
    df_r1["hijri_day_tag"] = df_r1["train_departure_date_short"].apply(get_hijri_info)
    hijri_dummies = pd.get_dummies(df_r1["hijri_day_tag"], prefix="", prefix_sep="")
    hijri_dummies = hijri_dummies.drop(columns=["None"], errors="ignore")
    
    # alternatively we can use the lags 364 and 354
    df_r1['lag364'] = df_r1['passengers'].shift(364).astype(float)
    df_r1['lag354'] = df_r1['passengers'].shift(354).astype(float)

    # get smoothing features
    df_r1['rolling6'] = df_r1['passengers'].astype(float).rolling(6).mean()
    df_r1['lag7'] = df_r1['passengers'].shift(7)
    df_r1['lag14'] = df_r1['passengers'].shift(14)

    #join dummies
    df_r1 = pd.concat([df_r1, month_dummies, week_dummies, weekday_dummies, hijri_dummies, kind_date_dummies], axis=1)
    df_r1.index = df_r1['train_departure_date_short']

    return df_r1, month_dummies, week_dummies, kind_date_dummies, weekday_dummies, hijri_dummies

######################################################################################################################################################
# class to wrapp up a model and use it to predict, but calculate the expected passenger based on the current level of prices.
# used when the model was trained with the passengers adjusted to the prices
class AdjustedPredictionWrapper:
    def __init__(self, model, elasticity, ratio_column='price_ratio'):
        self.model = model
        self.ratio_column = ratio_column
        self.elasticity = float(elasticity)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        # Predict adjusted values
        y_pred_adj = self.model.predict(X)

        # Convert to real passengers using ratio
        if isinstance(X, pd.DataFrame):
            ratio = X[self.ratio_column].values
        else:
            raise ValueError("X must be a pandas DataFrame to access ratio column.")

        return y_pred_adj * ratio ** self.elasticity

######################################################################################################################################################
# function to calculate the r2 score without taking in consideration the effect of the prices (remove adjustment in both the target and the prediction)
def r2_score_real(test_values, predicted_values, prices, elasticity):
    real_test_values = test_values * (prices ** elasticity)
    real_predicted_values = predicted_values * (prices ** elasticity)

    return r2_score(real_test_values, real_predicted_values)
    
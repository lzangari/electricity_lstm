import os
import pandas as pd

from energy.utils import utils

TF_ENABLE_ONEDNN_OPTS = 0

data_name = "201501010000_202407010000_15_min"

################################################################################
###################################Import Data##################################
################################################################################

# Load the uploaded data files
production_df = pd.read_csv(rf"data/realised_creation_{data_name}.csv", delimiter=";")
consumption_df = pd.read_csv(
    rf"data/realised_consumption_{data_name}.csv", delimiter=";"
)

# Check the first few rows of the dataframes of production and consumption
print(f"consumption columns: {consumption_df.columns}")
print(f"production columns: {production_df.columns}")


################################################################################
#################################Data Processing################################
################################################################################

#### Convert the columns to a more readable format
# Translate column names for production data
production_df.columns = [
    "start_time",
    "end_time",
    "biomass_mwh",
    "hydropower_mwh",
    "wind_offshore_mwh",
    "wind_onshore_mwh",
    "solar_mwh",
    "other_renewables_mwh",
    "nuclear_mwh",
    "lignite_mwh",
    "hard_coal_mwh",
    "natural_gas_mwh",
    "pumped_storage_mwh",
    "other_conventional_mwh",
]

# Translate column names for consumption data
consumption_df.columns = [
    "start_time",
    "end_time",
    "total_load_mwh",
    "residual_load_mwh",
    "saved_pumped_storage_mwh",
]


# Convert numeric columns in production data
for col in production_df.columns[2:]:
    production_df[col] = production_df[col].apply(utils.convert_to_float)

# Convert numeric columns in consumption data
for col in consumption_df.columns[2:]:
    consumption_df[col] = consumption_df[col].apply(utils.convert_to_float)


#################################Time Processing################################

#### Convert numeric columns in production data
for col in production_df.columns[2:]:
    production_df[col] = production_df[col].apply(utils.convert_to_float)

# Convert numeric columns in consumption data
for col in consumption_df.columns[2:]:
    consumption_df[col] = consumption_df[col].apply(utils.convert_to_float)


#### Convert date and time columns to datetime format
# production data
production_df["start_time"] = pd.to_datetime(
    production_df["start_time"], format="%d.%m.%Y %H:%M"
)
production_df["end_time"] = pd.to_datetime(
    production_df["end_time"], format="%d.%m.%Y %H:%M"
)

# consumption data
consumption_df["start_time"] = pd.to_datetime(
    consumption_df["start_time"], format="%d.%m.%Y %H:%M"
)
consumption_df["end_time"] = pd.to_datetime(
    consumption_df["end_time"], format="%d.%m.%Y %H:%M"
)


###################################Data Merging###################################

# merge the two dataframes
data = pd.merge(production_df, consumption_df, on=["start_time", "end_time"])

# pring the data head
print(f"Data head: {data.head()}")


###################################Data Cleaning###################################
# check if there is any NaN values in the data
print(f"Check NaN values: {data.isnull().sum()}")

# replace the NaN values with 0
data.fillna(0, inplace=True)


# check it again
print(f"Check NaN values after replacing: {data.isnull().sum()}")

# calculate total production energy
data["total_production_mwh"] = data[
    [
        "biomass_mwh",
        "hydropower_mwh",
        "wind_offshore_mwh",
        "wind_onshore_mwh",
        "solar_mwh",
        "other_renewables_mwh",
        "nuclear_mwh",
        "lignite_mwh",
        "hard_coal_mwh",
        "natural_gas_mwh",
        "pumped_storage_mwh",
        "other_conventional_mwh",
    ]
].sum(axis=1)


###################################Data Transforming###################################
# extract granular time information from the date_time in the data
data["hour"] = data["start_time"].dt.hour
data["minute"] = data["start_time"].dt.minute
data["day_of_week"] = data["start_time"].dt.dayofweek
data["day_of_year"] = data["start_time"].dt.dayofyear
data["week_of_year"] = data["start_time"].dt.isocalendar().week

# encode the date and time features in a way that captures the periodicity of the data
# capture the cyclical nature of hours in a day.
utils.encode_feature(data, "hour", 24)
# capture the cyclical nature of minutes within each hour.
utils.encode_feature(data, "minute", 60)
# capture the cyclical nature of days within a week.
utils.encode_feature(data, "day_of_week", 7)
# capture the cyclical nature of days within a year.
utils.encode_feature(data, "day_of_year", 365)
# capture the cyclical nature of weeks within a year.
utils.encode_feature(data, "week_of_year", 52)

# print the data after transformation
print(f"Data after transformation: {data.head()}")

# create a "transformed" folder if not existed and save the data there
# check if the folder exists in the data directory
if not os.path.exists("data/transformed"):
    os.makedirs("data/transformed")
    print("Directory 'transformed' created")

# save the data to the transformed folder
data.to_csv(rf"data/transformed/transformed_{data_name}.csv", index=False)

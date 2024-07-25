import pandas as pd

from energy.utils import utils

TF_ENABLE_ONEDNN_OPTS = 0

data_name = "201501010000_202407010000_15_min"
production_column_translation = [
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

consumption_column_translation = [
    "start_time",
    "end_time",
    "total_load_mwh",
    "residual_load_mwh",
    "saved_pumped_storage_mwh",
]


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
production_df.columns = production_column_translation

# Translate column names for consumption data
consumption_df.columns = consumption_column_translation

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

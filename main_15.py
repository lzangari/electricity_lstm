import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from energy.utils import utils
from energy.modelling import optimization, model_creation

TF_ENABLE_ONEDNN_OPTS = 0

DATA_NAME = "hour"  # quarterhour
MODEL_NAME = "lstm_stacked"  # lstm_naive, lstm_stacked, lstm_seq2seq
OPTIMIZE_LENGTH_SEQUENCE = True
if OPTIMIZE_LENGTH_SEQUENCE:
    selected_seq_lengths = [24, 24 * 2, 24 * 7]  # 1 day, 2 days, 1 week

SEQ_LENGTH = 24 * 1  # 24 hours of one hour intervals
PRED_LENGTH = 24 * 1  # Predict one day ahead

if (MODEL_NAME == "lstm_naive") or (MODEL_NAME == "lstm_stacked"):
    build_lstm = model_creation.build_lstm_based_model
################################################################################
###################################Import Data##################################
################################################################################

# Load the uploaded data files
production_df = pd.read_csv(
    rf"electricity_data/creation_realized_{DATA_NAME}.csv", delimiter=";"
)
consumption_df = pd.read_csv(
    rf"electricity_data/consumption_realized_{DATA_NAME}.csv", delimiter=";"
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
data.to_csv(rf"data/transformed/transformed_{DATA_NAME}.csv", index=False)


################################################################################
##############################FEATURE SELECTION#################################
################################################################################
# Select relevant features for training
features = [
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
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "week_of_year_sin",
    "week_of_year_cos",
]

# Output features
output_features = ["total_production_mwh", "total_load_mwh"]
output_size = len(output_features)

# Combine into a complete list for dataframe alignment
all_columns = output_features + features

# describe the data with the selected features
print(f"Data description with selected features: {data[all_columns].describe()}")

# check the type of the features
print(f"Data types of the selected features: {data[all_columns].dtypes}")


##################################Data Preparation################################
# Split the data into train, validation, and test sets
train_data = data[data["start_time"] < "2022-01-01"]  # '2022-01-01'
validation_data = data[
    (data["start_time"] >= "2022-01-01") & (data["start_time"] < "2023-01-01")
]
test_data = data[data["start_time"] >= "2023-01-01"]


#################################Data Transformation###############################
# Scale the data
scaler = MinMaxScaler()
# fit the scaler on the training data
scaled_train_data = scaler.fit_transform(train_data[features])
# transform the validation and test data
scaled_validation_data = scaler.transform(validation_data[features])
scaled_test_data = scaler.transform(test_data[features])

###################################Data Sequencing#################################
X_train, y_train = utils.create_sequences(
    scaled_train_data, SEQ_LENGTH, PRED_LENGTH, output_size
)
X_val, y_val = utils.create_sequences(
    scaled_validation_data, SEQ_LENGTH, PRED_LENGTH, output_size
)
X_test, y_test = utils.create_sequences(
    scaled_test_data, SEQ_LENGTH, PRED_LENGTH, output_size
)

print("X_train shape: ", X_train.shape)

###############################Finding Sequence Length#############################
if OPTIMIZE_LENGTH_SEQUENCE:
    best_sequence_length, losses, calculated_seq_lengths = (
        optimization.optimize_seq_length(
            data=scaled_train_data,
            seq_lengths=selected_seq_lengths,
            pred_length=PRED_LENGTH,
            output_size=output_size,
            built_model=build_lstm,
            path="model_info",
            model_type=MODEL_NAME,
            epochs=50,
            batch_size=64,
            num_folds=3,
            verbose=2,
            name=f"{MODEL_NAME}_{DATA_NAME}",
        )
    )
    SEQ_LENGTH = best_sequence_length
    print(f"Best sequence length: {best_sequence_length}")

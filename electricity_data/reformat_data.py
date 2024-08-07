import pandas as pd

# Load the CSV file
df = pd.read_csv(r"electricity_data\creation_forecast_day_ahead_hour.csv", sep=";")

# Rename the columns
# df.columns = [
#     'Datum von',
#     'Datum bis',
#     'Gesamt (Netzlast) [MWh] Berechnete Auflösungen',
#     'Residuallast [MWh] Berechnete Auflösungen'
# ]
df.columns = [
    "Datum von",
    "Datum bis",
    "Gesamt [MWh] Berechnete Auflösungen",
    "Photovoltaik und Wind [MWh] Berechnete Auflösungen",
    "Wind Offshore [MWh] Berechnete Auflösungen",
    "Wind Onshore [MWh] Berechnete Auflösungen",
    "Photovoltaik [MWh] Berechnete Auflösungen",
    "Sonstige [MWh] Berechnete Auflösungen",
]


# Function to reformat the date with hour
def reformat_date(date_str):
    return pd.to_datetime(date_str).strftime("%d.%m.%Y %H:%M")


# Apply the date reformatting
df["Datum von"] = df["Datum von"].apply(reformat_date)
df["Datum bis"] = df["Datum bis"].apply(reformat_date)


# Function to reformat the numbers
def reformat_number(num_str):
    if num_str == "-":
        return 0.0
    return (
        f"{float(num_str.replace(',', '')):,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


# Apply the number reformatting
# df['Gesamt (Netzlast) [MWh] Berechnete Auflösungen'] = df['Gesamt (Netzlast) [MWh] Berechnete Auflösungen'].apply(reformat_number)
# df['Residuallast [MWh] Berechnete Auflösungen'] = df['Residuallast [MWh] Berechnete Auflösungen'].apply(reformat_number)
for column in df.columns[2:]:
    df[column] = df[column].apply(reformat_number)


# Save the transformed CSV
df.to_csv(
    r"electricity_data\creation_forecast_day_ahead_hour_transformed.csv",
    sep=";",
    index=False,
)

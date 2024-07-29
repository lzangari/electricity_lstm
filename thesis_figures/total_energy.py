import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
generation_file_path = "electricity_data/creation_realized_month.csv"
consumption_file_path = "electricity_data/consumption_realized_month.csv"
generation_data = pd.read_csv(generation_file_path, delimiter=";")
consumption_data = pd.read_csv(consumption_file_path, delimiter=";")


# Define a function to convert German-style numbers to standard float
def convert_german_number(number_str):
    if number_str == "-":
        return 0.0
    return float(number_str.replace(".", "").replace(",", "."))


# Apply the conversion function to the generation dataset
for column in generation_data.columns[2:]:
    generation_data[column] = generation_data[column].apply(convert_german_number)

# Rename columns for clarity in generation data
generation_data.columns = [
    "Start Date",
    "End Date",
    "Biomass",
    "Hydropower",
    "Wind Offshore",
    "Wind Onshore",
    "Photovoltaics",
    "Other Renewables",
    "Nuclear",
    "Lignite",
    "Hard Coal",
    "Fossil Gas",
    "Hydro Pumped Storage",
    "Other Conventional",
]

# Convert date columns to datetime format in generation data
generation_data["Start Date"] = pd.to_datetime(
    generation_data["Start Date"], format="%d.%m.%Y"
)
generation_data["End Date"] = pd.to_datetime(
    generation_data["End Date"], format="%d.%m.%Y"
)

# Convert the German-style numbers to standard float for the consumption data
consumption_data["Gesamt (Netzlast) [MWh]"] = consumption_data[
    "Gesamt (Netzlast) [MWh] Berechnete Auflösungen"
].apply(convert_german_number)

# Convert date columns to datetime format in consumption data
consumption_data["Start Date"] = pd.to_datetime(
    consumption_data["Datum von"], format="%d.%m.%Y"
)
consumption_data["End Date"] = pd.to_datetime(
    consumption_data["Datum bis"], format="%d.%m.%Y"
)

# Calculate cumulative sums for generation data
cumulative_data_corrected = pd.DataFrame(
    {
        "Date": generation_data["Start Date"],
        "Biomass": generation_data["Biomass"],
        "Hydropower": generation_data["Biomass"] + generation_data["Hydropower"],
        "Wind Offshore": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"],
        "Wind Onshore": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"],
        "Photovoltaics": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"]
        + generation_data["Photovoltaics"],
        "Other Renewables": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"]
        + generation_data["Photovoltaics"]
        + generation_data["Other Renewables"],
        "Nuclear": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"]
        + generation_data["Photovoltaics"]
        + generation_data["Other Renewables"]
        + generation_data["Nuclear"],
        "Lignite": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"]
        + generation_data["Photovoltaics"]
        + generation_data["Other Renewables"]
        + generation_data["Nuclear"]
        + generation_data["Lignite"],
        "Hard Coal": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"]
        + generation_data["Photovoltaics"]
        + generation_data["Other Renewables"]
        + generation_data["Nuclear"]
        + generation_data["Lignite"]
        + generation_data["Hard Coal"],
        "Fossil Gas": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"]
        + generation_data["Photovoltaics"]
        + generation_data["Other Renewables"]
        + generation_data["Nuclear"]
        + generation_data["Lignite"]
        + generation_data["Hard Coal"]
        + generation_data["Fossil Gas"],
        "Hydro Pumped Storage": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"]
        + generation_data["Photovoltaics"]
        + generation_data["Other Renewables"]
        + generation_data["Nuclear"]
        + generation_data["Lignite"]
        + generation_data["Hard Coal"]
        + generation_data["Fossil Gas"]
        + generation_data["Hydro Pumped Storage"],
        "Other Conventional": generation_data["Biomass"]
        + generation_data["Hydropower"]
        + generation_data["Wind Offshore"]
        + generation_data["Wind Onshore"]
        + generation_data["Photovoltaics"]
        + generation_data["Other Renewables"]
        + generation_data["Nuclear"]
        + generation_data["Lignite"]
        + generation_data["Hard Coal"]
        + generation_data["Fossil Gas"]
        + generation_data["Hydro Pumped Storage"]
        + generation_data["Other Conventional"],
    }
)

# Merge consumption data with generation data based on the dates
merged_data = pd.merge(
    cumulative_data_corrected,
    consumption_data[["Start Date", "Gesamt (Netzlast) [MWh]"]],
    left_on="Date",
    right_on="Start Date",
)

# Define color schemes for renewable and non-renewable sources
renewable_colors = ["#2ca02c", "#98df8a", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
non_renewable_colors = [
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
]
colors = renewable_colors + non_renewable_colors

# Plot the cumulative stacked area plot with specified colors and alpha
fig, ax = plt.subplots(figsize=(14, 8))

# Plot cumulative data with specified colors and alpha
ax.stackplot(
    merged_data["Date"],
    merged_data["Biomass"],
    merged_data["Hydropower"] - merged_data["Biomass"],
    merged_data["Wind Offshore"] - merged_data["Hydropower"],
    merged_data["Wind Onshore"] - merged_data["Wind Offshore"],
    merged_data["Photovoltaics"] - merged_data["Wind Onshore"],
    merged_data["Other Renewables"] - merged_data["Photovoltaics"],
    merged_data["Nuclear"] - merged_data["Other Renewables"],
    merged_data["Lignite"] - merged_data["Nuclear"],
    merged_data["Hard Coal"] - merged_data["Lignite"],
    merged_data["Fossil Gas"] - merged_data["Hard Coal"],
    merged_data["Hydro Pumped Storage"] - merged_data["Fossil Gas"],
    merged_data["Other Conventional"] - merged_data["Hydro Pumped Storage"],
    labels=[
        "Biomass",
        "Hydropower",
        "Wind Offshore",
        "Wind Onshore",
        "Photovoltaics",
        "Other Renewables",
        "Nuclear",
        "Lignite",
        "Hard Coal",
        "Fossil Gas",
        "Hydro Pumped Storage",
        "Other Conventional",
    ],
    colors=colors,
    alpha=0.7,
)

# Plot total consumption
ax.plot(
    merged_data["Date"],
    merged_data["Gesamt (Netzlast) [MWh]"],
    color="darkred",
    linewidth=1,
    linestyle="dashed",
)


# Legend for renewable sources
renewable_legend = ax.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=renewable_colors[i]) for i in range(6)],
    [
        "Biomass",
        "Hydropower",
        "Wind Offshore",
        "Wind Onshore",
        "Photovoltaics",
        "Other Renewables",
    ],
    loc="upper right",
    title="Renewable Sources",
)
ax.add_artist(renewable_legend)

# Legend for non-renewable sources
non_renewable_legend = ax.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=non_renewable_colors[i]) for i in range(6)],
    [
        "Nuclear",
        "Lignite",
        "Hard Coal",
        "Fossil Gas",
        "Hydro Pumped Storage",
        "Other Conventional",
    ],
    loc="upper left",
    title="Non-Renewable Sources",
)
ax.add_artist(non_renewable_legend)

# Legend for total consumption
total_consumption_legend = ax.legend(
    [plt.Line2D([0], [0], color="darkred", linewidth=2, linestyle="dashed")],
    ["Total Consumption"],
    loc="upper center",
)

# Customize the plot
# ax.set_title("Cumulative Electricity Generation and Consumption (MWh)")
ax.set_xlabel("Date")
ax.set_ylabel("Electricity Generation (GWh)")

# X lim for 2015
ax.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2024-06-30"))
ax.set_ylim(0, 65000000)

# Y axsis in GWh
ax.set_yticklabels([f"{int(x/1000)}" for x in ax.get_yticks()])
# Remove last two yticsk and ytiklabel
ax.set_yticks(ax.get_yticks()[:-2])

ax.grid(False)

# Get rid of top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("thesis_figures/total_montly_energy.png")
plt.savefig("thesis_figures/total_montly_energy.pdf")

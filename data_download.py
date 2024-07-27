# Python script to download some data from the internet from smard.de
# and save it in a local directory

import os
import requests

ids = [
    {
        "name": "creation",
        "category": "realized",
        "moduleIds": [
            1001224,
            1004066,
            1004067,
            1004068,
            1001223,
            1004069,
            1004071,
            1004070,
            1001226,
            1001228,
            1001227,
            1001225,
        ],
    },
    {
        "name": "creation",
        "category": "forecast_day_ahead",
        "moduleIds": [2000122, 2005097, 2000715, 2003791, 2000123, 2000125],
    },
    {
        "name": "creation",
        "category": "forecast_intraday",
        "moduleIds": [32005129, 32005128, 32005127, 32005126],
    },
    {
        "name": "consumption",
        "category": "realized",
        "moduleIds": [5000410, 5004387, 5004359],
    },
    {"name": "consumption", "category": "forecast", "moduleIds": [6000411, 6004362]},
]


additional_data = {
    "format": "CSV",
    "region": "DE",
    "type": "discrete",
    "language": "de",
    "timestamp_from": 1420066800000,
    "timestamp_to": 1719784799999,
}

resolutions = ["day", "hour", "quarterhour", "week", "month", "year"]


link = "https://www.smard.de/home/downloadcenter/download-marktdaten/?downloadAttributes=%7B%22selectedCategory%22:1,%22selectedSubCategory%22:1,%22selectedRegion%22:%22DE%22,%22selectedFileType%22:%22CSV%22,%22from%22:1420066800000,%22to%22:1719784799999%7D"

# Create a directory to save the data
if not os.path.exists("electricity_data"):
    os.makedirs("electricity_data")

for id in ids:
    for resolution in resolutions:
        request_form = {
            "request_form": [
                {
                    **additional_data,
                    "moduleIds": id["moduleIds"],
                    "resolution": resolution,
                }
            ]
        }

        link = "https://www.smard.de/nip-download-manager/nip/download/market-data"
        request_method = "POST"

        r = requests.post(link, json=request_form)

        if r.status_code == 200:
            with open(
                f"electricity_data/{id['name']}_{id['category']}_{resolution}.csv", "wb"
            ) as f:
                f.write(r.content)
        else:
            print(f"Error: {r.status_code}")

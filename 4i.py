import pandas as pd
import requests
from tqdm import tqdm
from collections import defaultdict
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Load the dataset
file_path = "haunted_places.tsv"
df = pd.read_csv(file_path, sep='\t')

# Limit the number of rows to parse (adjustable)
num_rows_to_parse = 10993  # Change this value to process more or fewer rows
df = df.head(num_rows_to_parse)

# Cache for reducing duplicate requests
cache = defaultdict(dict)


# Function to extract daylight duration from the USNO website
def get_daylight_duration(lat, lon, date):
    year, month, day = date.split('-')
    lat = round(float(lat), 2) if pd.notna(lat) else None
    lon = round(float(lon), 2) if pd.notna(lon) else None

    if lat is None or lon is None:
        return "unknown"

    cache_key = (lat, lon, year)
    if cache_key in cache:
        yearly_data = cache[cache_key]
    else:
        url = f"https://aa.usno.navy.mil/calculated/durdaydark?year={year}&task=0&lat={lat}&lon={lon}&label=&tz=0&tz_sign=1&submit=Get+Data"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find_all('table')[1]  # The second table contains the data
                if table:
                    rows = table.find_all('tr')[2:]  # Skip headers
                    yearly_data = {}
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) == 13:  # Ensure it's a valid row
                            day_of_month = int(cells[0].text.strip())
                            durations = [cell.text.strip() for cell in cells[1:]]
                            yearly_data[day_of_month] = durations
                    cache[cache_key] = yearly_data
                else:
                    return "unknown"
            else:
                return "unknown"
        except:
            return "unknown"

    # Retrieve the duration for the specific month and day
    day_of_month = int(day)
    month_index = int(month) - 1  # Convert month to zero-based index
    if day_of_month in yearly_data:
        return yearly_data[day_of_month][month_index]
    else:
        return "unknown"


# Function to process a single row
def process_row(row):
    lat = row['latitude'] if not pd.isna(row['latitude']) else row['city_latitude']
    lon = row['longitude'] if not pd.isna(row['longitude']) else row['city_longitude']
    date = row['Haunted Places Date']

    if pd.isna(lat) or pd.isna(lon) or pd.isna(date):
        return "unknown"
    else:
        return get_daylight_duration(lat, lon, date)


# Process each row in the dataset with multithreading
with ThreadPoolExecutor(max_workers=10) as executor:
    daylight_durations = list(
        tqdm(executor.map(process_row, [row for _, row in df.iterrows()]), total=df.shape[0], desc="Processing rows"))

# Append the new column to the dataframe
df["Amount of daylight"] = daylight_durations

# Save the updated dataset
df.to_csv("updated_haunted_places.tsv", sep='\t', index=False)

import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from urllib.parse import unquote
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

def search(query, driver):
   """
   Navigates to DuckDuckGo image search for the given query,
   then yields image URLs extracted from the 'data-src' attribute of elements with
   class 'tile--img__img'.
   """
   driver.get(f'https://duckduckgo.com/?q={query}&t=h_&iax=images&ia=images')
   time.sleep(0.8)  # Allow time for dynamic content to load

   img_tags = driver.find_elements(By.CLASS_NAME, 'tile--img__img')

   for tag in img_tags:
       src = tag.get_attribute('data-src')
       if src:
           src = unquote(src)
           parts = src.split('=', maxsplit=1)
           if len(parts) > 1:
               yield parts[1]

def sanitize_filename(filename):
   """Remove characters that are not allowed in filenames."""
   return re.sub(r'[\\/*?:"<>|]', "_", filename)

# Set up headless Firefox via Selenium
options = Options()
options.add_argument("--headless")
driver = webdriver.Firefox(options=options)

# Create the "image" folder if it doesn't exist
os.makedirs("image", exist_ok=True)

# Load your TSV file (using first 1000 rows for testing)
tsv_file = "1_merged_final_with_urban.tsv"
df = pd.read_csv(tsv_file, sep="\t").head(1000)

# Track already downloaded locations
downloaded_locations = set()

# Process each row with a progress bar and download images based on unique location
for index, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Images"):
   location = row["location"]

   # Skip if this location has already been downloaded
   if location in downloaded_locations:
       continue

   query = f"{location} {row['city']}"

   # Retrieve image URLs using our search function
   image_generator = search(query, driver)
   try:
       # Get the first available image URL
       image_url = next(image_generator)
   except StopIteration:
       print(f"No image found for query '{query}'")
       continue

   # Download and save the image
   try:
       response = requests.get(image_url, stream=True)
       if response.status_code == 200:
           ext = os.path.splitext(image_url)[1]
           if not ext or len(ext) > 5:
               ext = ".jpg"
           filename = sanitize_filename(f"{location}{ext}")
           file_path_img = os.path.join("image", filename)
           with open(file_path_img, "wb") as f:
               for chunk in response.iter_content(chunk_size=1024):
                   f.write(chunk)

           # Mark this location as downloaded
           downloaded_locations.add(location)
       else:
           print(f"Failed to download image for '{query}': HTTP {response.status_code}")
   except Exception as e:
       print(f"Error downloading image for '{query}': {e}")

# Close the Selenium driver
driver.quit()












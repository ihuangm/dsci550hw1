import os
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed  # Multithreading

# Load YOLOv5 Model
model = YOLO("yolov5s.pt")

# Define Image Folder
IMAGE_FOLDER = "image"

# Load the TSV file
tsv_file = "1_merged_final_with_urban.tsv"
df = pd.read_csv(tsv_file, sep="\t").head(1500)

# Dictionary to store extracted features
image_data = {}


def get_dominant_color(image_path, k=3):
    """Extract the dominant color from an image using K-Means clustering."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Unknown"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(image)
        dominant_color = kmeans.cluster_centers_[0]  # Get the most dominant color
        return f"RGB({int(dominant_color[0])}, {int(dominant_color[1])}, {int(dominant_color[2])})"
    except:
        return "Unknown"


def detect_elements(image_path):
    """Use YOLOv5 to detect objects in an image and return a comma-separated string of elements."""
    try:
        results = model(image_path)  # Run YOLO on the image

        # Ensure results exist
        if not results or len(results) == 0:
            return "Unknown"

        # Extract detected object names
        detected_objects = [results[0].names[int(box.cls)] for box in results[0].boxes]  # Corrected extraction

        # If objects exist, return them as a comma-separated string
        return ", ".join(detected_objects) if detected_objects else "Unknown"

    except Exception as e:
        print(f"Error detecting objects in {image_path}: {e}")
        return "Unknown"


def get_image_dimensions(image_path):
    """Extract image dimensions (Width x Height)."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Unknown"
        height, width = image.shape[:2]
        return f"{width}x{height}"
    except:
        return "Unknown"


def process_image(image_file, progress_bar):
    """Process a single image to extract features (runs in a thread)."""
    location_name = os.path.splitext(image_file)[0]  # Extract filename without extension
    image_path = os.path.join(IMAGE_FOLDER, image_file)

    # Extract Features
    color = get_dominant_color(image_path)
    elements = detect_elements(image_path)
    dimensions = get_image_dimensions(image_path)

    # Store results in dictionary
    image_data[location_name] = {"color": color, "element": elements, "Dimensions": dimensions}

    # Update progress bar
    progress_bar.update(1)


# Get all image files in the folder and select only the first 10
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Process images using multithreading
with tqdm(total=len(image_files), desc="Processing Images") as progress_bar:
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust workers for your CPU
        futures = {executor.submit(process_image, img, progress_bar): img for img in image_files}
        for future in as_completed(futures):
            pass  # Ensure all threads complete

# Match extracted data with the TSV file
color_list, element_list, dimension_list = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching Data to TSV"):
    location = row["location"]
    if location in image_data:
        color_list.append(image_data[location]["color"])
        element_list.append(image_data[location]["element"])
        dimension_list.append(image_data[location]["Dimensions"])
    else:
        # If no image found for this location, mark as "Unknown"
        color_list.append("Unknown")
        element_list.append("Unknown")
        dimension_list.append("Unknown")

# Append new columns to the DataFrame
df["color"] = color_list
df["element"] = element_list
df["Dimensions"] = dimension_list

# Save updated TSV file
updated_tsv_file = "test_updated_haunted_places.tsv"
df.to_csv(updated_tsv_file, sep="\t", index=False)

print(f"✅ Updated dataset saved as {updated_tsv_file}")











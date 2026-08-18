import csv
import json
import os

# Ask user for CSV file path
csv_file = input("Enter CSV file path: ").strip()

# Check if file exists
if not os.path.exists(csv_file):
    print("Error: File not found!")
    exit()

# Create JSON file name in the same folder
json_file = os.path.splitext(csv_file)[0] + ".json"

data = []

# Read CSV
with open(csv_file, mode="r", encoding="utf-8") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data.append(row)

# Write JSON
with open(json_file, mode="w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)

print(f"JSON file created successfully: {json_file}")
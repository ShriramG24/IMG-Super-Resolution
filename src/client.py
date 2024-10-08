import argparse
import os
import shutil

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

OUTPUT_DIR = "output"
SR_MODEL_URL = "http://localhost:5000/super-resolution"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Resolution Client")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Path to the input directory containing images.",
    )
    args = parser.parse_args()

    INPUT_DIR = args.input_dir

    if not os.path.exists(INPUT_DIR):
        print(f"Error: The input directory '{INPUT_DIR}' does not exist.")
        exit(1)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    client = MLClient(SR_MODEL_URL)
    data_type = DataTypes.IMAGE
    inputs = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            name = filename.split(".")[0].split("/")[-1]
            inputs.append({"file_path": f"{INPUT_DIR}/{filename}"})

    if len(inputs) == 0:
        print("No images found in the input directory.")
    else:
        response = client.request(inputs, data_type, {"output_dir": OUTPUT_DIR})
        print(response)

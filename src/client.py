import argparse
import os

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

SR_MODEL_URL = "http://localhost:5000/run-inference"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Resolution Client")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Path to the input directory containing images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to the output directory to save the super-resolved images.",
    )
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(INPUT_DIR):
        print(f"Error: The input directory '{INPUT_DIR}' does not exist.")
        exit(1)
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: The output directory '{OUTPUT_DIR}' does not exist.")
        exit(1)

    client = MLClient(SR_MODEL_URL)
    data_type = DataTypes.IMAGE

    inputs = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            inputs.append({"file_path": f"{INPUT_DIR}/{filename}"})

    response = client.request(inputs, data_type, {"input_dir": INPUT_DIR, "output_dir": OUTPUT_DIR})
    print(response)

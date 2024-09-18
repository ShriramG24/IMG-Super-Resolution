import os, shutil
from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

INPUT_DIR = "input"
OUTPUT_DIR = "output"
SR_MODEL_URL = "http://localhost:5000/super-resolution"

if __name__ == '__main__':
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    client = MLClient(SR_MODEL_URL)
    data_type = DataTypes.IMAGE
    inputs = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            name = filename.split('.')[0].split('/')[-1]
            inputs.append({ 'file_path': f'{INPUT_DIR}/{filename}' })
    
    response = client.request(inputs, data_type, { 'output_dir': OUTPUT_DIR })
    print(response)

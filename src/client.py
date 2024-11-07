from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import Input, BatchFileInput

ISR_MODEL_URL = "http://localhost:5000/super-resolution"

client = MLClient(ISR_MODEL_URL)

inputs = {
    "input_images": {
        "files": [
            { "path": "./input/baboon.png" },
            { "path": "./input/meerkat.png" },
            { "path": "./input/street.png" },
        ]
    },
    "output_directory": { "path": "./output" },
}

parameters = {
    "weights": "gans",
    "scale": 4.0,
}

response = client.request(inputs, parameters)
print(f"ISR Response:\n{response}")
import os

from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ImageResult, ResponseModel
from flask import request

from super_resolution import SuperResolution

model = SuperResolution()
server = MLServer(__name__)

# @server.route("/run-inference", DataTypes.Image)
# def run_inference():
#     body = request.get_json()
#     inputs = []
#     for filename in os.listdir(body['inputDir']):
#         if filename.endswith((".png", ".jpg", ".jpeg")):
#             inputs.append({"file_path": f"{body['inputDir']}/{filename}"})

#     results = [
#         ImageResult(file_path=res["file_path"], result=res["result"])
#         for res in model.predict(inputs, body['outputDir'])
#     ]

#     return ResponseModel(results=results).get_response()

@server.app.post("/run-inference")
def run_inference():
    body = request.get_json()
    print(body)
    inputs = []
    for filename in os.listdir(body['inputDir']):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            inputs.append({"file_path": f"{body['inputDir']}/{filename}"})

    results = [
        ImageResult(file_path=res["file_path"], result=res["result"])
        for res in model.predict(inputs, {
            "outputDir": body['outputDir'],
            "weights": body['parameters'][0]['weights'],
            "scale": float(body['parameters'][0]['scale'])
        })
    ]

    return ResponseModel(results=results).get_response()

@server.app.get("/health")
def health():
    return {"status": 200}

if __name__ == "__main__":
    server.run()

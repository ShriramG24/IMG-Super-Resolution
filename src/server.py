from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ImageResult, ResponseModel

from super_resolution import SuperResolution

model = SuperResolution()
server = MLServer(__name__)


@server.route("/super-resolution", DataTypes.IMAGE)
def super_resolution(inputs: list, parameters: dict):
    results = [
        ImageResult(file_path=res["file_path"], result=res["result"])
        for res in model.predict(inputs, parameters["output_dir"])
    ]

    return ResponseModel(results=results).get_response()


if __name__ == "__main__":
    server.run()

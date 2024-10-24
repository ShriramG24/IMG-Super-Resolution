from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    FileResponse,
    ResponseBody,
    BatchFileInput,
    FileType,
    DirectoryInput,
    BatchFileResponse,
    TaskSchema,
    InputSchema,
    InputType,
)

from .super_resolution import SuperResolution

model = SuperResolution()
server = MLServer(__name__)


def create_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                input_type=InputType.BATCHFILE,
                key="input_images",
                subtitle="Images to be upscaled",
                label="Input Images",
            ),
            InputSchema(
                input_type=InputType.DIRECTORY,
                key="output_directory",
                label="Output Directory",
                subtitle="Directory to save the upscaled images",
            ),
        ],
        parameters=[],
    )


class Inputs(TypedDict):
    input_images: BatchFileInput
    output_directory: DirectoryInput


class Parameters(TypedDict):
    pass


@server.route("/super-resolution", task_schema_func=create_task_schema, short_title="Super Resolution")
def super_resolution(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    results = [
        FileResponse(title=res["file_path"], path=res["result"], file_type=FileType.IMG)
        for res in model.predict(inputs["input_images"].files, inputs["output_directory"].path)
    ]

    return ResponseBody(root=BatchFileResponse(files=results))


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=5001)

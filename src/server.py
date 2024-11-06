from typing import TypedDict
from flask_ml.flask_ml_server import MLServer, load_file_as_string
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
    ParameterSchema,
    EnumParameterDescriptor,
    EnumVal,
    RangedFloatParameterDescriptor,
    FloatRangeDescriptor,
)

from .super_resolution import SuperResolution

model = SuperResolution()
server = MLServer(__name__)

server.add_app_metadata(
    name="Image Super Resolution",
    author="Mr Bob",
    version="1.0.0",
    info=load_file_as_string("src/info/server_info.md"),
)

def create_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                input_type=InputType.BATCHFILE,
                key="input_images",
                subtitle="Images to be Upscaled",
                label="Input Images",
            ),
            InputSchema(
                input_type=InputType.DIRECTORY,
                key="output_directory",
                label="Output Directory",
                subtitle="Directory to save the upscaled images",
            ),
        ],
        parameters=[
            ParameterSchema(
                key="weights",
                label="Model Weights",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key="gans", label="GANS"),
                        EnumVal(key="psnr-large", label="PSNR Large"),
                        EnumVal(key="psnr-small", label="PSNR Small"),
                        EnumVal(key="noise-cancel", label="Noise Cancel"),
                    ],
                    default="gans",
                )            
            ),
            ParameterSchema(
                key="scale",
                label="Scale Factor",
                value=RangedFloatParameterDescriptor(
                    range=FloatRangeDescriptor(min=1.0, max=4.0),
                    default=4.0,
                ),
            ),
        ],
    )


class Inputs(TypedDict):
    input_images: BatchFileInput
    output_directory: DirectoryInput


class Parameters(TypedDict):
    weights: str
    scale: float


@server.route("/super-resolution", task_schema_func=create_task_schema, short_title="Super Resolution")
def super_resolution(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    results = [
        FileResponse(title=res["file_path"], path=res["result"], file_type=FileType.IMG)
        for res in model.predict(
            inputs["input_images"].files,
            inputs["output_directory"].path,
            weights=parameters["weights"],
            scale=parameters["scale"]
        )
    ]

    return ResponseBody(root=BatchFileResponse(files=results))


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=5000)

# IMG Super Resolution -- Modified to Work with [RescueBox Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop)

This simple project aims to utilize an existing state-of-the-art pre-trained model to upscale low-resolution images to a higher resolution, which is known in the computer vision community as **image super resolution**. The model used here is a Residual-in-Residual Dense Network (RRDN), which is based on an ESRGAN architecture. More about the specific implementation in this project can be found in the following [repository](https://github.com/idealo/image-super-resolution). 

## Installation Instructions

Make sure you have Python3 installed. Before installing dependencies, you may want to setup a virtual environment if you haven't already. Steps on how to do so using `venv` can be found [here](https://docs.python.org/3/library/venv.html). Follow these steps to setup and run the service:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/shriramg24/img-super-resolution.git
    cd img-super-resolution
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install --no-deps -r no-deps.txt
    ```

3. **Prepare Input Images:**

    Add any low-resolution images that need to be upscaled to the top-level of the `input` directory. You can also specify a different input directory by passing in a path as a command-line argument to the client (shown further below). All image files are expected to be in one of these formats: `.png`, `.jpg`, or `.jpeg`.

4. **Run the Application:**

    Make sure you are in the root directory, and run this command to initiate the server:
    ```bash
    python src/server.py
    ```

    By default, the server runs on `http://localhost:5000` and provides two endpoints:
    - `/run-inference` [POST]: Runs super resolution on the images in the specified input directory.
    - `/health` [GET]: Returns `200 OK` if the server is running.

    `src/client.py` hasn't been updated to work with this modified version, so please do not try using this.
    <!-- Once the server is up and running, you can start the client by running the following command in a separate terminal instance. The argument `--input-dir` has a default value of `input`, but you can pass in any valid directory path:
    ```bash
    python src/client.py --input-dir input
    ``` -->

    Once the job completes its execution, a new `output` directory should have been generated, containing upscaled versions of the images in `input`. For each input image, e.g. `input/img1.png`, the corresponding upscaled image is formatted as follows: `output/img1-hr.png`.

# IMG-Super-Resolution

This simple project aims to utilize an existing state-of-the-art pre-trained model to upscale low-resolution images to a higher resolution, which is known in the computer vision community as **image super resolution**. The model used here is a Residual-in-Residual Dense Network (RRDN), which is based on an ESRGAN architecture. More about the specific implementation in this project can be found in the following repository: https://github.com/idealo/image-super-resolution. 

## Installation Instructions

Follow these steps to setup and run the service:

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

3. **Prepare Input Images**

    Add any low-resolution images that need to be upscaled to the top-level of the `input` directory. All image files are expected to be in one of these formats: `.png`, `.jpg`, or `.jpeg`.

4. **Run the Application:**

    Make sure you are in the root directory, and run this command to initiate the server:
    ```bash
    python src/server.py
    ```

    Once the serveris up and running, you can start the client by running the following command in a separate terminal instance:
    ```bash
    python src/client.py
    ```
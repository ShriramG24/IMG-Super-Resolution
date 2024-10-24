import logging
import os

import numpy as np
from PIL import Image

# logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask_ml.flask_ml_server.models import FileInput

class SuperResolution:
    def predict(self, data: list[FileInput], output_dir: str) -> list[dict]:
        from ISR.models import RRDN
        rrdn = RRDN(weights="gans")
        results = []
        for img in data:
            lr_img = np.array(Image.open(img.path))
            sr_img = Image.fromarray(rrdn.predict(lr_img))

            name = img.path.split("/")[-1].split(".")[0]
            sr_img.save(f"{output_dir}/{name}-hr.jpg")
            results.append(
                {"file_path": img.path, "result": f"{output_dir}/{name}-hr.jpg"}
            )
        return results

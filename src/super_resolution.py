import logging
import os

import numpy as np
from PIL import Image

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ISR.models import RRDN


class SuperResolution:
    def predict(self, data: list[dict], output_dir: str) -> list[dict]:
        rrdn = RRDN(weights="gans")
        results = []
        for img in data:
            lr_img = np.array(Image.open(img.file_path))
            sr_img = Image.fromarray(rrdn.predict(lr_img))

            name = img.file_path.split("/")[-1].split(".")[0]
            sr_img.save(f"{output_dir}/{name}-hr.jpg")
            results.append(
                {"file_path": img.file_path, "result": f"{output_dir}/{name}-hr.jpg"}
            )
        return results

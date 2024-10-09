import logging
import os

import numpy as np
from PIL import Image

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ISR.models import RRDN, RDN


class SuperResolution:
    def predict(self, data: list[dict], params: dict) -> list[dict]:
        weights = params.get("weights", "gans")
        output_dir = params.get("outputDir")
        scale = float(params.get("scale", 4.0))

        model = RRDN(weights=weights) if weights == 'gans' else RDN(weights=weights)
        results = []
        for img in data:
            lr_img = np.array(Image.open(img['file_path']))
            new_w, new_h = int(lr_img.shape[1] * scale), int(lr_img.shape[0] * scale)
            sr_img = Image.fromarray(model.predict(lr_img)).resize((new_w, new_h))

            name = img['file_path'].split("/")[-1]
            sr_img.save(f"{output_dir}/{name}")
            results.append(
                {"file_path": img['file_path'], "result": f"{output_dir}/{name}"}
            )
        return results

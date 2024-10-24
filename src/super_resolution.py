import os

import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask_ml.flask_ml_server.models import FileInput

class SuperResolution:
    def predict(self, data: list[FileInput], output_dir: str, weights='gans', scale=4.0) -> list[dict]:
        from ISR.models import RRDN, RDN
        model = RRDN(weights=weights) if weights == 'gans' else RDN(weights=weights)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = []
        for img in data:
            lr_img = np.array(Image.open(img.path))
            new_w, new_h = int(lr_img.shape[1] * scale), int(lr_img.shape[0] * scale)
            sr_img = Image.fromarray(model.predict(lr_img)).resize((new_w, new_h))

            name = img.path.split("/")[-1]
            name, ext = name.split(".")
            sr_img.save(f"{output_dir}/{name}_hr.{ext}")
            results.append(
                {"file_path": img.path, "result": f"{output_dir}/{name}_hr.{ext}"}
            )
        return results

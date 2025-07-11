from tire_vision.text.pipeline import TireAnnotationPipeline
from tire_vision.config import TireVisionConfig

import cv2

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    config = TireVisionConfig()
    pipeline = TireAnnotationPipeline(
        config.sidewall_segmentator_config,
        config.sidewall_unwrapper_config,
        config.ocr_config,
        config.index_config,
    )

    image_path = "/Users/n-zagainov/kolobok/ml/data/annotations/39.jpg"
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    result = pipeline(image)

    print(result)

    strings = [
        "Nokian",
        "TYRES",
        "Hakka Blue 2 SUV",
        "Hakka Blue 2",
        "Hakka Blue",
        "2 SUV",
        "225/60R17",
        "95H",
        "225/60R17 95H",
    ]

    result = pipeline.index.get_best_matches(strings)
    print(result)


if __name__ == "__main__":
    main()

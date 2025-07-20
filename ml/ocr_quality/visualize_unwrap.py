from pathlib import Path

import cv2

from tqdm import tqdm

from tire_vision.text.preprocessor.model import SidewallSegmentator
from tire_vision.text.preprocessor.unwrapper import SidewallUnwrapper
from tire_vision.config import SidewallSegmentatorConfig, SidewallUnwrapperConfig


def main():
    cfg_segmentator = SidewallSegmentatorConfig()
    cfg_segmentator.segmentator_checkpoint = "checkpoints/tmp.pth"
    cfg_unwrapper = SidewallUnwrapperConfig()

    model = SidewallSegmentator(cfg_segmentator)
    unwrapper = SidewallUnwrapper(cfg_unwrapper)

    input_dir = Path("/Users/n-zagainov/kolobok/ml/data/annotations")
    output_dir = Path("/Users/n-zagainov/kolobok/ml/data/annotations_unwrapped2")

    for img_path in tqdm(list(input_dir.iterdir())):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = model.forward(img)
        unwrapped = unwrapper.get_unwrapped_tire(img, mask)
        unwrapped = cv2.cvtColor(unwrapped, cv2.COLOR_RGB2BGR)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / img_path.name), unwrapped)


if __name__ == "__main__":
    main()

import subprocess
from itertools import product


def main():
    models = ["swin_v2_s", "effnet_b7", "densenet201", "googlenet"]
    with_scheduler = [True]
    gradient_clip = [None]
    with_synthetic = [True]
    with_aug = [False]
    with_clahe = [False]
    num_epochs = [50]

    for (
        model,
        with_synthetic,
        with_aug,
        with_clahe,
        with_scheduler,
        gradient_clip,
        num_epochs,
    ) in product(
        models,
        with_synthetic,
        with_aug,
        with_clahe,
        with_scheduler,
        gradient_clip,
        num_epochs,
    ):
        cmd = [
            "python",
            "scripts/train_depth_classification.py",
            "--model_name",
            model,
        ]

        if with_synthetic:
            cmd.append("--with_synthetic")
        if with_aug:
            cmd.append("--with_aug")
        if with_clahe:
            cmd.append("--with_clahe")
        if with_scheduler:
            cmd.append("--use_scheduler")
        if gradient_clip is not None:
            cmd.append(f"--gradient_clip={gradient_clip}")
        cmd.append(f"--num_epochs={num_epochs}")
        print(" ".join(cmd))

        subprocess.run(cmd)


if __name__ == "__main__":
    main()

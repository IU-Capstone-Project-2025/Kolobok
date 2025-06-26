import torch
import torch.nn as nn
from torchvision.models import (
    densenet201,
    googlenet,
    swin_v2_t,
    swin_s,
    efficientnet_b3,
    efficientnet_b7,
)


def get_swin_v2():
    model = swin_v2_t()
    model.head = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 1))
    return model


def get_swin_s():
    model = swin_s()
    model.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))
    return model


def get_effnet_b3():
    model = efficientnet_b3()
    model.classifier = nn.Sequential(nn.Linear(1536, 512), nn.SiLU(), nn.Linear(512, 1))
    return model


def get_effnet_b7():
    model = efficientnet_b7()
    model.classifier = nn.Sequential(nn.Linear(2560, 512), nn.SiLU(), nn.Linear(512, 1))
    return model


def get_densenet201():
    model = densenet201()
    model.classifier = nn.Sequential(nn.Linear(1920, 512), nn.ReLU(), nn.Linear(512, 1))
    return model


def get_googlenet():
    model = googlenet()
    model.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    return model


models = {
    "swin_v2": get_swin_v2,
    "swin_s": get_swin_s,
    "effnet_b3": get_effnet_b3,
    "effnet_b7": get_effnet_b7,
    "densenet201": get_densenet201,
    "googlenet": get_googlenet,
}


def get_depth_estimator(model_name: str, checkpoint_path: str):
    model = models[model_name]()
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    )
    model.eval()
    return model

import os
import base64
import io

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import numpy as np

from utils import get_thread_stats, add_annotations, extract_tire_info

import logging


app = FastAPI()
bearer_scheme = HTTPBearer()

API_TOKEN = os.environ["API_TOKEN"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger("app")


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    """
    Ensure Authorization: Bearer <token> is present and valid.
    """
    token = credentials.credentials
    if credentials.scheme.lower() != "bearer" or token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


class ImageRequest(BaseModel):
    image: str


def validate_image(b64_data: str) -> None:
    try:
        raw = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except (base64.binascii.Error, UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Image is corrupted or not valid")


@app.post("/api/v1/analyze_thread")
def analyze_thread(
    req: ImageRequest,
    token: str = Depends(verify_token),
):
    logger.info("/api/v1/analyze_thread: received request")
    validate_image(req.image)

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))))
    logger.info("/api/v1/analyze_thread: running thread pipeline")
    result = get_thread_stats(image)
    if result["success"] == 0:
        return result

    image_with_annotations = add_annotations(result["cropped_image"], result["spikes"])
    logger.info("/api/v1/analyze_thread: thread pipeline completed")

    pil_image = Image.fromarray(image_with_annotations)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "success": 1,
        "thread_depth": result["depth"],
        "spikes": result["spikes"],
        "image": img_str,
    }


@app.post("/api/v1/extract_information")
def extract_information(
    req: ImageRequest,
    token: str = Depends(verify_token),
):
    validate_image(req.image)

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))))

    logger.info("/api/v1/extract_information: running tire information extraction")
    result = extract_tire_info(image)
    logger.info("/api/v1/extract_information: tire information extraction completed")

    return result

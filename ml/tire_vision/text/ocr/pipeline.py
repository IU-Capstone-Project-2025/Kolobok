import base64
import json
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np
import replicate
import io
from traceback import format_exc
import logging
import re

from tire_vision.config import OCRConfig


class TireOCR:
    """OCR class for extracting tire information from images."""

    def __init__(self, config: OCRConfig):
        self.config = config

        self.logger = logging.getLogger("ocr")
        self.logger.info("TireOCR module initialized")

    def extract_tire_info(
        self,
        images: list[np.ndarray],
        prompt: str,
    ) -> Dict[str, Optional[str]]:
        """Extract tire information from one or more images.

        Args:
            images: List of images (numpy arrays in RGB format)
            prompt: Prompt to send to the VLM.

        Returns:
            Dictionary with manufacturer, model, and tire_size_string fields
        """
        file_inputs = [self._prepare_image(img) for img in images]

        try:
            result = self._get_llm_response(file_inputs, prompt)
            tire_info = self._parse_llm_response(result)
            return tire_info
        except Exception:
            self.logger.error(format_exc())
            self.logger.error(
                "Error during OCR processing. Falling back to default values"
            )
            return self._get_default_response()

    def _prepare_image(self, image: np.ndarray) -> str:
        """Prepare image for OCR processing."""
        pil_image = Image.fromarray(image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:application/octet-stream;base64,{b64_data}"

    def _get_llm_response(self, file_inputs: list[str], prompt: str) -> str:
        """Get response from LLM model."""
        result = ""
        for event in replicate.stream(
            self.config.model_name,
            input={
                "top_p": self.config.top_p,
                "prompt": prompt,
                "image_input": file_inputs,
                "temperature": self.config.temperature,
                "presence_penalty": self.config.presence_penalty,
                "frequency_penalty": self.config.frequency_penalty,
                "max_completion_tokens": self.config.max_completion_tokens,
            },
        ):
            result += str(event)

        self.logger.info(f"LLM response: {result}")
        return result

    def _parse_llm_response(self, result: str) -> Dict[str, Optional[str]]:
        match = re.search(r"\{.*\}", result, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response")
        json_str = match.group(0)
        tire_info = json.loads(json_str)
        self.logger.info(f"Parsed OCR result: {tire_info}")

        return {
            "manufacturer": tire_info.get("manufacturer"),
            "model": tire_info.get("model"),
            "tire_size_string": tire_info.get("tire_size_string"),
        }

    def _get_default_response(self) -> Dict[str, Optional[str]]:
        """Return default response when processing fails."""
        self.logger.info("Falling back to default values")
        return {"manufacturer": None, "model": None, "tire_size_string": None}

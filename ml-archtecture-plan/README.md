# Pipeline

Our service requires some `ML elements` to be implemented. Specifically:

- `Tire classification` - The service must be able to extract the model, manufacturer, and size of the tire from the side-view image.
    - `Tire segmentator` - Entrance of the pipeline. Required to the rubber tire ring on the image for further processing.
    - `Unfolder` - Algorithmic CV technique to *unwrap* the detectd tire ring into a rectangular sign-like image for ease of processing.
    - `OCR` - Optical Character Recognition to extract the model, manufacturer, and size of the tire from the unwrapped image.
    - `Index (optional)` - for enhanced quality of the OCR, we can use a database of existing tire models to extract closest matches to the detected information.

- `Tire quality estimation` - The service aims to estimate the quality of the tire based on thread-view image (remaining thread depth, number of spikes)
    - `Thread segmentator` - Entrance of the pipeline. Required to find the tire tread on the image.
    - `Thread depth estimator` - ML model trainded to approximate the remaining thread depth on the image.
    - `Spike detector` - Segmentation or Object Detection ML model for identifying spikes on the tire thread.
    - `Spike classifier` - Classification ML model for estimating the number of spikes on the tire thread.



# Research

We conducted a research to find approaches suitable for the implementation of the above elements.

## Tire segmentator
For segmentation of the tire ring, we will use open-source labeled dataset and models of this [Roboflow project.](https://universe.roboflow.com/segmentation-k0zny/tyre-wpkj0)

## Unfolder
For tire *unwrapping* we will use [OpenCV implementation](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga49481ab24fdaa0ffa4d3e63d14c0d5e4) of Polar transformation.

## OCR
For OCR we will use open-source [Tesseract](https://github.com/tesseract-ocr/tesseract) model.

## Thread segmentator 
For thread segmentation we will use open-vocabulary segmentation pipeline with Side-Adapter Network proposed by this paper: [ArXiv](https://arxiv.org/pdf/2302.12242)

## Thread depth estimator & Spike classifier
For depth estimation we will use our hand-labeled dataset to train a model of the `GoogLeNet` architecture: [ArXiv](https://arxiv.org/pdf/1409.4842)

## Spike detector
For spike detection we will use our hand-labeled dataset to train a segmentation model of `SegFormer` architecture: [ArXiv](https://arxiv.org/pdf/2105.15203)
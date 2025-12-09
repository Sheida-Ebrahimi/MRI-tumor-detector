# MRI Brain Tumor Detector

A simple Flask web application that uses a fine-tuned ResNet-18 model to detect brain tumors from MRI scans. Users can upload an image and receive a prediction and an optional Grad-CAM heatmap.

## Features

* Upload MRI images through a web interface

* Binary classification: tumor / no tumor

* Confidence score

* Grad-CAM heatmap

* Flask backend with PyTorch inference

## Installation

Create and activate a virtual environment:
```
python3 -m venv venv source venv/bin/activate
```
## Install dependencies:
```
pip install -r requirements.txt
```

## Running the App
```
python app.py
```


## Open in your browser:

http://127.0.0.1:5000/

## Test Images

Sample MRI scans are located under:

```data/raw/brain-mri/```

## Model File

The trained model is stored in:

```notebooks/brain_tumor_resnet18.pt```


## Disclaimer

This project is for educational purposes only and is not a medical diagnostic tool.

## Demo
Detailed demo on how to set up and run the app: https://www.youtube.com/watch?v=q1kjIoUEupo

[![Watch the video](https://img.youtube.com/vi/q1kjIoUEupo/maxresdefault.jpg)](https://youtu.be/q1kjIoUEupo)





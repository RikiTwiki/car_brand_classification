import os
from IPython import display
import ultralytics
import sys
from typing import List

import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate
import os
from keras.utils import plot_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from fastai.vision.all import *
from PIL import Image
# import torch

import matplotlib.pyplot as plt

# HOME = os.getcwd()

# # Define your home directory
HOME = '/home/sanarip03/Desktop/бренд_машин'  # Replace with your home directory path

# # Change directory and clone the repository
# os.chdir(HOME)
# # os.system('git clone https://github.com/ifzhang/ByteTrack.git')

# # Change to the ByteTrack directory
# os.chdir(f'{HOME}/ByteTrack')

# # Modify requirements.txt
# with open('requirements.txt', 'r') as file:
#     requirements = file.read()

# requirements = requirements.replace('onnx==1.8.1', 'onnx==1.9.0')

# with open('requirements.txt', 'w') as file:
#     file.write(requirements)

# # Install requirements and setup
# os.system('pip3 install -q -r requirements.txt')
# os.system('python3 setup.py -q develop')
# os.system('pip install -q cython_bbox')
# os.system('pip install -q onemetric')
# os.system('pip install -q loguru lap thop')

# Add ByteTrack to the system path
# sys.path.append(f"{HOME}/ByteTrack")

# # Import yolox and print version
# import yolox
# print("yolox.__version__:", yolox.__version__)

# import yolox
# print("yolox.__version__:", yolox.__version__)

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# import supervision
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

MODEL = "yolov8x.pt"

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0]

SOURCE_VIDEO_PATH = f"{HOME}/test.jpeg"

# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
# acquire first video frame
iterator = iter(generator)
frame = next(iterator)
# model prediction on single frame and conversion to supervision Detections
results = model(frame)
detections = Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
)

# Filter detections for 'car' class only
car_class_id = [k for k, v in CLASS_NAMES_DICT.items() if v.lower() == 'car'][0]
car_detections = [
    det for det in detections
    if det[2] == car_class_id
]

# Find the largest car detection
largest_car_detection = None
max_area = 0
for det in car_detections:
    x1, y1, x2, y2 = det[0]
    area = (x2 - x1) * (y2 - y1)
    if area > max_area:
        max_area = area
        largest_car_detection = det

# Proceed only if a car is detected
if largest_car_detection:
    # format custom label for the largest car
    labels = [f"{CLASS_NAMES_DICT[largest_car_detection[2]]} {largest_car_detection[1]:0.2f}"]

    # annotate and display frame with only the largest car detection
    frame = box_annotator.annotate(frame=frame, detections=[largest_car_detection], labels=labels)

    # %matplotlib inline
    show_frame_in_notebook(frame, (16, 16))
else:
    print("No cars detected in this frame.")

# Cropping function
def crop_detections(frame, detections):
    cropped_images = []
    for detection in [detections[0]]:
        # Extracting the bounding box coordinates from the detection array
        x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
        cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_images.append(cropped_image)
    return cropped_images

# Rest of your existing code
# ...
# Acquire first video frame, perform model prediction, etc.

# Cropping detected parts of the frame
cropped_frames = crop_detections(frame, detections.xyxy)

# Function to display images in a Jupyter Notebook
def show_images(images, figsize):
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        # plt.imshow(image)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

# Display the cropped images
show_images(cropped_frames, (16, 16))

img = cropped_frames[0][:,:,::-1]

# data = ImageDataLoaders.from_folder('../imgs_zip/imgs/train', train='.', valid_pct=0.2, 
#                                    size=224, item_tfms=Resize(224))

learn_car_brand = load_learner('/home/sanarip03/Desktop/бренд_машин/Image-Classification-using-fastai-main/export (3).pkl')
learn_car_color = load_learner('/home/sanarip03/Desktop/бренд_машин/Image-Classification-using-fastai-main/export_car_color_detection.pkl')
learn_car_body = load_learner('/home/sanarip03/Desktop/бренд_машин/Image-Classification-using-fastai-main/export_car_body.pkl')

def predict_image(img):
    # Open the image
    # img = Image.open(image_path)

    # Resize the image (if necessary)
    img = Image.fromarray(img)
    img_resized = img.resize((224, 224))

    # Predict
    pred_class, pred_idx, probs = learn_car_brand.predict(img_resized)
    pred_class_color, pred_idx_color, probs_color = learn_car_color.predict(img_resized)
    pred_class_body, pred_idx_body, probs_body = learn_car_body.predict(img_resized)

    # Print results
    print(f"Predicted class brand: {pred_class}")
    print(f"Predicted probabilities brand: {probs[pred_idx]:.4f}")

    print(f"Predicted class color: {pred_class_color}")
    print(f"Predicted probabilities color: {probs_color[pred_idx_color]:.4f}")

    print(f"Predicted class body: {pred_class_body}")
    print(f"Predicted probabilities body: {probs_body[pred_idx_body]:.4f}")

    # Display the image
    img_resized.show()

# Example usage
predict_image(img)

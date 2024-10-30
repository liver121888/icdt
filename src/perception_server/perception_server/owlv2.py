# Import necessary libraries
import time
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

# Model encapsulation for object detection
class ObjectDetectionModel:
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble"):
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
    
    # Preprocess image with the model processor
    def preprocess_image(self, image, texts):
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        return inputs

    # Perform object detection
    def detect_objects(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    # Unnormalize and preprocess the image
    @staticmethod
    def get_preprocessed_image(pixel_values):
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (
            (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + 
            np.array(OPENAI_CLIP_MEAN)[:, None, None]
        )
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

        # Crop Bottom 400px
        unnormalized_image = unnormalized_image[:-416, :, :]
        return Image.fromarray(unnormalized_image)

    # Post-process results to extract bounding boxes, scores, and labels
    def post_process_results(self, outputs, image, texts, threshold=0.2):
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        
        print("Detected Objects:")
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {texts[label]} with confidence {round(score.item(), 3)} at location {box}")
        return boxes.numpy(), scores.numpy(), labels.numpy()
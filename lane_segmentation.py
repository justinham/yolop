"""
Lane Segmentation Module for YOLOP
Takes realtime image and returns lane segmentation map (640x480)
"""

import os
import sys
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image

# Add YOLOP to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from lib.config import cfg, update_config
from lib.models import get_net
from lib.utils.utils import select_device


# Image preprocessing
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


class LaneSegmenter:
    """YOLOP-based lane segmentation predictor"""
    
    def __init__(self, weights_path=None, device='cpu', img_size=640):
        if weights_path is None:
            weights_path = os.path.join(BASE_DIR, 'weights/End-to-end.pth')
        
        self.img_size = img_size
        self.device = device
        
        # Load model
        print(f"Loading YOLOP model from {weights_path}...")
        self.model = get_net(cfg)
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Warmup
        dummy_img = torch.zeros((1, 3, img_size, img_size), device=self.device)
        _ = self.model(dummy_img)
        print("Model loaded and ready!")
    
    def predict(self, image, output_size=(640, 480)):
        """
        Run lane segmentation on input image.
        
        Args:
            image: numpy array (BGR format from OpenCV) or PIL Image
            output_size: tuple (width, height) for output mask
            
        Returns:
            lane_mask: numpy array (H, W) binary mask where 1 = lane, 0 = non-lane
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            # Assume BGR from OpenCV, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Resize to model input size
        image_resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Transform
        img_tensor = transform(image_resized).to(self.device)
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            _, _, ll_seg_out = self.model(img_tensor)
        
        # Get lane segmentation mask (class 1 = lane line)
        ll_seg_mask = torch.nn.functional.interpolate(
            ll_seg_out, scale_factor=1, mode='bilinear', align_corners=False
        )
        _, lane_mask = torch.max(ll_seg_mask, 1)  # argmax across classes
        lane_mask = lane_mask.squeeze().cpu().numpy()
        
        # Resize to output size
        lane_mask_resized = cv2.resize(
            lane_mask.astype(np.uint8), 
            output_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        return lane_mask_resized
    
    def predict_colored(self, image, output_size=(640, 480)):
        """
        Run lane segmentation and return colored visualization.
        
        Args:
            image: numpy array (BGR) or PIL Image
            output_size: tuple (width, height)
            
        Returns:
            colored_mask: numpy array (BGR) with lanes in red, background in black
        """
        lane_mask = self.predict(image, output_size)
        
        # Create colored visualization: lanes in red
        colored = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        colored[lane_mask == 1] = [0, 0, 255]  # Red for lanes
        colored[lane_mask == 0] = [0, 0, 0]   # Black for background
        
        return colored


def get_lane_segmentation(image, model=None, weights_path=None, device='cpu'):
    """
    Convenience function to get lane segmentation mask.
    
    Args:
        image: numpy array (BGR, from cv2) or PIL Image
        weights_path: optional path to weights file
        device: 'cpu' or 'cuda'
        
    Returns:
        lane_mask: numpy array (480, 640) binary mask
    """
    if model is None:
        model = LaneSegmenter(weights_path=weights_path, device=device)
    
    return model.predict(image, output_size=(640, 480))


# Example usage
if __name__ == '__main__':
    # Initialize segmenter
    segmenter = LaneSegmenter(weights_path='weights/End-to-end.pth', device='cpu')
    
    # Test with an image
    test_img_path = 'test.jpg'
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        
        # Get binary mask
        lane_mask = segmenter.predict(img)
        print(f"Lane mask shape: {lane_mask.shape}")
        print(f"Lane pixels: {np.sum(lane_mask == 1)}")
        
        # Get colored visualization
        colored = segmenter.predict_colored(img)
        cv2.imwrite('lane_mask_output.png', colored)
        print("Saved lane_mask_output.png")
    else:
        print(f"Test image not found: {test_img_path}")
        print("Usage: segmenter = LaneSegmenter()")
        print("       mask = segmenter.predict(your_image)")
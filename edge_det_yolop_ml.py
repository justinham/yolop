"""
YOLOP-based Lane Detection for stream21.py
Drop-in replacement for edge_det_mask_range using ML-based lane segmentation
"""

import numpy as np
import cv2
import torch
import sys
import os

# Add YOLOP path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import YOLOP components (lazy load to avoid loading model multiple times)
_lane_segmenter = None


def _get_lane_segmenter():
    """Lazy-load the YOLOP model"""
    global _lane_segmenter
    if _lane_segmenter is None:
        from lane_segmentation import LaneSegmenter
        weights_path = os.path.join(BASE_DIR, 'weights/End-to-end.pth')
        _lane_segmenter = LaneSegmenter(weights_path=weights_path, device='cpu')
        print("[YOLOP] Model loaded successfully")
    return _lane_segmenter


def edge_det_yolop(frame, hough_thr=100, blur_thr=5, mask=None, output_size=(640, 480)):
    """
    YOLOP-based lane detection - drop-in replacement for edge_det_mask_range
    
    Args:
        frame: input image (BGR, from cv2)
        hough_thr: Hough transform threshold (kept for API compatibility, not used)
        blur_thr: blur kernel size (kept for API compatibility, not used)  
        mask: ROI mask (optional, will be applied to output)
        output_size: (width, height) for output - default 640x480
        
    Returns:
        edges: binary edge map from lane segmentation
        lines: detected line segments (same format as HoughLinesP)
    """
    segmenter = _get_lane_segmenter()
    
    # Get lane segmentation mask (480x640 binary)
    lane_mask = segmenter.predict(frame, output_size=output_size)
    
    # Get colored visualization for debugging
    lane_colored = segmenter.predict_colored(frame, output_size=output_size)
    
    # Convert binary mask to edge map (Canny on the mask to get contours)
    # Lane pixels = 1, background = 0
    edges = cv2.Canny(lane_mask * 255, 100, 200)
    
    # Apply mask if provided (zero out edges outside ROI)
    if mask is not None:
        # Resize mask to match output size if needed
        if mask.shape[:2] != (output_size[1], output_size[0]):
            mask = cv2.resize(mask, output_size)
        edges = cv2.bitwise_and(edges, mask)
    
    # Detect line segments from the lane mask using HoughLinesP
    # This gives us lines in the same format as the traditional approach
    hough_lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=hough_thr,      # from parameter
        minLineLength=30,          # minimum line length
        maxLineGap=20              # maximum gap between points
    )
    
    # Convert to HoughLines format (rho, theta) for compatibility
    # Or return None if no lines detected
    lines = hough_lines
    
    return edges, lines


def edge_det_yolop_simple(frame, hough_thr=100, blur_thr=5, mask=None, output_size=(640, 480)):
    """
    Simplified YOLOP lane detection - returns just the lane mask as edges
    
    Args:
        frame: input image (BGR)
        hough_thr: not used (kept for API compatibility)
        blur_thr: not used (kept for API compatibility)
        mask: optional ROI mask
        output_size: (width, height)
        
    Returns:
        edges: binary edge map from lane segmentation
        lines: None (no line vectorization)
    """
    segmenter = _get_lane_segmenter()
    
    # Get lane segmentation mask
    lane_mask = segmenter.predict(frame, output_size=output_size)
    
    # Convert to edge map
    edges = cv2.Canny(lane_mask * 255, 100, 200)
    
    # Apply mask if provided
    if mask is not None:
        if mask.shape[:2] != (output_size[1], output_size[0]):
            mask = cv2.resize(mask, output_size)
        edges = cv2.bitwise_and(edges, mask)
    
    # Return None for lines (just return edge map)
    return edges, None


# Integration example for stream21.py
# 
# To use in stream21.py, replace:
#   
#   edges, lines = edge_det_mask_range(ori_img, hough_thr, blur_thr, mask)
#
# with:
#
#   from edge_det_yolop_ml import edge_det_yolop
#   edges, lines = edge_det_yolop(ori_img, hough_thr, blur_thr, mask)


if __name__ == '__main__':
    # Test the function
    import os
    
    test_img = os.path.join(BASE_DIR, 'test.jpg')
    if os.path.exists(test_img):
        print("Testing YOLOP lane detection...")
        img = cv2.imread(test_img)
        
        # Create a simple mask (full image)
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Run YOLOP detection
        edges, lines = edge_det_yolop(img, hough_thr=100, blur_thr=5, mask=mask)
        
        print(f"Edge map shape: {edges.shape}")
        if lines is not None:
            print(f"Detected {len(lines)} line segments")
        else:
            print("No line segments detected")
        
        # Save results
        cv2.imwrite(os.path.join(BASE_DIR, 'yolop_edges.png'), edges)
        print("Saved yolop_edges.png")
    else:
        print("Test image not found")
        print("Usage from stream21.py:")
        print("  from edge_det_yolop_ml import edge_det_yolop")
        print("  edges, lines = edge_det_yolop(ori_img, hough_thr, blur_thr, mask)")
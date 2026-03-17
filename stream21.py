## test
# change roi world during calibration (green mask area)

## test
# use dynamic buffer, whenever veh move too close (5m), instead of counting valid frame, 
# but looking into all past 7s 210 frame date, selecting all valid estimation, filtering out outliers, 
# then make decision. also visualize if decision is made and it's reference. 

## try yolop demo_j.py 
# python3.11 tools/demo_j.py --source 0

# ffplay /dev/video0 88inch left, 98 inch right

# exposure control (x)
# (check) black-color line remove first (binary black-white)
## sift filter (scale invariant) (x different on logic than hough)
## (check okay but need test) hough with dynamic threshold (decrease img up half area thr by 50%)
# (check) decision earlier (OSTU + dyn hough enhance this feature)
# (check) whenever line detected, check how close it is to the spot, if tooo far, dismiss

# mask & grid for104x86 fov 

import cv2
import numpy as np
import json
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import time
import requests
import math  
from ultralytics import YOLO
from datetime import datetime
from edge_det_yolop_ml import edge_det_yolop


#########################

# Load JSON config
with open("cam_config.json", "r") as f:
    config = json.load(f)

# Vehicle
vehicle = config["vehicle"]
vehicle_width = vehicle["width_m"]
vehicle_height = vehicle["height_m"]
vehicle_length = vehicle["length_m"]
wheel_dis = vehicle["wheel_distance_m"]

# Lane
lane = config["lane"]
lane_length = lane["lane_length_m"]
lane_offset = lane["lane_offset_m"]
lane_width = lane["lane_width_m"]
spot_width = lane["spot_width_m"]
spot_length = lane["spot_length_m"]
lane_tape_right = lane["lane_tape_right_m"]

# ROI
roi = config["roi_world"]
ROI_LB = tuple(roi["left_bottom"])
ROI_LT = tuple(roi["left_top"])
ROI_RB = tuple(roi["right_bottom"])
ROI_RT = tuple(roi["right_top"])
valid_depth = [ROI_LB[1], ROI_LT[1]]

# mask
mask = config["grid_mask"]
mask_L = mask["mask_L_m"]
mask_R = mask["mask_R_m"]
mask_D = mask["mask_D_m"]
mask_U = mask["mask_U_m"]

# Intrinsics
intrinsic = config["intrinsic"]
w = intrinsic["image_width_px"]
h = intrinsic["image_height_px"]

K = np.array(intrinsic["camera_matrix"])
dist_coeffs = np.array(intrinsic["distortion_coefficients"])

hfov_deg = intrinsic["hfov_deg"]
vfov_deg = intrinsic["vfov_deg"]

# Extrinsics
extrinsic = config["extrinsic"]
camera_height = extrinsic["camera_height_m"]
pitch_angle = np.radians(extrinsic["pitch_angle_deg"])

print("Parameters loaded successfully.")

#############


## init para

## file location
can_h_fn = "./birdview/hummer_path/can_heading.txt"
gps_ref_fn = "./birdview/hummer_path/gps_ref_p2.txt"
gps_fn = "./birdview/hummer_path/gps.txt"
path_fn = "./birdview/hummer_path/pathx5_can_heading.txt"
path_fn_pre = "./birdview/hummer_path/pathx5_can_heading_pre.txt"
path_intp_fn = "./birdview/hummer_path/path_local_den_1_stage_can_heading.txt"
steer_fn = "./birdview/hummer_path/steer.txt"


v2spot_file_path = "vehicle2spot_path.txt"
v2spot_file_path_world = "vehicle2spot_path0.txt"
track_fn = "track_test.txt"

## endpoint validation
rtp_valid = False
rbp_valid = False
ltp_valid = False
lbp_valid = False


## range where trigger lane detection
Det_heading_trigger = [0, 50]
Det_distance_trigger = [1, 8]
Decision_make_distance = 6 # during the test, lane is visible between 4m-6.5m

## only trigger when delta sensed heading is mall enough (finish turn)
Delta_heading_thr = 2

## range when reset counting (larger range 100 means no reset in 100m)
Det_dis_reset = 10 # 10

## edge & line detection thrs
blur_thr = 7 # test good at 9, smaller resolution lower blur to 5-7
hough_thr = 50 # 100, smaller resolution lower to 60
angle_thr_deg = 85
sta_dyn_thr = 2.5 ## if cam sensed location is too far away from the statice planned target, no count


## when to make path change decision
decision_point = 10 ## how many right point estimation valid before making path modification
LR_vary_thr = 0.5 # left/right lane agree in estimation
decision_point2 = 5
  
## scope of path replan (x,y,h)
UPDATE_X_TAG = True
UPDATE_Y_TAG = True
UPDATE_H_TAG = False

## person detection and emg stop
emg_stop_tag = True
YOLO_DET = False
# YOLO_SEG = False

## switch on path replan execution
PATH_REP = True # True #True # True


###########################

def get_curr_timing():
    # Get current UTC time
    date = datetime.utcnow()

    # Calculate total seconds since the epoch
    seconds = (date - datetime(1970, 1, 1)).total_seconds()

    # Convert to milliseconds and round
    milliseconds = round(seconds * 1000)

    return milliseconds


def load_interp_path():
    
    with open(path_intp_fn, "r") as f:
        data = json.load(f)  # data is a list of lists

    # Extract first 3 values from each sublist
    xyh = [item[:3] for item in data]

    # Convert to numpy array for easier saving
    xyh = np.array(xyh)

    return xyh


def search_heading_by_loc(x_query, y_query, arr):

    diff = arr[:, :2] - np.array([x_query, y_query])
    dist2 = np.sum(diff**2, axis=1)

    # Find index of closest point
    idx_min = np.argmin(dist2)

    # Get the closest point
    closest_point = arr[idx_min]

    return closest_point


def plot_lane_history(history_data):
    """
    history_data: dictionary from tracker.get_history_list()
    containing 'left_start', 'left_end', 'right_start', 'right_end'
    """
    frames = np.arange(len(history_data['right_start']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- Plot 1: Lateral (X) Movement ---
    for key, color, label in [
        ('left_start', 'lightgreen', 'L-Start'), ('left_end', 'darkgreen', 'L-End'),
        ('right_start', 'salmon', 'R-Start'), ('right_end', 'red', 'R-End')
    ]:
        data = history_data[key]
        # Extract X coordinates, handling None values
        x_vals = [pt[0] if pt else None for pt in data]
        ax1.plot(frames, x_vals, label=label, color=color, linewidth=2)

    ax1.set_title("Lateral (X) Position Change (Meters)")
    ax1.set_ylabel("Meters (Left - / Right +)")
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Longitudinal (Y) Movement ---
    for key, color, label in [
        ('left_start', 'lightgreen', 'L-Start'), ('left_end', 'darkgreen', 'R-Start'),
        ('right_start', 'salmon', 'R-Start'), ('right_end', 'red', 'R-End')
    ]:
        data = history_data[key]
        # Extract Y coordinates
        y_vals = [pt[1] if pt else None for pt in data]
        ax2.plot(frames, y_vals, label=label, color=color, linewidth=2)

    ax2.set_title("Longitudinal (Y) Distance Change (Meters)")
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Distance Forward (m)")
    ax2.legend(loc='center right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('location_change.png')
    plt.close(fig) 
    
    # --- Top-Down Path Plot ---
    plt.figure(figsize=(8, 10)) # Increased size slightly to accommodate text

    # for key, color in [('left_start', 'Greens'), ('left_end', 'YlGn'), 
                       # ('right_start', 'Reds'), ('right_end', 'OrRd')]:
    for key, color in [('right_start', 'Reds')]:
        
        # Filter valid points and keep track of their original indices
        valid_indices = [i for i, pt in enumerate(history_data[key]) if pt is not None]
        data = np.array([history_data[key][i] for i in valid_indices])

        if len(data) > 0:
            # Plot the points
            plt.scatter(data[:,0], data[:,1], c=np.arange(len(data)), cmap=color, s=15)
            
            # Add index labels beside each point
            for i, idx in enumerate(valid_indices):
                # Coordinates for the text (with a tiny offset for readability)
                plt.text(data[i, 0] + 0.05, data[i, 1] + 0.05, str(idx), 
                         fontsize=8, alpha=0.7)

    plt.axvline(0, color='black', linestyle='--') 
    plt.title("Top-Down Trajectory with Frame Indices")
    plt.xlabel("Lateral (m)")
    plt.ylabel("Forward (m)")
    plt.xlim(-4, 4)
    plt.grid(True, alpha=0.3)
    plt.savefig('top_down_trajectory.png')
    plt.close()




def plot_pc_history(history_data, tag):

    # Extract data and filter out None values
    raw_data = history_data[tag]
    
    # Zip frames with data so we only plot valid points
    # pt[0] is Lateral (X), pt[1] is Forward (Y)
    valid_points = [(i, pt[0], pt[1]) for i, pt in enumerate(raw_data) if pt is not None]
    
    if not valid_points:
        print("No valid data to plot.")
        return

    frames, x_vals, y_vals = zip(*valid_points)

    # --- Plot 1: X and Y Position over Time ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subplot A: Lateral (X)
    ax1.plot(frames, x_vals, label='Lateral (X)', color='blue', linewidth=2)
    ax1.set_title("Parking Center Position Change")
    ax1.set_ylabel("X (Meters: Left - / Right +)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Subplot B: Forward (Y)
    ax2.plot(frames, y_vals, label='Forward (Y)', color='blue', linewidth=2)
    ax2.set_ylabel("Y (Meters: Distance to Center)")
    ax2.set_xlabel("Frame")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig('pc_location_change.png')
    plt.close(fig)   

    # --- Plot 2: Top-Down Path (Remains the same) ---
    fig = plt.figure(figsize=(8, 10))

    # 1. Filter data while preserving original indices
    valid_indices = [i for i, pt in enumerate(raw_data) if pt is not None]
    path_data = np.array([raw_data[i] for i in valid_indices])

    if len(path_data) > 0:
        # 2. Draw the scatter points
        scatter = plt.scatter(path_data[:, 0], path_data[:, 1],
                             c=np.arange(len(path_data)), cmap='Greens', s=15)
        
        # 3. Annotate each point with its frame index
        for i, idx in enumerate(valid_indices):
            # We add a small offset (+0.05) so the text doesn't sit on the dot
            plt.annotate(str(idx), 
                         (path_data[i, 0], path_data[i, 1]),
                         textcoords="offset points", 
                         xytext=(5, 5), 
                         fontsize=8, 
                         alpha=0.6)

    plt.axvline(0, color='black', linestyle='--')
    plt.title("Top-Down Trajectory (Path Map) with Frame Indices")
    plt.xlabel("Lateral (m)")
    plt.ylabel("Forward (m)")
    plt.xlim(-4, 4)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pc_top_down_trajectory.png')
    plt.close(fig)


def lane_history_filtered(history_data):
    filtered_history = {}
    
    for key in ['left_start', 'left_end', 'right_start', 'right_end', 'park_cen', 'park_cen_veh', 'park_cen_world']:
        raw_pts = history_data[key]
        
        # 1. Handle Nones (Interpolate so the filter doesn't break)
        x_raw = np.array([p[0] if p else np.nan for p in raw_pts])
        y_raw = np.array([p[1] if p else np.nan for p in raw_pts])
        
        # Simple linear interpolation for missing frames
        x_clean = pd.Series(x_raw).interpolate(method='linear')
        y_clean = pd.Series(y_raw).interpolate(method='linear')

        # 2. Apply Savitzky-Golay Smoothing
        # Window size 15 is usually good for 30fps video
        filtered_history[key] = list(zip(
            savgol_filter(x_clean, 15, 2),
            savgol_filter(y_clean, 15, 2)
        ))

    return filtered_history
    
#########################


def est_distance_grid(frame):
    h, w = frame.shape[:2]
    
    # Draw Depth Lines (Lane width markers)
    # Using the standard 9ft (~2.7m) parking width as the outer bounds
    for xw in [mask_L, 0.0, mask_R]: 
        pts = []
        for yw in np.linspace(1.0, mask_U, 20):
            # pt = project_point_with_fl(xw, yw)
            pt = project_point_with_fov_to_img(xw, yw)
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                pts.append(pt)
        
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts)], False, (0, 255, 255), 1)

            # Labeling
            label_pt = pts[0]
            cv2.putText(frame, f"{xw}m", (label_pt[0]-25, label_pt[1]+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    ## tire
    for xw in [-wheel_dis/2, wheel_dis/2]: 
        pts = []
        for yw in np.linspace(0.5, 0.6, 20):
            # pt = project_point_with_fl(xw, yw)
            pt = project_point_with_fov_to_img(xw, yw)
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                pts.append(pt)
        
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts)], False, (0, 0, 255), 2)

            # Labeling
            label_pt = pts[1]
            cv2.putText(frame, f"wheel", (label_pt[0]-25, label_pt[1]-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # lane
    for xw in [-spot_width/2, spot_width/2]: 
        pts = []
        for yw in np.linspace(1.0, 5, 20):
            # pt = project_point_with_fl(xw, yw)
            pt = project_point_with_fov_to_img(xw, yw)
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                pts.append(pt)
        
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts)], False, (255, 255, 0), 2)

            # Labeling
            label_pt = pts[0]
            cv2.putText(frame, f"lane", (label_pt[0]-25, label_pt[1]+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)



    # Draw Distance Markers (1m to 9m)
    for yw in range(1, int(mask_U)+1):
        pts = []
        for xw in np.linspace(mask_L, mask_R, 10):
            # pt = project_point_with_fl(xw, yw)
            pt = project_point_with_fov_to_img(xw, yw)
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                pts.append(pt)
        
        if len(pts) > 1:
            # Red for close-range (<2m), Yellow otherwise
            # color = (0, 0, 255) if yw <= 2 else (0, 255, 255)
            color = (0, 255, 255)
            cv2.polylines(frame, [np.array(pts)], False, color, 1)
            
            # Labeling
            label_pt = pts[0]
            cv2.putText(frame, f"{yw}m", (label_pt[0]-30, label_pt[1]+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Generate final mask and corners
    # mask, corners_10m = get_10m_mask(h, w)
    mask, corners_10m = get_15m_mask(h, w)
    # mask, corners_10m = get_15m_mask_oov(h, w)

    return frame, np.array(corners_10m, dtype=np.int32), mask

# Coordinate Projection from dis to imge pixel
def project_point_with_fl(xw, yw):
    """
    xw: Lateral distance from center (meters)
    yw: Longitudinal distance forward (meters)
    """
    # Transform Ground (xw, yw, 0) to Camera Coordinates (Pitched Down)
    # Z is forward, Y is down, X is right
    z_cam = yw * np.cos(pitch_angle) + camera_height * np.sin(pitch_angle)
    y_cam = camera_height * np.cos(pitch_angle) - yw * np.sin(pitch_angle)
    x_cam = xw
    
    # Clipping: Only project points in front of the camera
    if z_cam <= 0.1: 
        return None
    
    # Pinhole projection to pixel coordinates
    u = int((x_cam * focal_length / z_cam) + w / 2)
    v = int((y_cam * focal_length / z_cam) + h / 2)
    
    return (u, v)



def project_point_with_fov_to_img(xw, yw):
    """
    xw: lateral distance (meters, +right)
    yw: forward distance on ground (meters)
    """

    # Convert FOVs to radians
    fov_x = np.deg2rad(hfov_deg)   # horizontal FOV
    fov_y = np.deg2rad(vfov_deg)    # vertical FOV

    # Compute focal lengths in pixels
    fx = (w / 2) / np.tan(fov_x / 2)
    fy = (h / 2) / np.tan(fov_y / 2)

    # Ground → Camera coordinates (camera pitched down)
    z_cam = yw * np.cos(pitch_angle) + camera_height * np.sin(pitch_angle)
    y_cam = camera_height * np.cos(pitch_angle) - yw * np.sin(pitch_angle)
    x_cam = xw

    # Only project points in front of camera
    # if z_cam <= 0.1:
    #     return None

    # Perspective projection
    u = int(fx * x_cam / z_cam + w / 2)
    v = int(fy * y_cam / z_cam + h / 2)

    # Image bounds check
    # if not (0 <= u < w and 0 <= v < h):
    #     return None

    return (u, v)


def get_10m_mask(h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        # 19ft x 9ft area (~5.8m x 2.7m) or your custom 10m zone
        corners_world = [(-2.0, 2.0), (2.0, 2.0), (2.0, 10.0), (-2.0, 10.0)]
        
        pixel_pts = []
        for xw, yw in corners_world:
            # pt = project_point_with_fl(xw, yw)
            pt = project_point_with_fov_to_img(xw, yw)
            if pt: pixel_pts.append(pt)
            
        if len(pixel_pts) >= 3:
            cv2.fillPoly(mask, [np.array(pixel_pts, dtype=np.int32)], 255)
        return mask, pixel_pts


def get_15m_mask(h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        # 19ft x 9ft area (~5.8m x 2.7m) or your custom 10m zone
        corners_world = [ROI_LB, ROI_RB, ROI_RT, ROI_LT]
        
        pixel_pts = []
        for xw, yw in corners_world:
            # pt = project_point_with_fl(xw, yw)
            pt = project_point_with_fov_to_img(xw, yw)
            if pt: pixel_pts.append(pt)
            
        if len(pixel_pts) >= 3:
            cv2.fillPoly(mask, [np.array(pixel_pts, dtype=np.int32)], 255)
        return mask, pixel_pts




def sutherland_hodgman_clip(polygon, img_w, img_h):
    """
    Clip a polygon to image boundaries (0,0)-(img_w-1, img_h-1)
    polygon: list of [u,v]
    Returns: clipped polygon points
    """
    def clip_edge(polygon, edge):
        clipped = []
        for i in range(len(polygon)):
            curr = polygon[i]
            prev = polygon[i-1]
            
            if edge(curr):
                if not edge(prev):
                    # Intersection
                    inter = line_intersect(prev, curr, edge)
                    if inter is not None:
                        clipped.append(inter)
                clipped.append(curr)
            elif edge(prev):
                inter = line_intersect(prev, curr, edge)
                if inter is not None:
                    clipped.append(inter)
        return clipped

    def line_intersect(p1, p2, edge_func):
        # Find intersection of line segment p1-p2 with boundary defined by edge_func
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return None

        # We iterate over 4 borders
        for border in ['left', 'right', 'top', 'bottom']:
            if border == 'left':
                x = 0
                if dx != 0:
                    t = (x - x1)/dx
                    if 0 <= t <= 1:
                        y = y1 + t*dy
                        if 0 <= y <= img_h-1:
                            return [x, y]
            elif border == 'right':
                x = img_w-1
                if dx != 0:
                    t = (x - x1)/dx
                    if 0 <= t <= 1:
                        y = y1 + t*dy
                        if 0 <= y <= img_h-1:
                            return [x, y]
            elif border == 'top':
                y = 0
                if dy != 0:
                    t = (y - y1)/dy
                    if 0 <= t <= 1:
                        x = x1 + t*dx
                        if 0 <= x <= img_w-1:
                            return [x, y]
            elif border == 'bottom':
                y = img_h-1
                if dy != 0:
                    t = (y - y1)/dy
                    if 0 <= t <= 1:
                        x = x1 + t*dx
                        if 0 <= x <= img_w-1:
                            return [x, y]
        return None

    # Define inside functions for each border
    inside_left   = lambda p: p[0] >= 0
    inside_right  = lambda p: p[0] <= img_w-1
    inside_top    = lambda p: p[1] >= 0
    inside_bottom = lambda p: p[1] <= img_h-1

    clipped = polygon
    for edge in [inside_left, inside_right, inside_top, inside_bottom]:
        clipped = clip_edge(clipped, edge)
        if not clipped:
            break
    return np.array(clipped, dtype=np.int32)

def get_15m_mask_oov(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # World corners of ROI
    corners_world = [ROI_LB, ROI_RB, ROI_RT, ROI_LT]
    
    pixel_pts = []
    for xw, yw in corners_world:
        pt = project_point_with_fov_to_img(xw, yw)
        if pt is not None:
            pixel_pts.append(list(pt))
        else:
            # Keep as is, will be clipped
            pixel_pts.append(list(pt) if pt else [np.nan, np.nan])
    
    # Remove NaNs for clipping algorithm
    pixel_pts = [p for p in pixel_pts if not np.isnan(p[0])]
    
    if len(pixel_pts) >= 3:
        clipped_pts = sutherland_hodgman_clip(pixel_pts, w, h)
        cv2.fillPoly(mask, [clipped_pts], 255)
    else:
        clipped_pts = np.array([], dtype=np.int32)
    
    return mask, clipped_pts




def pixel_to_world(u, v):
    """
    Exact inverse of project_point_with_fov_to_img
    Assumes the point lies on the ground plane
    """

    cx = w / 2
    cy = h / 2

    # Focal lengths
    fov_x = np.deg2rad(hfov_deg)
    fov_y = np.deg2rad(vfov_deg)

    fx = (w / 2) / np.tan(fov_x / 2)
    fy = (h / 2) / np.tan(fov_y / 2)

    # Normalized camera ray
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = 1.0

    r = y_cam / z_cam

    cp = np.cos(pitch_angle)
    sp = np.sin(pitch_angle)

    denom = (r * cp + sp)
    if denom <= 0:
        return None  # above horizon

    # Solve forward distance
    yw = camera_height * (cp - r * sp) / denom

    if yw <= 0:
        return None

    # Recover z_cam scale
    zc = yw * cp + camera_height * sp

    # Lateral distance
    xw = x_cam * zc

    return xw, yw, 0.0



def find_anchor_points(endpoints):
    real_world_coords = {}

    width, height = w, h

    for side in ['left', 'right']:
        if endpoints[side] is not None:
            start_px = endpoints[side]['start']
            end_px = endpoints[side]['end']
            
            # Map Start Point (usually the point closer to the car)
            rw_start = pixel_to_world(start_px[0], start_px[1])
            
            # Map End Point (usually the point further away)
            rw_end = pixel_to_world(end_px[0], end_px[1])
            
            real_world_coords[side] = {
                'start_meters': rw_start,
                'end_meters': rw_end
            }
            # print("---", real_world_coords)
        else:
            real_world_coords[side] = None

    
    return real_world_coords


class LaneTracker:
    def __init__(self, max_history=300):
        # We use deque with maxlen to automatically remove old frames
        self.history = {
            'left_start': deque(maxlen=max_history),
            'left_end':   deque(maxlen=max_history),
            'right_start': deque(maxlen=max_history),
            'right_end':  deque(maxlen=max_history),
            'park_cen':  deque(maxlen=max_history),
            'park_cen_veh':  deque(maxlen=max_history),
            'park_cen_world':  deque(maxlen=max_history),
            'park_heading_world': deque(maxlen=max_history),
            'park_cen_l':  deque(maxlen=max_history),
            'park_cen_veh_l':  deque(maxlen=max_history),
            'park_cen_world_l':  deque(maxlen=max_history),
            'park_heading_world_l': deque(maxlen=max_history),
            'veh_gps':  deque(maxlen=max_history),
            'veh_heading':  deque(maxlen=max_history),
            'r_valid':  deque(maxlen=max_history),
            'l_valid':  deque(maxlen=max_history)
        }

        self.stable_tag = False
        self.start_gps = None
        self.current_gps = None
        self.current_heading = None
        self.trust_range_2v = [0, 0]
        self.target_static = None
        self.delta_heading = 100
        self.pred_heading = 100
        self.delay = 0
        self.curr_timing = None
        self.steer = 0
        self.dis2tar = 100



    # def check_stop():
    #     pass


    def get_static_target(self):

        with open(path_fn, 'r') as f:
            line = f.readline()
        path = json.loads(line)
        target = path[-2]

        self.target_static = target

        return self.target_static


    def update(self, real_world_coords, pc, pc_l, pc2veh, pc2veh_l, pc2world, pc2world_l, gps, heading, ph, ph_l, rv, lv):
        
        self.curr_timing = get_curr_timing()

        self.history['veh_gps'].append(gps)
        self.history['veh_heading'].append(heading)
        
        self.history['park_cen'].append(pc)
        self.history['park_cen_veh'].append(pc2veh)
        self.history['park_cen_world'].append(pc2world)
        self.history['park_heading_world'].append(ph)
        self.history['r_valid'].append(rv)        

        self.history['park_cen_l'].append(pc_l)
        self.history['park_cen_veh_l'].append(pc2veh_l)
        self.history['park_cen_world_l'].append(pc2world_l)
        self.history['park_heading_world_l'].append(ph_l)
        self.history['l_valid'].append(lv)        

        

        for side in ['left', 'right']:
            data = real_world_coords.get(side)
            
            if data and data['start_meters'] and data['end_meters']:
                # Save actual coordinates
                self.history[f'{side}_start'].append(data['start_meters'])
                self.history[f'{side}_end'].append(data['end_meters'])
                
            else:
                # Append None to keep the timeline consistent, 
                # or skip if you only want valid detections.
                self.history[f'{side}_start'].append(None)
                self.history[f'{side}_end'].append(None)

        self.current_gps = gps
        self.current_heading = heading


    def filter_win(self, alpha=0.3):
        """
        Applies exponential smoothing to the world coordinates and heading.
        Formula: s_t = alpha * x_t + (1 - alpha) * s_{t-1}
        
        Args:
            alpha (float): Smoothing factor (0 to 1). 
                           Lower = smoother but more lag.
                           Higher = more responsive but noisier.
        """
        # 1. Get the most recent valid points
        pc_history = [pt for pt in self.history['park_cen_world'] if pt is not None]
        h_history = [h for h in self.history['veh_heading'] if h is not None]

        if len(pc_history) < 2:
            return None, None

        # 2. Initialize smoothed values with the first valid data point
        smooth_pc = np.array(pc_history[0])
        smooth_h = h_history[0]

        # 3. Iteratively apply the EMA filter through the valid history
        for i in range(1, len(pc_history)):
            current_pc = np.array(pc_history[i])
            smooth_pc = alpha * current_pc + (1 - alpha) * smooth_pc
            
        for i in range(1, len(h_history)):
            current_h = h_history[i]
            # Handle heading: simple EMA works if angles don't jump 360->0
            smooth_h = alpha * current_h + (1 - alpha) * smooth_h

        return smooth_pc.tolist(), float(smooth_h)



    def check_consistance(self, window_size=30, std_threshold=0.15):
  
        # 1. Get the most recent N points from history, filtering out None values
        recent_data = list(self.history['park_cen_world'])[-window_size:]
        valid_points = [pt for pt in recent_data if pt is not None]

        # 2. We need a minimum number of valid frames to decide on stability
        if len(valid_points) < (window_size // 2):
            self.stable_tag = False
            return False

        # 3. Calculate Standard Deviation for X (Lateral) and Y (Forward)
        points_array = np.array(valid_points)
        std_x = np.std(points_array[:, 0])
        std_y = np.std(points_array[:, 1])

        print("std. left/right", std_x)

        # 4. Update the stable_tag
        # If the points are staying within a tight circle, it's consistent.
        if std_x < std_threshold and std_y < std_threshold:
            self.stable_tag = True
        else:
            self.stable_tag = False

        return self.stable_tag



    def update_delta_heading(self):
        ## assme stream at 15hz
        read_fre = 30
        delta_h = 100
        delay = 0
        arr = list(self.history['veh_heading'])[-40:]
        if len(arr) < 2:
            pass
        for i in range(len(arr) - 1, 0, -1):
            if arr[i] != arr[i - 1]:
                delta_h = arr[i] - arr[i - 1]
                break
            delay += 1

        # if delta_h<self.delta_heading:
        self.delta_heading = delta_h
        self.delay = delay
            
        ## current reading (prediction only)
        if delta_h<100 and delay>0:
            self.pred_heading = arr[-1]+self.delta_heading*(delay/read_fre) # left turn -/ right turn+
        else: ## delta_h not valid
            self.pred_heading = arr[-1]

        ## load steering
        try:
            with open(steer_fn, 'r') as f:
                line = f.read().strip()
                self.steer = float(data)
        except:
            self.steer = 0




    def get_history_list(self):
        """Returns the history as a standard Python list for saving/JSON"""
        return {key: list(val) for key, val in self.history.items()}

    def get_recent_est(self, v2w_heading, offset, lr, window_size=10):
        
        heading = None
        recent_data_w = list(self.history['park_cen_world'])[-window_size:]
        recent_data_c = list(self.history['park_cen'])[-window_size:]
        recent_data_h = list(self.history['park_heading_world'])[-window_size:]
        recent_vh_data = list(self.history['veh_heading'])[-window_size:] 
        
        if lr=="l":
            recent_data_w = list(self.history['park_cen_world_l'])[-window_size:]
            recent_data_c = list(self.history['park_cen_l'])[-window_size:]
            recent_data_h = list(self.history['park_heading_world_l'])[-window_size:]
            
        val = None
        for i in range(window_size):
            if recent_data_c[-i]:
                val = recent_data_c[-i]
                heading = recent_data_h[-i]
                vh = recent_vh_data[-i]
                break

        if not val:
            return None, None

        world_x, world_y = parking_center_to_vehicle(val[0], val[1], v2w_heading) ## doesn't matter with the sensed heading
        world_x0, world_y0 = target_wrt_initial_point(world_x, world_y, offset)
        pc2world = [world_x0, world_y0]
        heading = heading-vh+v2w_heading ## offset consider
        
        print("track recent cen", recent_data_w, recent_data_c)
        print("last one using stopped sensed heading transfer again", pc2world)

        return pc2world, heading


    def get_recent_est_mul_avg(self, lr, window_size=5):

        pc2world = None
        heading = None
        recent_data_w = list(self.history['park_cen_world'])[-window_size:]
        recent_data_h = list(self.history['park_heading_world'])[-window_size:]
        
        if lr=='l':
            recent_data_w = list(self.history['park_cen_world_l'])[-window_size:]
            recent_data_h = list(self.history['park_heading_world_l'])[-window_size:]
            
        vals = []
        headings = []
        for i in range(window_size):
            if recent_data_w[-i]:
                vals.append(recent_data_w[-i])
                headings.append(recent_data_h[-i]) ## JJJJJ

        if len(vals)==0:
            return None, None

        # print(vals)
        points = np.array([v[:2] for v in vals])

        avg_x = np.mean(points[:, 0])
        avg_y = np.mean(points[:, 1])

        pc2world = [avg_x, avg_y]

        heading = sum(headings)/len(headings)

        print("track recent cen", recent_data_w, "avg", pc2world)
        print("track recent heading", recent_data_h, "avg", heading)
        
        return pc2world, heading


    def get_recent_est_mul_avg_rm_outliers(self, lr, window_size=10):

        # 1. Extract recent data
        recent_data_w = list(self.history['park_cen_world'])[-window_size:]
        recent_data_h = list(self.history['park_heading_world'])[-window_size:]

        if lr=="l":
            recent_data_w = list(self.history['park_cen_world_l'])[-window_size:]
            recent_data_h = list(self.history['park_heading_world_l'])[-window_size:]


        # Filter out None values
        valid_pairs = [(w, h) for w, h in zip(recent_data_w, recent_data_h) if w is not None]
        
        if not valid_pairs:
            return None, None

        # Convert to numpy for vector operations
        points = np.array([p[0][:2] for p in valid_pairs]) # Shape (N, 2)
        headings = np.array([p[1] for p in valid_pairs])   # Shape (N,)
        thr = None

        if len(points) > 2:
            # 2. Identify Outliers for Location (Euclidean Distance from Median)
            median = np.median(points, axis=0)
            distances = np.linalg.norm(points - median, axis=1)
            std_dist = np.std(distances)
            thr = 1 * std_dist
            
            # Keep points within 2 standard deviations of the median distance
            # (Or use a fixed threshold if you know your sensor noise)
            spatial_mask = distances <= (1 * std_dist if std_dist > 0 else 1.0)

            # 3. Identify Outliers for Heading
            h_median = np.median(headings)
            h_diffs = np.abs(headings - h_median)
            h_std = np.std(h_diffs)
            heading_mask = h_diffs <= (1 * h_std if h_std > 0 else 1.0)

            # Combine masks (Optional: only remove if both are outliers, or either)
            final_mask = spatial_mask & heading_mask
            
            # Ensure we don't accidentally remove everything
            if np.any(final_mask):
                filtered_points = points[final_mask]
                filtered_headings = headings[final_mask]
            else:
                filtered_points, filtered_headings = points, headings
        else:
            filtered_points, filtered_headings = points, headings

        # 4. Calculate Final Averages
        avg_x, avg_y = np.mean(filtered_points, axis=0)
        pc2world = [avg_x, avg_y]
        avg_heading = np.mean(filtered_headings)

        ## std filter 1x, 2x, 3x equals to 68%, 95%, 99%
        print(f"Filtered, outliers:", len(points) - len(filtered_points), "by 1x stand dev", thr)
        return pc2world, avg_heading


    def get_recent_est_mul_avg_rm_outliers_dy_buffer(self, lr, window_size=30): # up to 10 valid frame is good enough

        # 1. Extract recent data
        recent_data_w = list(self.history['park_cen_world'])[-window_size:]
        recent_data_h = list(self.history['park_heading_world'])[-window_size:]
        recent_data_valid = list(self.history['r_valid'])[-window_size:]

        if lr=="l":
            recent_data_w = list(self.history['park_cen_world_l'])[-window_size:]
            recent_data_h = list(self.history['park_heading_world_l'])[-window_size:]
            recent_data_valid = list(self.history['l_valid'])[-window_size:]


        # Filter out None values
        valid_pairs = [(w, h) for w, h, v in zip(recent_data_w, recent_data_h, recent_data_valid) if w is not None and v]
        
        if not valid_pairs:
            return None, None
        else:
            print("*** valid reference frames", len(valid_pairs))
            print("sel point", valid_pairs)

        # Convert to numpy for vector operations
        points = np.array([p[0][:2] for p in valid_pairs]) # Shape (N, 2)
        headings = np.array([p[1] for p in valid_pairs])   # Shape (N,)
        thr = None

        if len(points) > 2:
            # 2. Identify Outliers for Location (Euclidean Distance from Median)
            median = np.median(points, axis=0)
            distances = np.linalg.norm(points - median, axis=1)
            std_dist = np.std(distances)
            thr = 1 * std_dist
            
            # Keep points within 2 standard deviations of the median distance
            # (Or use a fixed threshold if you know your sensor noise)
            spatial_mask = distances <= (1 * std_dist if std_dist > 0 else 1.0)

            # 3. Identify Outliers for Heading
            h_median = np.median(headings)
            h_diffs = np.abs(headings - h_median)
            h_std = np.std(h_diffs)
            heading_mask = h_diffs <= (1 * h_std if h_std > 0 else 1.0)

            # Combine masks (Optional: only remove if both are outliers, or either)
            final_mask = spatial_mask & heading_mask
            
            # Ensure we don't accidentally remove everything
            if np.any(final_mask):
                filtered_points = points[final_mask]
                filtered_headings = headings[final_mask]
            else:
                filtered_points, filtered_headings = points, headings
        else:
            filtered_points, filtered_headings = points, headings

        print("filtered est", filtered_points, filtered_headings)
        # 4. Calculate Final Averages
        avg_x, avg_y = np.mean(filtered_points, axis=0)
        pc2world = [avg_x, avg_y]
        avg_heading = np.mean(filtered_headings)

        ## std filter 1x, 2x, 3x equals to 68%, 95%, 99%
        print(f"Filtered, outliers:", len(points) - len(filtered_points), "by 1x stand dev", thr)
        return pc2world, avg_heading


    def get_recent_est_by_both(self, v2w_heading, offset, window_size=10):

        heading = None
        recent_data_w = list(self.history['park_cen_world'])[-window_size:]
        recent_data_c = list(self.history['park_cen'])[-window_size:]

        recent_data_wl = list(self.history['park_cen_world_l'])[-window_size:]
        recent_data_cl = list(self.history['park_cen_l'])[-window_size:]
        
        val1 = None
        for i in range(window_size):
            if recent_data_c[-i]:
                val1 = recent_data_c[-i]
                break

        if not val1:
            return None, None

        val2 = None
        for i in range(window_size):
            if recent_data_cl[-i]:
                val2 = recent_data_cl[-i]
                break

        if not val2:
            return None, None

        val = [(val1[0]+val2[0])/2, (val1[1]+val2[1])/2]

        world_x, world_y = parking_center_to_vehicle(val[0], val[1], v2w_heading) ## doesn't matter with the sensed heading
        world_x0, world_y0 = target_wrt_initial_point(world_x, world_y, offset)
        pc2world = [world_x0, world_y0]
        
        print("track recent cen R+L", recent_data_w, recent_data_c, recent_data_wl, recent_data_cl)
        print("last one using stopped sensed heading transfer again R+L", pc2world)

        return pc2world, heading



    def get_recent_est_by_both_mul_avg(self, window_size=5):

        pc2world = None
        heading = None

        recent_data_w = list(self.history['park_cen_world'])[-window_size:]
        recent_data_wl = list(self.history['park_cen_world_l'])[-window_size:]
        
        vals1 = []
        for i in range(window_size):
            if recent_data_w[-i]:
                vals1.append(recent_data_w[-i])
        
        vals2 = []
        for i in range(window_size):
            if recent_data_wl[-i]:
                vals2.append(recent_data_wl[-i])
            

        if len(vals1)==0 and len(vals2)==0:
            return None, None


        points = np.array(vals1)
        avg_x = np.mean(points[:, 0])
        avg_y = np.mean(points[:, 1])
        val1 = [avg_x, avg_y]

        points2 = np.array(vals2)
        avg_x2 = np.mean(points2[:, 0])
        avg_y2 = np.mean(points2[:, 1])
        val2 = [avg_x2, avg_y2]

        
        val = [(val1[0]+val2[0])/2, (val1[1]+val2[1])/2]

        
        print("track recent cen", recent_data_w, recent_data_wl, "avg", pc2world)
        
        return pc2world, heading




def edge_det_mask_range(frame, hough_thr, blur_thr, mask):

    # h, w = frame.shape[:2]
    # new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
    # undistorted = cv2.undistort(frame, K, D, None, new_K)
    # gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # 1. Standard Gray and Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k_size = blur_thr if blur_thr % 2 != 0 else blur_thr + 1
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)


    # 2. Apply mask to the blurred image 
    # This zeroes out everything outside the 10m zone
    masked_input = cv2.bitwise_and(blurred, mask)

    # 3. Canny Edge Detection
    edges = cv2.Canny(masked_input, 100, 200) # prev 50, 150

    # 4. REMOVE THE MASK BORDER
    # Canny will detect the edge of the mask itself. 
    # We shrink the mask slightly (erode) to "cut off" the fake border edges.
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    final_edges = cv2.bitwise_and(edges, eroded_mask)

    # 5. Line Detection
    lines = cv2.HoughLines(final_edges, 1, np.pi/180, hough_thr)
    
    return final_edges, lines



def edge_det_mask_range_otsu(frame, hough_thr, blur_thr, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k_size = blur_thr if blur_thr % 2 != 0 else blur_thr + 1
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

    # 1. Apply Otsu to get the optimal threshold value
    # Otsu’s calculated threshold that can be used to dynamically set the Canny hysteresis values.
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh

    # 2. Mask the blurred image
    masked_input = cv2.bitwise_and(blurred, mask)

    # 3. Canny using Otsu's values
    modify = 1.2
    # edges = cv2.Canny(masked_input, low_thresh, high_thresh)
    edges = cv2.Canny(masked_input, low_thresh*modify, high_thresh*modify)
    # print(low_thresh*1.4, high_thresh*1.4)
    # exit()

    # 4. Remove Mask Border
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    final_edges = cv2.bitwise_and(edges, eroded_mask)

    # 5. Line Detection
    lines = cv2.HoughLines(final_edges, 1, np.pi/180, hough_thr)
    
    return final_edges, lines

def edge_det_mask_range_otsu_rm_dark(frame, hough_thr, blur_thr, mask):

    # try local contrast enhancement
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # enhanced = clahe.apply(gray)
    # k_size = blur_thr if blur_thr % 2 != 0 else blur_thr + 1
    # blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

    # 1. Grayscale and Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k_size = blur_thr if blur_thr % 2 != 0 else blur_thr + 1
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

    # 2. Use Otsu to create a "Bright Areas" mask
    # This automatically finds the threshold to separate white lines from dark road/shadows
    _, white_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # print(cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    # exit()

    # 3. Combine your positional mask with the Otsu "white" mask
    # This removes both the out-of-range areas AND dark shadows/road
    combined_mask = cv2.bitwise_and(mask, white_mask)

    # 4. Apply combined mask to the blurred image before Canny
    masked_input = cv2.bitwise_and(blurred, combined_mask)

    # 5. Canny Edge Detection
    # Because masked_input is now black in shadow areas, Canny won't find edges there
    edges = cv2.Canny(masked_input, 100, 150)

    # 6. Remove the artificial mask border
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(combined_mask, kernel, iterations=1)
    final_edges = cv2.bitwise_and(edges, eroded_mask)

    # 7. Line Detection
    lines = cv2.HoughLines(final_edges, 1, np.pi/180, hough_thr)
    
    return final_edges, lines

def edge_det_adaptive_thresh(frame, hough_thr, blur_thr, mask):
    # 1. Grayscale and Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k_size = blur_thr if blur_thr % 2 != 0 else blur_thr + 1
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

    # 2. Adaptive Thresholding
    # This evaluates local neighborhoods rather than a global brightness, 
    # making it much more robust against bright glare or harsh shadows.
    # blockSize=15 (local area size), C=3 (constant subtracted from mean)
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, -3)

    # 3. Mask out the unrelated regions
    masked_binary = cv2.bitwise_and(binary, mask)

    # 4. Canny Edge Detection (helps thin the thresholded lines)
    edges = cv2.Canny(masked_binary, 50, 150)

    # 5. Remove the artificial mask border
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    final_edges = cv2.bitwise_and(edges, eroded_mask)

    # 6. Line Detection
    lines = cv2.HoughLines(final_edges, 1, np.pi/180, hough_thr)
    
    return final_edges, lines

def edge_det_mask_range_otsu_rm_dark_dyn_hough(frame, hough_thr, blur_thr, mask):
    
    ## try local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    k_size = blur_thr if blur_thr % 2 != 0 else blur_thr + 1
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

    # 1. Grayscale and Blur
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # k_size = blur_thr if blur_thr % 2 != 0 else blur_thr + 1
    # blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

    # 2. Otsu "White" Mask
    _, white_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined_mask = cv2.bitwise_and(mask, white_mask)
    masked_input = cv2.bitwise_and(blurred, combined_mask)

    # 3. Canny Edge Detection
    edges = cv2.Canny(masked_input, 50, 150)

    # 4. Remove Mask Border
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(combined_mask, kernel, iterations=1)
    final_edges = cv2.bitwise_and(edges, eroded_mask)

    # --- DYNAMIC HOUGH START ---
    h, w = final_edges.shape
    horizon_line = int(h * 0.5) # Adjust this (0.4 = top 40% is "Far")

    # Split the edge image into two zones
    far_zone = final_edges[0:horizon_line, :]
    near_zone = final_edges[horizon_line:h, :]

    # Detect lines in Far Zone (Lower threshold for small/faint lines)
    # Using 60% of hough_thr as a heuristic
    far_lines = cv2.HoughLines(far_zone, 1, np.pi/180, int(hough_thr * 0.5))

    # Detect lines in Near Zone (Standard threshold)
    near_lines = cv2.HoughLines(near_zone, 1, np.pi/180, hough_thr)

    # Adjust the 'rho' (distance) for near_lines because they were detected in a cropped image
    # Rho is relative to the top-left (0,0) of the input image
    if near_lines is not None:
        for line in near_lines:
            rho, theta = line[0]
            # Adjust rho by the vertical offset (y * cos(theta))
            line[0][0] += horizon_line * np.cos(theta)

    # Combine results
    if far_lines is not None and near_lines is not None:
        lines = np.vstack((far_lines, near_lines))
    elif far_lines is not None:
        lines = far_lines
    else:
        lines = near_lines
    # --- DYNAMIC HOUGH END ---
    
    return final_edges, lines

def edge_det_adaptive_thresh_dyn_hough(frame, hough_thr, blur_thr, mask):
    # 1. Grayscale and Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k_size = blur_thr if blur_thr % 2 != 0 else blur_thr + 1
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

    # 2. Adaptive Thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, -3)

    # 3. Mask out the unrelated regions
    masked_binary = cv2.bitwise_and(binary, mask)

    # 4. Canny Edge Detection
    edges = cv2.Canny(masked_binary, 50, 150)

    # 5. Remove the artificial mask border
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    final_edges = cv2.bitwise_and(edges, eroded_mask)

    # --- DYNAMIC HOUGH START ---
    h, w = final_edges.shape
    horizon_line = int(h * 0.5) # Adjust this (0.4 = top 40% is "Far")

    # Split the edge image into two zones
    far_zone = final_edges[0:horizon_line, :]
    near_zone = final_edges[horizon_line:h, :]

    # Detect lines in Far Zone (Lower threshold for small/faint lines)
    far_lines = cv2.HoughLines(far_zone, 1, np.pi/180, int(hough_thr * 0.5))

    # Detect lines in Near Zone (Standard threshold)
    near_lines = cv2.HoughLines(near_zone, 1, np.pi/180, hough_thr)

    # Adjust the 'rho' (distance) for near_lines because they were detected in a cropped image
    if near_lines is not None:
        for line in near_lines:
            rho, theta = line[0]
            line[0][0] += horizon_line * np.cos(theta)

    # Combine results
    if far_lines is not None and near_lines is not None:
        lines = np.vstack((far_lines, near_lines))
    elif far_lines is not None:
        lines = far_lines
    else:
        lines = near_lines
    # --- DYNAMIC HOUGH END ---
    
    return final_edges, lines




def draw_all_hough_lines(frame, lines, color=(255, 255, 255), thickness=1, alpha=0.4):
    """
    Draws every line detected by the Hough transform with transparency.
    """
    if lines is None:
        return frame

    overlay = frame.copy()

    for line in lines:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)

    # Blend overlay with original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame




def add_edge_on_img(frame, edge):
    output = frame.copy()
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_color[:, :, 0] = 0   # remove blue
    edges_color[:, :, 1] = 0   # remove green
    output = cv2.addWeighted(output, 1.0, edges_color, 0.7, 0)
    return output


def add_lanes_on_img(frame, ll, rl, mask, mask_pts):
    """
    mask_pts: pixel corners from draw_distance_grid [BL, BR, TR, TL]
    Returns: (frame, endpoints_dict)
    """
    line_layer = np.zeros_like(frame)
    endpoints = {"left": None, "right": None}
    
    # Define the vertical bounds of the mask
    y_bottom = np.max(mask_pts[:, 1]) # Bottom of image/mask
    y_top = np.min(mask_pts[:, 1])    # Top of mask (10m away)

    for i, (best_line, color) in enumerate([(ll, (0, 255, 0)), (rl, (255, 0, 0))]):
        label = "left" if i == 0 else "right"
        
        if best_line:
            rho, theta = best_line
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            # Avoid division by zero for horizontal-ish lines
            if abs(cos_t) < 0.001: continue 

            # Calculate intersection points at the mask's Y-boundaries
            # Formula: x = (rho - y * sin(theta)) / cos(theta)
            x_bottom = int((rho - y_bottom * sin_t) / cos_t)
            x_top = int((rho - y_top * sin_t) / cos_t)
            
            p1 = (x_bottom, y_bottom)
            p2 = (x_top, y_top)
            
            # Store endpoints
            endpoints[label] = {"start": p1, "end": p2}
            
            # Draw the line on the overlay
            cv2.line(line_layer, p1, p2, color, 8, cv2.LINE_AA)

    # Apply the mask and blend
    masked_lines = cv2.bitwise_and(line_layer, line_layer, mask=mask)
    cv2.addWeighted(frame, 1.0, masked_lines, 1.0, 0, frame)

    return frame, endpoints


# find the accurate end point within mask
def add_lanes_on_img_with_endpoints(frame, ll, rl, mask, mask_pts, edge_map, conn_thr=1, min_len_thr=50):
    """
    edge_map: The binary image from edge_det_mask_range (final_edges)
    """
    line_layer = np.zeros_like(frame)
    endpoints = {"left": None, "right": None}
    
    y_bottom = np.max(mask_pts[:, 1])
    y_top = np.min(mask_pts[:, 1])

    for i, (best_line, color) in enumerate([(ll, (0, 255, 0)), (rl, (255, 0, 0))]):
        label = "left" if i == 0 else "right"
        if not best_line: continue
        
        rho, theta = best_line
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        if abs(cos_t) < 0.001: continue 

        # 1. Sample along the line to find where edge pixels actually are
        # We walk from bottom to top
        active_points = []
        for y in range(int(y_bottom), int(y_top), -2): # step -2 for speed
            x = int((rho - y * sin_t) / cos_t)
            
            # Bound check for image width
            if 0 <= x < edge_map.shape[1]:
                # 2. Search window: check if there's an edge pixel nearby (±5px)
                try:
                    window = edge_map[y, max(0, x-conn_thr):min(edge_map.shape[1], x+conn_thr)]
                    if np.any(window > 0):
                        active_points.append((x, y))
                except:
                    pass
            
        # print(active_points)
        # 3. Filter for the longest continuous segment
        # If we found enough points, define the new start/end
        if len(active_points) > min_len_thr:
            # Finding the "extremes" of the detected edge clusters
            p_start = active_points[0]  # Closest to bottom
            p_end = active_points[-1]   # Furthest away

            rw_start = pixel_to_world(p_start[0], p_start[1])
            rw_end = pixel_to_world(p_end[0], p_end[1])
            if label=='right':
                global rbp_valid, rtp_valid 
                if rw_start[1]>valid_depth[0]+0.2: # 0.2m above ROI bottom (closer more trustful)
                    rbp_valid = True
                    # print("** btm point in view", rw_start, ROI_RB)
                else:
                    rbp_valid = False

                if rw_end[1]<valid_depth[1]-5: # 2m below ROI top
                    rtp_valid = True
                    # print("** top point in view", rw_end, ROI_RT)
                else:
                    rtp_valid = False

            if label=='left':
                global lbp_valid, ltp_valid 
                if rw_start[1]>valid_depth[0]+0.2: # 0.2m above ROI bottom (closer more trustful)
                    lbp_valid = True
                    # print("** btm point in view", rw_start, ROI_RB)
                else:
                    lbp_valid = False

                if rw_end[1]<valid_depth[1]-5: # 2m below ROI top
                    ltp_valid = True
                    # print("** top point in view", rw_end, ROI_RT)
                else:
                    ltp_valid = False
                
            endpoints[label] = {"start": p_start, "end": p_end}
            
            # 4. Draw only the segment where edges were actually found
            cv2.line(line_layer, p_start, p_end, color, 4, cv2.LINE_AA)

    if not endpoints["right"]:
        rtp_valid = False
        rbp_valid = False
    if not endpoints["left"]:
        ltp_valid = False
        lbp_valid = False


    # Blend with frame
    masked_lines = cv2.bitwise_and(line_layer, line_layer, mask=mask)
    cv2.addWeighted(frame, 1.0, masked_lines, 1.0, 0, frame)

    return frame, endpoints

# def get_validated_lane_endpoints(edge_map, ll, rl, mask_pts, conn_thr=2, min_len_thr=50):
#     """
#     Analyzes edge map to find start and end points of lanes.
#     Returns a dictionary of validated endpoints.
#     """
#     endpoints = {"left": None, "right": None}
#     validity_flags = {
#         "left": {"bottom": False, "top": False},
#         "right": {"bottom": False, "top": False}
#     }
    
#     y_bottom = min(int(np.max(mask_pts[:, 1])), edge_map.shape[0] - 1)
#     y_top = max(int(np.min(mask_pts[:, 1])), 0)

#     for i, best_line in enumerate([ll, rl]):
#         label = "left" if i == 0 else "right"
#         if not best_line: continue
        
#         rho, theta = best_line
#         cos_t, sin_t = np.cos(theta), np.sin(theta)
#         if abs(cos_t) < 0.001: continue 

#         active_points = []
#         # Bottom-to-top scan
#         for y in range(y_bottom, y_top, -2):
#             x = int((rho - y * sin_t) / cos_t)
#             if 0 <= x < edge_map.shape[1]:
#                 x_min, x_max = max(0, x - conn_thr), min(edge_map.shape[1], x + conn_thr)
#                 if np.any(edge_map[y, x_min:x_max] > 0):
#                     # Density check
#                     check_y = y - 4
#                     check_x = int((rho - check_y * sin_t) / cos_t)
#                     if 0 <= check_x < edge_map.shape[1]:
#                         check_win = edge_map[check_y, max(0, check_x-conn_thr):min(edge_map.shape[1], check_x+conn_thr)]
#                         if np.any(check_win > 0) or len(active_points) > 0:
#                             active_points.append((x, y))

#         if len(active_points) > min_len_thr:
#             p_start = active_points[2]  
#             p_end = active_points[-3]   
            
#             # World Coordinate Validation
#             rw_start = pixel_to_world(p_start[0], p_start[1])
#             rw_end = pixel_to_world(p_end[0], p_end[1])
            
#             # Logic: y-axis in world usually depth. valid_depth defined globally/outer scope
#             validity_flags[label]["bottom"] = rw_start[1] > (valid_depth[0] + 0.2)
#             validity_flags[label]["top"] = rw_end[1] < (valid_depth[1] - 5)
            
#             endpoints[label] = {"start": p_start, "end": p_end}

#     return endpoints, validity_flags

def find_lanes_on_img_with_endpoints_seg_check(frame, ll, rl, mask, mask_pts, edge_map, conn_thr=2, min_len_thr=50):
    """
    Improved lane segment detection using a density check to prevent 
    endpoints from jumping to noise at the ROI boundaries.
    """
    # line_layer = np.zeros_like(frame)
    endpoints = {"left": None, "right": None}
    
    # Define vertical bounds from the mask
    y_bottom = int(np.max(mask_pts[:, 1]))
    y_top = int(np.min(mask_pts[:, 1]))

    for i, best_line in enumerate([ll, rl]):
        label = "left" if i == 0 else "right"
        if not best_line: continue
        
        rho, theta = best_line
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        if abs(cos_t) < 0.001: continue 

        active_points = []
        # We search from bottom to top to find the physical start of the line
        # Step -2 for performance; search window helps bridge gaps

        h, w = edge_map.shape[:2]

        y_bottom = min(y_bottom, h - 1)
        y_top = max(y_top, 0)

        for y in range(y_bottom, y_top, -2):
            x = int((rho - y * sin_t) / cos_t)
            
            if 0 <= x < edge_map.shape[1]:
                # Define a localized window on the edge map
                x_min = max(0, x - conn_thr)
                x_max = min(edge_map.shape[1], x + conn_thr)
                window = edge_map[y, x_min:x_max]
                
                if np.any(window > 0):
                    # --- NOISE FILTERING ---
                    # Only accept points if there are other edges nearby vertically
                    # This prevents a single noise speck from being the start point
                    check_y = y - 4 # Look slightly ahead
                    check_x = int((rho - check_y * sin_t) / cos_t)
                    
                    if 0 <= check_x < edge_map.shape[1]:
                        check_win = edge_map[check_y, max(0, check_x-conn_thr):min(edge_map.shape[1], check_x+conn_thr)]
                        if np.any(check_win > 0) or len(active_points) > 0:
                            active_points.append((x, y))

        # Check if the detected segment meets the minimum length requirement
        if len(active_points) > min_len_thr:
            # 1. Use small internal offsets to avoid boundary noise
            # Instead of the absolute first/last, take the 3rd and 3rd-to-last
            p_start = active_points[2]  
            p_end = active_points[-3]   

            # 2. World Coordinate Validation (Consistency Checks)
            rw_start = pixel_to_world(p_start[0], p_start[1])
            rw_end = pixel_to_world(p_end[0], p_end[1])
            
            if label=='right':
                global rbp_valid, rtp_valid 
                if rw_start[1]>valid_depth[0]: # 0.2m above ROI bottom (closer more trustful)
                    rbp_valid = True
                    # print("** btm point in view", rw_start, ROI_RB)
                else:
                    rbp_valid = False

                if rw_end[1]<valid_depth[1]: # 2m below ROI top
                    rtp_valid = True
                    # print("** top point in view", rw_end, ROI_RT)
                else:
                    rtp_valid = False

            if label=='left':
                global lbp_valid, ltp_valid 
                if rw_start[1]>valid_depth[0]: # 0.2m above ROI bottom (closer more trustful)
                    lbp_valid = True
                    # print("** btm point in view", rw_start, ROI_RB)
                else:
                    lbp_valid = False

                if rw_end[1]<valid_depth[1]: # 2m below ROI top
                    ltp_valid = True
                    # print("** top point in view", rw_end, ROI_RT)
                else:
                    ltp_valid = False
            
            endpoints[label] = {"start": p_start, "end": p_end}
            
            # 3. Draw the validated segment
            # cv2.line(line_layer, p_start, p_end, color, 8, cv2.LINE_AA)

    # Fallback for missing right line
    if not endpoints["right"]:
        rtp_valid = False
        rbp_valid = False
    if not endpoints["left"]:
        ltp_valid = False
        lbp_valid = False

    # Blend the layer with the original frame
    # masked_lines = cv2.bitwise_and(line_layer, line_layer, mask=mask)
    # cv2.addWeighted(frame, 1.0, masked_lines, 1.0, 0, frame)

    return endpoints


def draw_lane_lines(frame, endpoints, mask):
    """
    Blends the validated lane segments onto the original frame.
    """
    line_layer = np.zeros_like(frame)
    colors = {"left": (0, 255, 0), "right": (255, 0, 0)}

    # Update Global Flags (if needed for your system)
    global lbp_valid, ltp_valid, rbp_valid, rtp_valid
    
    for side in ["left", "right"]:
        data = endpoints[side]
        # Draw line if endpoints exist
        if data:
            cv2.line(line_layer, data["start"], data["end"], colors[side], 8, cv2.LINE_AA)

    # Blend layer
    masked_lines = cv2.bitwise_and(line_layer, line_layer, mask=mask)
    cv2.addWeighted(frame, 1.0, masked_lines, 1.0, 0, frame)
    
    return frame


def put_text_bg(img, text, org, font, font_scale, text_color, bg_color, thickness=2, padding=5):
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = org
    cv2.rectangle(img, (x - padding, y - h - padding), (x + w + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def draw_decision_tag(frame, decision, fr):
    
    text = "Path update by camera "+fr if decision else "Path fr phone"
    
    color = (0, 255, 0) if decision else (0, 0, 255)  # green / red
    
    cv2.putText(
        frame,
        text,
        (20, 30),                 # position (x,y) -> top-left
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,                      # font scale
        color,
        2,                        # thickness
        cv2.LINE_AA
    )

    return frame


def find_LR2(lines, width, height, angle_thr_deg):
    close_left = None
    close_right = None
    # img_center = width//2
    img_center = int((width)/2)
    eval_y = height//2
    max_x_left = -float('inf')
    min_x_right = float('inf')

    lower_b = np.deg2rad(angle_thr_deg)
    upper_b = np.pi-np.deg2rad(angle_thr_deg)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # vertical is 0, horizental is pi/2 (1.57), dismiss 
            if lower_b < theta < upper_b:
                continue

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            if abs(cos_t)<0.01: continue # avoid 0
            x_at_eval_y = (rho-eval_y*sin_t)/cos_t

            if x_at_eval_y<img_center:
                if x_at_eval_y>max_x_left:
                    max_x_left = x_at_eval_y
                    close_left = (rho, theta)
            else:
                if x_at_eval_y<min_x_right:
                    min_x_right = x_at_eval_y
                    close_right = (rho, theta)

    # middle of two line
    mid_p = None
    error_pixel = 1000
    err_heading = 1000
    if close_left and close_right:
        l_rho, l_theta = close_left
        r_rho, r_theta = close_right
        x_left = (l_rho-eval_y*np.sin(l_theta))/np.cos(l_theta)
        x_right = (r_rho-eval_y*np.sin(r_theta))/np.cos(r_theta)
        mid_p = (x_left+x_right)/2

        error_pixel = mid_p-img_center
        
        deg_l = norm_theta(l_theta)
        deg_r = norm_theta(r_theta)
        err_heading = (deg_l+deg_r)/2 # vertical is zero, otherwise, in radians

    # if close_right:
    #     print("++ LR close", close_right)
    
    return close_left, close_right, mid_p, error_pixel, err_heading


def find_LR2_target(lines, width, height, angle_thr_deg, target_center_x=None):
    close_left = None
    close_right = None
    # img_center = width//2
    target_x = target_center_x if target_center_x is not None else int((width)/2)
    eval_y = height//2
    max_x_left = -float('inf')
    min_x_right = float('inf')

    lower_b = np.deg2rad(angle_thr_deg)
    upper_b = np.pi-np.deg2rad(angle_thr_deg)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # vertical is 0, horizental is pi/2 (1.57), dismiss 
            if lower_b < theta < upper_b:
                continue

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            if abs(cos_t)<0.01: continue # avoid 0
            x_at_eval_y = (rho-eval_y*sin_t)/cos_t

            if x_at_eval_y<target_x:
                if x_at_eval_y>max_x_left:
                    max_x_left = x_at_eval_y
                    close_left = (rho, theta)
            else:
                if x_at_eval_y<min_x_right:
                    min_x_right = x_at_eval_y
                    close_right = (rho, theta)

    # middle of two line
    mid_p = None
    error_pixel = 1000
    err_heading = 1000
    if close_left and close_right:
        l_rho, l_theta = close_left
        r_rho, r_theta = close_right
        x_left = (l_rho-eval_y*np.sin(l_theta))/np.cos(l_theta)
        x_right = (r_rho-eval_y*np.sin(r_theta))/np.cos(r_theta)
        mid_p = (x_left+x_right)/2

        error_pixel = mid_p-int(width/2)
        
        deg_l = norm_theta(l_theta)
        deg_r = norm_theta(r_theta)
        err_heading = (deg_l+deg_r)/2 # vertical is zero, otherwise, in radians

    # if close_right:
    #     print("++ LR close", close_right)
    
    return close_left, close_right, mid_p, error_pixel, err_heading


def find_PL(lines, width, height, angle_thr_deg):
    """
    Finds the two lines closest to the image center at the horizontal midline.
    """
    close_right = None
    
    eval_y = height / 2  # The horizontal line where we check for intersection
    
    # Initialize trackers for the closest X-coordinates to the center
    min_x_right = float('inf')   # Closest to center from the right (smallest X > center)

    # Thresholds to ignore horizontal-ish lines (around pi/2 or 90 degrees)
    lower_b = np.deg2rad(angle_thr_deg)
    upper_b = np.pi - np.deg2rad(angle_thr_deg)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]

            # 1. Filter out horizontal lines that don't represent lane boundaries
            # In Hough space, vertical is 0, horizontal is pi/2.
            if lower_b < theta < upper_b:
                continue

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # 2. Avoid division by zero for perfectly horizontal lines
            if abs(cos_t) < 1e-6: 
                continue 

            # 3. Calculate X intersection at the middle of the image height
            # Formula derived from: rho = x*cos(theta) + y*sin(theta)
            x_at_eval_y = (rho - eval_y * sin_t) / cos_t

            # 4. Identify lines closest to the center point
            if x_at_eval_y < min_x_right:
                min_x_right = x_at_eval_y
                close_right = (rho, theta)
    
    if close_right:
        print("+++ PL close", close_right)
    
    return close_right


def norm_theta(theta):
    deg = np.rad2deg(theta)
    if deg>90:
        deg -= 180
    return deg






    


def draw_distance_labels(frame, lane_ep, anchor_points_rw):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Slightly smaller for stacked lines
    thickness = 2
    line_spacing = 20 # Pixels between the two lines of text
    dot_radius = 10   # As requested
    vertical_margin = 15 # Space between the top of the dot and the bottom of the text

    for side in ['left', 'right']:
        if lane_ep[side] and anchor_points_rw[side]:
            if not rbp_valid and side=="right":
                continue
            elif not lbp_valid and side=="left":
                continue

            for point_type in ['start', 'end']:
                pixel_pos = lane_ep[side][point_type]
                rw_key = f"{point_type}_meters"
                rw_val = anchor_points_rw[side][rw_key]

                # 1. Create two separate lines of text
                line1 = f"X: {rw_val[0]:.2f}m"
                line2 = f"Z: {rw_val[1]:.1f}m"

                # 2. Calculate position to be ABOVE the dot
                # Shift X to center the text relative to the dot
                text_x = pixel_pos[0] - 60 
                # Shift Y up: dot_radius + margin + total height of both lines
                text_y_base = pixel_pos[1] - dot_radius - vertical_margin - line_spacing

                color = (255, 0, 255) if side == 'left' else (255, 0, 0)

                # 3. Draw Line 1 (X distance)
                # Shadow
                cv2.putText(frame, line1, (text_x, text_y_base), font, font_scale, 
                            (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # Color
                cv2.putText(frame, line1, (text_x, text_y_base), font, font_scale, 
                            color, thickness, cv2.LINE_AA)

                # 4. Draw Line 2 (Y distance)
                # Shadow
                cv2.putText(frame, line2, (text_x, text_y_base + line_spacing), font, font_scale, 
                            (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # Color
                cv2.putText(frame, line2, (text_x, text_y_base + line_spacing), font, font_scale, 
                            color, thickness, cv2.LINE_AA)

                # 5. Draw the Big Anchor Point
                # Black outer ring for contrast
                cv2.circle(frame, pixel_pos, dot_radius, (0, 0, 0), -1, cv2.LINE_AA)
                # Main 
                # if point_type=="start":
                    # cv2.circle(frame, pixel_pos, dot_radius, (255, 0, 0), -1, cv2.LINE_AA)
                # elif point_type=="end":
                cv2.circle(frame, pixel_pos, dot_radius, (0, 0, 0), -1, cv2.LINE_AA)
                # Small colored center dot for side identification
                cv2.circle(frame, pixel_pos, 4, color, -1, cv2.LINE_AA)

    return frame


def rotate_point(x, y, theta):
    """Rotate point by theta (radians)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c*x - s*y, s*x + c*y])


def estimate_parking_center_from_start(rw_start, heading_deg):

    x_s, y_s, _ = rw_start

    # Local offset to parking center
    dx_local = -spot_width / 2
    dy_local =  spot_length / 2

    # Rotate by -heading (world → camera)
    theta = -np.deg2rad(heading_deg)
    c, s = np.cos(theta), np.sin(theta)

    dx = c * dx_local - s * dy_local
    dy = s * dx_local + c * dy_local

    # Translate to camera-ground frame
    x_center = x_s + dx
    y_center = y_s + dy

    return x_center, y_center


def estimate_parking_center_from_start_l(rw_start, heading_deg):

    x_s, y_s, _ = rw_start

    # Local offset to parking center
    dx_local = spot_width / 2
    dy_local =  spot_length / 2

    # Rotate by -heading (world → camera)
    theta = -np.deg2rad(heading_deg)
    c, s = np.cos(theta), np.sin(theta)

    dx = c * dx_local - s * dy_local
    dy = s * dx_local + c * dy_local

    # Translate to camera-ground frame
    x_center = x_s + dx
    y_center = y_s + dy

    return x_center, y_center



def estimate_parking_center_from_end_l(rw_end, heading_deg):

    x_s, y_s, _ = rw_end

    # Local offset to parking center
    dx_local = spot_width / 2
    dy_local = -lane_tape_right+spot_length / 2

    # Rotate by -heading (world → camera)
    theta = -np.deg2rad(heading_deg)
    c, s = np.cos(theta), np.sin(theta)

    dx = c * dx_local - s * dy_local
    dy = s * dx_local + c * dy_local

    # Translate to camera-ground frame
    x_center = x_s + dx
    y_center = y_s + dy

    return x_center, y_center



def estimate_parking_area_from_start(rw_start, heading_deg):
    """
    rw_start: (x, y) right-lane start point in camera-ground frame (meters)
    returns: (x, y, z) parking center relative to camera
    """

    x0, y0, _ = rw_start

    local_pts = np.array([
        [-spot_width, 0],              # p1: left start
        [0, 0],                        # p2: right start
        [0, spot_length],              # p3: right end
        [-spot_width, spot_length]     # p4: left end
    ])

    # Rotate by -heading (world → camera frame)
    theta = -np.deg2rad(heading_deg)

    rotated_pts = np.array([
        rotate_point(px, py, theta)
        for px, py in local_pts
    ])

    # Translate to rw_start
    world_pts = rotated_pts + np.array([x0, y0])

    p1, p2, p3, p4 = world_pts.tolist()
    return p1, p2, p3, p4



def estimate_parking_area_from_start_l(rw_start, heading_deg):
    """
    rw_start: (x, y) right-lane start point in camera-ground frame (meters)
    returns: (x, y, z) parking center relative to camera
    """

    x0, y0, _ = rw_start

    local_pts = np.array([
        [0, 0],     # p1: left start
        [spot_width, 0],     # p2: right start
        [spot_width, spot_length],     # p3: right end
        [0, spot_length]     # p4: left end
    ])

    # Rotate by -heading (world → camera frame)
    theta = -np.deg2rad(heading_deg)

    rotated_pts = np.array([
        rotate_point(px, py, theta)
        for px, py in local_pts
    ])

    # Translate to rw_start
    world_pts = rotated_pts + np.array([x0, y0])

    p1, p2, p3, p4 = world_pts.tolist()
    return p1, p2, p3, p4

def estimate_parking_area_from_end_l(rw_end, heading_deg):
    """
    rw_start: (x, y) right-lane start point in camera-ground frame (meters)
    returns: (x, y, z) parking center relative to camera
    """

    x0, y0, _ = rw_end

    local_pts = np.array([
        [0, -spot_length+lane_tape_right],     # p1: left start
        [spot_width, -spot_length+lane_tape_right],     # p2: right start
        [spot_width, spot_length-lane_tape_right],     # p3: right end
        [0, spot_length-lane_tape_right]     # p4: left end
    ])

    # Rotate by -heading (world → camera frame)
    theta = -np.deg2rad(heading_deg)

    rotated_pts = np.array([
        rotate_point(px, py, theta)
        for px, py in local_pts
    ])

    # Translate to rw_start
    world_pts = rotated_pts + np.array([x0, y0])

    p1, p2, p3, p4 = world_pts.tolist()
    return p1, p2, p3, p4



def estimate_parking_center_from_end(rw_end, heading_deg):
    """
    rw_start: (x, y) right-lane start point in camera-ground frame (meters)
    returns: (x, y, z) parking center relative to camera
    """
    # x_s, y_s, _ = rw_end

    # x_center = x_s - spot_width / 2
    # y_center = y_s - spot_length / 2

    # return x_center, y_center, 0.0

    x_s, y_s, _ = rw_end

    # Local offset to parking center
    dx_local = -spot_width / 2 
    dy_local =  -lane_tape_right+spot_length / 2

    # Rotate by -heading (world → camera)
    theta = -np.deg2rad(heading_deg)
    c, s = np.cos(theta), np.sin(theta)

    dx = c * dx_local - s * dy_local
    dy = s * dx_local + c * dy_local

    # Translate to camera-ground frame
    x_center = x_s + dx
    y_center = y_s + dy

    return x_center, y_center



def estimate_parking_area_from_end(rw_end, heading_deg):
    """
    rw_start: (x, y) right-lane start point in camera-ground frame (meters)
    returns: (x, y, z) parking center relative to camera
    """

    x0, y0, _ = rw_end

    local_pts = np.array([
        [-spot_width, -spot_length+lane_tape_right],              # p1: left start
        [0, -spot_length+lane_tape_right],                        # p2: right start
        [0, spot_length-lane_tape_right],              # p3: right end (anchor)
        [-spot_width, spot_length-lane_tape_right]     # p4: left end
    ])



    # Rotate by -heading (world → camera frame)
    theta = -np.deg2rad(heading_deg)

    rotated_pts = np.array([
        rotate_point(px, py, theta)
        for px, py in local_pts
    ])

    # Translate to rw_start
    world_pts = rotated_pts + np.array([x0, y0])

    p1, p2, p3, p4 = world_pts.tolist()
    return p1, p2, p3, p4





def draw_parking_center(frame, center_px, rx, ry, color=(255, 0, 0), by='left'):
    if center_px is None:
        return frame

    u,v = center_px

    # Draw center point
    cv2.circle(frame, (u, v), radius=10, color=color, thickness=-1)

    # Label
    cv2.putText(
        frame,
        # "sensed by %s (%.2f,%.2f)"%(by,rx,ry),
        "Cen (%.2f,%.2f)"%(rx,ry),
        (u + 8, v - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

    return frame



def draw_parking_area(frame, points, color=(255, 0, 0), alpha=0.3):
    if points is None or len(points) < 3: # Need at least 3 points for a poly
        return frame

    h, w = frame.shape[:2]
    
    # 1. Convert to numpy array 
    pts = np.array(points, dtype=np.int32)

    print("\n&&& parking area",pts, "\n")
    if pts[0][0]>5000: # fly out
        return frame
       

    # 2. Check if the polygon is entirely outside the view
    # If all x < 0 or all x > w, etc., we can skip drawing to save processing
    if (np.all(pts[:, 0] < 0) or np.all(pts[:, 0] > w) or 
        np.all(pts[:, 1] < 0) or np.all(pts[:, 1] > h)):
        return frame

    # 3. Create a mask for the semi-transparent overlay
    # This is often safer than copying the whole frame if the frame is 4K
    overlay = frame.copy()
    
    # Reshape for OpenCV requirements
    pts_reshaped = pts.reshape((-1, 1, 2))

    # Draw the filled area on the overlay
    cv2.fillPoly(overlay, [pts_reshaped], color)
    
    # Apply transparency
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 4. Draw the outline
    # cv2.polylines handles points outside the image gracefully by clipping 
    # the lines at the edge of the canvas.
    cv2.polylines(frame, [pts_reshaped], isClosed=True, color=color, thickness=2)

    return frame

def clip_polygon_to_image(pts, w, h):
    # Image rectangle
    rect = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    pts = pts.astype(np.float32)
    clipped = cv2.intersectConvexConvex(pts, rect)[1]

    if clipped is None:
        return None

    return clipped.astype(np.int32)

def draw_parking_center2(frame, center_px):
    if center_px is None:
        return frame

    u,v = center_px
    color=(0, 0, 0)

    # Draw center point
    cv2.circle(frame, (u, v), radius=10, color=color, thickness=-1)

    # Label
    cv2.putText(
        frame,
        "Cam sensed target by far right point",
        (u + 8, v - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

    return frame


def draw_ref_indicator(frame, ind1, ind2, ind3, ind4):
    # if not ind1 and not ind2 and not ind3 and not ind4:
        # return frame

    color=(0, 0, 255)

    if ind1:
        cv2.putText(
            frame,
            "R-C ref valid",
            (w - 120, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    if ind2:
        cv2.putText(
            frame,
            "R-F ref valid",
            (w - 120, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    if ind3:
        cv2.putText(
            frame,
            "L-C ref valid",
            (w - 240, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    if ind4:
        cv2.putText(
            frame,
            "L-F ref valid",
            (w - 240, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    return frame


# spot in vehicle-center world coord: -0.27687143885717225 4.849720453516838 -0.27687143885717225 7.349220453516837

def parking_center_to_vehicle(x, y, veh_heading):
    """
    Input:
        (x, y): parking center relative to camera (meters)
                x = right, y = forward

    Output:
        (E, N): parking center relative to vehicle center in world frame
    """

    # Step 1: Camera → Vehicle-center (translation)
    x_v = x
    y_v = y + vehicle_length / 2

    # Step 2: Vehicle → World (rotation by heading)
    h = np.deg2rad(veh_heading)

    E =  np.sin(h) * y_v + np.cos(h) * x_v
    N =  np.cos(h) * y_v - np.sin(h) * x_v

    return E, N


def rotate_world(x, y, veh_heading):
  
    # Step 1: Camera → Vehicle-center (translation)
    x_v = x
    y_v = y + vehicle_length / 2

    # Step 2: Vehicle → World (rotation by heading)
    h = np.deg2rad(veh_heading)

    E =  np.sin(h) * y_v + np.cos(h) * x_v
    N =  np.cos(h) * y_v - np.sin(h) * x_v

    return E, N

def target_wrt_initial_point(E, N, vehicle_pos):
    Ev, Nv = vehicle_pos
    return Ev + E, Nv + N






def gps_to_meter(gps0, gps1):
    """
    Translates gps1 relative to gps0.
    gps format: [latitude, longitude]
    Returns: [x_meters (East), y_meters (North)]
    """
    R = 6378137.0 # Earth radius in meters
    
    lat0, lon0 = np.radians(gps0[0]), np.radians(gps0[1])
    lat1, lon1 = np.radians(gps1[0]), np.radians(gps1[1])
    
    # Calculate differences
    d_lat = lat1 - lat0
    d_lon = lon1 - lon0
    
    # North (Y) displacement
    y = d_lat * R
    
    # East (X) displacement
    # We use the average latitude to compensate for longitude convergence
    x = d_lon * R * np.cos((lat0 + lat1) / 2)
    
    return [x, y]


def estimate_v2p_heading_by_sense_line(p1, p2):
    """
    p1, p2: (x, y) tuples or arrays
    Returns heading in degrees where:
    North/Forward (+Y) = 0°
    East/Right (+X) = 90°
    South/Backward (-Y) = 180°
    West/Left (-X) = -90° (or 270°)
    """
    dx = p2[0] - p1[0] # Right/Left
    dy = p2[1] - p1[1] # Forward/Backward

    # In standard math: atan2(y, x) -> 0 is +X
    # For Navigation (0 is +Y): use atan2(x, y)
    orientation_rad = np.arctan2(dx, dy)

    # Convert to degrees
    orientation_deg = np.degrees(orientation_rad)
    
    return orientation_deg



## compare two estimation
def comp_lr_loc_est(park_center_r, park_cen_l):
    
    def extract_xy(data):
        xs, ys = [], []
        for p in data:
            if p is not None:
                xs.append(p[0])
                ys.append(p[1])
            else:
                xs.append(np.nan)
                ys.append(np.nan)
        return np.array(xs), np.array(ys)

    # Extract coordinates
    x1, y1 = extract_xy(park_center_r)
    x2, y2 = extract_xy(park_center_l)

    plt.figure(figsize=(8, 8))

    plt.plot(x1, y1, 'r.-', label='park_center_by_right')
    plt.plot(x2, y2, 'b.-', label='park_center_by_left')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Park Center est R vs L')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    plt.show()


def comp_lr_h_est(park_h_r, park_h_l):
    
    def extract_h(data):
        hs = []
        for p in data:
            if p is not None:
                if p>180:
                    p -= 360
                hs.append(p)
            else:
                hs.append(np.nan)
                
        return np.array(hs)

    # Extract coordinates
    h1 = extract_h(park_h_r)
    h2 = extract_h(park_h_l)

    plt.figure(figsize=(8, 8))

    plt.plot(h1, 'r.-', label='park_head_by_right')
    plt.plot(h2, 'b.-', label='park_head_by_left')

    plt.xlabel('T')
    plt.ylabel('Head')
    plt.title('Park heading est R vs L')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    plt.show()


def draw_middle_line_pixels(frame, left_lane_px, right_lane_px):
    """
    Draw a virtual middle line based purely on pixel coordinates.

    left_lane_px, right_lane_px: dict with keys 'start' and 'end', each as (x, y) pixel coordinates
    """
    
    if left_lane_px is None or right_lane_px is None:
        print("Cannot draw middle line; one lane is missing.")
        return frame

    # Compute middle points at start and end
    middle_start = (
        int((left_lane_px['start'][0] + right_lane_px['start'][0]) / 2),
        int((left_lane_px['start'][1] + right_lane_px['start'][1]) / 2)
    )
    middle_end = (
        int((left_lane_px['end'][0] + right_lane_px['end'][0]) / 2),
        int((left_lane_px['end'][1] + right_lane_px['end'][1]) / 2)
    )

    # Draw middle line (green, thickness=2)
    frame_with_line = frame.copy()
    cv2.line(frame_with_line, middle_start, middle_end, (0, 255, 0), 2)

    return frame_with_line



def load_veh_status():
    ## load real heading
    with open(can_h_fn, 'r') as f:
        line = f.read().strip()
        # Remove brackets and split by comma
        data = line.replace('[', '').replace(']', '').split(',')
        # h is the third element (index 2)
        v2w_heading = float(data[2])

    ## load ref gps
    with open(gps_ref_fn, 'r') as f:
        line = f.read().strip()
        # Remove brackets and split by comma
        data = line.replace('[', '').replace(']', '').split(',')
        # h is the third element (index 2)
        gps0 = [float(data[0]), float(data[1])]


    ## load realtime gps
    with open(gps_fn, 'r') as f:
        line = f.read().strip()
        # Remove brackets and split by comma
        data = line.replace('[', '').replace(']', '').split(',')
        # h is the third element (index 2)
        current_gps = [float(data[0]), float(data[1])]

    return v2w_heading, gps0, current_gps


## !! if not stop, need to predict/interpulate the current vehicle heading
def target_re_estimation(tracker, lr):
    v2w_heading, gps0, current_gps = load_veh_status()
    vehicle_cur_pos_meter = gps_to_meter(gps0, current_gps)
    target,heading = tracker.get_recent_est(v2w_heading, vehicle_cur_pos_meter, lr)
    return target,heading

def target_re_estimation_by_both(tracker):
    v2w_heading, gps0, current_gps = load_veh_status()
    vehicle_cur_pos_meter = gps_to_meter(gps0, current_gps)
    target,heading = tracker.get_recent_est_by_both(v2w_heading, vehicle_cur_pos_meter)
    return target,heading


def update_pathx5_by_cam(new_target, heading):

    with open(path_fn, 'r') as f:
        line = f.readline()
    path = json.loads(line)
    target = path[-2]

    print('old path', path)

    with open(path_fn_pre, 'w') as file:
        json.dump(path, file)

    ## change location first (x(East) first, then y(North)), heading not accurate yet
    if UPDATE_X_TAG:
        path[-2][0] = new_target[0]
    if UPDATE_Y_TAG:
        path[-2][1] = new_target[1] ## keep the y same, only change x for demo
    if UPDATE_H_TAG:
        path[-2][2] = heading
   
    new_target.append(path[-2][2])


    ## also change the extension point
    ref = path[-2]
    new_x, new_y = forward_point(ref[0], ref[1], ref[2], 5)
    if UPDATE_X_TAG:
        path[-1][0] = new_x 
    if UPDATE_Y_TAG:
        path[-1][1] = new_y ## keep the y same, only change x for demo
    if UPDATE_H_TAG:
        path[-1][2] = heading

    print('new path', path)

    with open(path_fn, 'w') as file:
        json.dump(path, file)

    return target, new_target
   


def update_pathx5_by_cam_x3(new_target, heading):

    y_offset = -1.0

    ## mandatory 6-2 degree
    heading = 4

    with open(path_fn, 'r') as f:
        line = f.readline()
    path = json.loads(line)
    target = path[-2]

    print('old path', path)

    if abs(float(target[0])-float(new_target[0]))>2:
        valid_l_count = 0
        valid_r_count = 0
        valid_lr_count = 0
        return target, target

    with open(path_fn_pre, 'w') as file:
        json.dump(path, file)

    ## change location first (x(East) first, then y(North)), heading not accurate yet
    if UPDATE_X_TAG:
        path[-2][0] = new_target[0]
    if UPDATE_Y_TAG:
        path[-2][1] = new_target[1] ## keep the y same, only change x for demo
    if UPDATE_H_TAG:
        path[-2][2] = heading
   
    new_target.append(path[-2][2])


    ## also change the extension point
    ref = path[-2]
    new_x, new_y = forward_point(ref[0], ref[1], ref[2], 5)
    if UPDATE_X_TAG:
        path[-1][0] = new_x 
    if UPDATE_Y_TAG:
        path[-1][1] = new_y ## keep the y same, only change x for demo
    if UPDATE_H_TAG:
        path[-1][2] = heading


    ## also shift the previous path point -3 to -4 and add a pre-enter point 
    ## need to use all 5 points? (no need for now)
    # path[-4] = path[-3]
    # path[-5] = path[-3]

    new_x_pre, new_y_pre = forward_point(ref[0], ref[1], ref[2], -8)
    if UPDATE_X_TAG:
        path[-3][0] = new_x_pre
    if UPDATE_Y_TAG:
        path[-3][1] = new_y_pre ## keep the y same, only change x for demo
    if UPDATE_H_TAG:
        path[-3][2] = heading


    ## modi target temporary fix overhoot on Y
    new_x_modi, new_y_modi = forward_point(ref[0], ref[1], ref[2], y_offset)
    if UPDATE_X_TAG:
        path[-2][0] = new_x_modi 
    if UPDATE_Y_TAG:
        path[-2][1] = new_y_modi ## keep the y same, only change x for demo
    if UPDATE_H_TAG:
        path[-2][2] = heading


    path[-1][0] = (path[-1][0]+path[-2][0])/2
    path[-1][1] = (path[-1][1]+path[-2][1])/2
    # path[-1][0] = path[-2][0]
    # path[-1][1] = path[-2][1]


    print('new path', path)

    with open(path_fn, 'w') as file:
        json.dump(path, file)

    return target, new_target
   



def forward_point(x, y, heading_deg, distance):
    theta = math.radians(heading_deg)

    dx = distance * math.sin(theta)
    dy = distance * math.cos(theta)

    return x + dx, y + dy



#### communication with current web server ########
def call_stop_service():
    url = "https://localhost:5001/stop_exc_for_replan"

    try:
        response = requests.post(url, verify=False)  # verify=False if self-signed cert
        
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Failed:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print("Error calling service:", e)


def call_resume_service():
    url = "https://localhost:5001/start_exc_for_replan"

    try:
        response = requests.post(url, verify=False)  # verify=False if self-signed cert
        
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Failed:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print("Error calling service:", e)


def target_est_by_rs(anchor_points_rw, frame, v2w_heading, vehicle_cur_pos_meter):

    pc = None
    pc2veh = None
    pc2world = None
    target_heading = None
    pc_img = None
    pa_img = []


    ## estimate heading by start/end point on image
    p1 = anchor_points_rw["right"]['start_meters']
    p2 = anchor_points_rw["right"]['end_meters']
    v2p_heading_sen = estimate_v2p_heading_by_sense_line(p1, p2)
    # print(f"Orientation est by right lane:({v2p_heading_sen:.2f}°)")

    if True:
        ##
        ref_p =  anchor_points_rw["right"]['start_meters']
        x,y = estimate_parking_center_from_start(ref_p, v2p_heading_sen) # -15 earlier stage (v2p heading)
        # print("parking2Cam est by right lane start point:", x, y)
        p1, p2, p3, p4 = estimate_parking_area_from_start(ref_p, v2p_heading_sen)
        pc = [x,y]
        
        try:
            ## parking center
            u,v = project_point_with_fov_to_img(x,y) 
            # print("est center img", u,v, "realworld", x,y, "referring to start point of right lane")
            
            ## parking area
            u1,v1 = project_point_with_fov_to_img(p1[0], p1[1])
            u2,v2 = project_point_with_fov_to_img(p2[0], p2[1])
            u3,v3 = project_point_with_fov_to_img(p3[0], p3[1])
            u4,v4 = project_point_with_fov_to_img(p4[0], p4[1])
            pc_img = (u,v)
            pa_img = [(u1,v1), (u2,v2), (u3,v3), (u4,v4)]
           
            # print("-(est by right)----",u,v,u1,v1,u2,v2,u3,v3,u4,v4)

            # frame = draw_parking_center(frame, (u,v), x, y, color=(255, 0, 0), by='right')
            # frame = draw_parking_area(frame, [(u1,v1), (u2,v2), (u3,v3), (u4,v4)], color=(255, 0, 0))
            
        except:
            # print("fail projection")
            # exit()
            pass

        ## (x,y) is the center of the parking lot regarding camera 
        ## world_x, world_y is the center of the parking lot (in East/North) regarding to veh heading can cam2veh mount
        ## finally also need to transform to the world coordinate regarding to the initial point of vehicle
        # world_x, world_y = parking_center_to_vehicle(x, y, v2p_heading_sen+v2w_heading)
        world_x, world_y = parking_center_to_vehicle(x, y, v2w_heading) ## doesn't matter with the sensed heading
        pc2veh = [world_x, world_y]
        # vehicle_cur_pos_meter = [10,10] ## need to transfer by gps
        world_x0, world_y0 = target_wrt_initial_point(world_x, world_y, vehicle_cur_pos_meter)
        pc2world = [world_x0, world_y0]
        
        ## from current vehicle position [0,0, CAN_h] to target [x,y,CAN_h+sensed_heading_off]
        target_heading = v2p_heading_sen+v2w_heading
        if target_heading<0:
            target_heading+= 360
        start = [0, 0, v2w_heading]
        target = [world_x, world_y, target_heading]
        
        # print("** spot in cam, vehicle, world x3 coord:", x, y, ", ", world_x, world_y, ", ", world_x0, world_y0, "heading x2", v2p_heading_sen, target_heading)

        ## compare and use the current (world_x0, world_y0) to replace the static planned target (sx,sy), 
        data_to_save = [start, target]

        # Save to a text file use current as [0,0]
        
        with open(v2spot_file_path, 'a') as f:
            # Option 1: Save as a string representation of the list
            f.write(str(data_to_save)+"\n")

        ## to test and compare with the pathx5
        ## compare and use the current (world_x0, world_y0) to replace the static planned target (sx,sy), 
        # Save to a text file use initial point as [0,0]
      
        start0 = [vehicle_cur_pos_meter[0], vehicle_cur_pos_meter[1], v2w_heading]
        target0 = [vehicle_cur_pos_meter[0]+world_x, vehicle_cur_pos_meter[1]+world_y, target_heading]
        data_to_save0 = [start0, target0]
        # print("new path obtain current fr:", start0, "to", target0)

        
        with open(v2spot_file_path_world, 'a') as f:
            # Option 1: Save as a string representation of the list
            f.write(str(data_to_save0)+"\n")

    return pc_img, pa_img, pc, pc2veh, pc2world, target_heading



def target_est_by_ls(anchor_points_rw, frame, v2w_heading, vehicle_cur_pos_meter):

    pc_l = None
    pc2veh_l = None
    pc2world_l = None
    target_heading_l = None
    pc_img_l = None
    pa_img = []


    # estimate heading by start/end point on image
    p1_l = anchor_points_rw["left"]['start_meters']
    p2_l = anchor_points_rw["left"]['end_meters']
    v2p_heading_sen_l = estimate_v2p_heading_by_sense_line(p1_l, p2_l)
    print(f"Orientation est by left lane:({v2p_heading_sen_l:.2f}°)")

    # if lbp_valid:
    if True:
        ##
        ref_p =  anchor_points_rw["left"]['start_meters']
        xl,yl = estimate_parking_center_from_start_l(ref_p, v2p_heading_sen_l) # -15 earlier stage (v2p heading)
        print("parking2Cam est by left lane start point:", xl, yl)
        p1, p2, p3, p4 = estimate_parking_area_from_start_l(ref_p, v2p_heading_sen_l)
        pc_l = [xl,yl]
        
        try:
            ## parking center
            u,v = project_point_with_fov_to_img(xl,yl) 
            # print("est center img", u,v, "realworld", x,y, "referring to start point of right lane")
            
            ## parking area
            u1,v1 = project_point_with_fov_to_img(p1[0], p1[1])
            u2,v2 = project_point_with_fov_to_img(p2[0], p2[1])
            u3,v3 = project_point_with_fov_to_img(p3[0], p3[1])
            u4,v4 = project_point_with_fov_to_img(p4[0], p4[1])
            pa_img = [(u1,v1), (u2,v2), (u3,v3), (u4,v4)]
           
            # print("-(est by left)----",u,v,u1,v1,u2,v2,u3,v3,u4,v4)
            pc_img_l = (u,v)
            # frame = draw_parking_center(frame, (u,v), xl, yl, color=(255, 0, 255), by='left')
            # frame = draw_parking_area(frame, [(u1,v1), (u2,v2), (u3,v3), (u4,v4)], color=(255, 0, 255))
            
        except:
            # print("fail projection")
            # exit()
            pass

        ## (x,y) is the center of the parking lot regarding camera 
        ## world_x, world_y is the center of the parking lot (in East/North) regarding to veh heading can cam2veh mount
        ## finally also need to transform to the world coordinate regarding to the initial point of vehicle
        world_x_l, world_y_l = parking_center_to_vehicle(xl, yl, v2w_heading)
        pc2veh_l = [world_x_l, world_y_l]
        
        # print(">>",vehicle_cur_pos_meter)
        # vehicle_cur_pos_meter = [10,10] ## need to transfer by gps
        world_x0_l, world_y0_l = target_wrt_initial_point(world_x_l, world_y_l, vehicle_cur_pos_meter)
        pc2world_l = [world_x0_l, world_y0_l]

        
        ## from current vehicle position [0,0, CAN_h] to target [x,y,CAN_h+sensed_heading_off]
        target_heading_l = v2p_heading_sen_l+v2w_heading
        if target_heading_l<0:
            target_heading_l+= 360
        start = [0, 0, v2w_heading]
        target = [world_x_l, world_y_l, target_heading_l]
        
        # print("** by left, spot in cam, vehicle, world x3 coord:", xl, yl, v2p_heading_sen_l, ", ", world_x_l, world_y_l, target_heading_l, ", ", world_x0_l, world_y0_l, target_heading_l)

    return pc_img_l, pa_img, pc_l, pc2veh_l, pc2world_l, target_heading_l


def target_est_by_le(anchor_points_rw, frame, v2w_heading, vehicle_cur_pos_meter):

    pc_l = None
    pc2veh_l = None
    pc2world_l = None
    target_heading_l = None
    pa_img = []


    # 4th end with left lane end point
    p1_l = anchor_points_rw["left"]['start_meters']
    p2_l = anchor_points_rw["left"]['end_meters']
    v2p_heading_sen_l = estimate_v2p_heading_by_sense_line(p1_l, p2_l)
    # print(f"Orientation est by left lane:({v2p_heading_sen_l:.2f}°)")

    if True:
        ##
        ref_p =  anchor_points_rw["left"]['end_meters']
        xl,yl = estimate_parking_center_from_end_l(ref_p, v2p_heading_sen_l) # -15 earlier stage (v2p heading)
        # print("parking2Cam est by left lane end point:", xl, yl)
        p1, p2, p3, p4 = estimate_parking_area_from_end_l(ref_p, v2p_heading_sen_l)
        pc_l = [xl,yl]
        
        try:
            ## parking center
            u,v = project_point_with_fov_to_img(xl,yl) 
            # print("est center img", u,v, "realworld", x,y, "referring to start point of right lane")
            
            ## parking area
            u1,v1 = project_point_with_fov_to_img(p1[0], p1[1])
            u2,v2 = project_point_with_fov_to_img(p2[0], p2[1])
            u3,v3 = project_point_with_fov_to_img(p3[0], p3[1])
            u4,v4 = project_point_with_fov_to_img(p4[0], p4[1])
            pa_img = [(u1,v1), (u2,v2), (u3,v3), (u4,v4)]
           
            # print("-(est by left)----",u,v,u1,v1,u2,v2,u3,v3,u4,v4)
            # frame = draw_parking_center(frame, (u,v), xl, yl, color=(255, 0, 0), by='left')
            # frame = draw_parking_area(frame, [(u1,v1), (u2,v2), (u3,v3), (u4,v4)], color=(255, 0, 255))
            
        except:
            # print("fail projection")
            # exit()
            pass

        ## (x,y) is the center of the parking lot regarding camera 
        ## world_x, world_y is the center of the parking lot (in East/North) regarding to veh heading can cam2veh mount
        ## finally also need to transform to the world coordinate regarding to the initial point of vehicle
        world_x_l, world_y_l = parking_center_to_vehicle(xl, yl, v2w_heading)
        pc2veh_l = [world_x_l, world_y_l]
        
        # print(">>",vehicle_cur_pos_meter)
        # vehicle_cur_pos_meter = [10,10] ## need to transfer by gps
        world_x0_l, world_y0_l = target_wrt_initial_point(world_x_l, world_y_l, vehicle_cur_pos_meter)
        pc2world_l = [world_x0_l, world_y0_l]

        
        ## from current vehicle position [0,0, CAN_h] to target [x,y,CAN_h+sensed_heading_off]
        target_heading_l = v2p_heading_sen_l+v2w_heading
        if target_heading_l<0:
            target_heading_l+= 360
        start = [0, 0, v2w_heading]
        target = [world_x_l, world_y_l, target_heading_l]
        
        # print("** by left, spot in cam, vehicle, world x3 coord:", xl, yl, v2p_heading_sen_l, ", ", world_x_l, world_y_l, target_heading_l, ", ", world_x0_l, world_y0_l, target_heading_l)

    return pc_l, pa_img, pc2veh_l, pc2world_l, target_heading_l



def control_switch_path(tracker, lr):

    # 1. stop veh
    print("send stop cmd")
    call_stop_service()
    
    # 2. reload heading and get current path
    time.sleep(2.0) ## wait till heading converge
    new_target,heading = target_re_estimation(tracker, lr)
    print("new planned target", new_target)

     # 3. resume (update pathx5, resume task)
    if new_target:
        # pass
        # t0, t1 = update_pathx5_by_cam(new_target, heading)
        t0, t1 = update_pathx5_by_cam_x3(new_target, heading)
        print("!! path succ switched target fr", t0, "to", t1)

    else:
        print("!! not enough cam info to replan path")

    call_resume_service()

    return new_target, heading


# def control_switch_path2(tracker):
    
#     # 1. stop veh
#     print("send stop cmd")
#     call_stop_service()
    
#     # 2. reload heading and get current path
#     time.sleep(2.0) ## wait till heading converge
#     new_target,heading = target_re_estimation_by_both(tracker)
#     print("new planned target", new_target)

#      # 3. resume (update pathx5, resume task)
#     if new_target:
#         # pass
#        # t0, t1 = update_pathx5_by_cam(new_target, heading)
#         t0, t1 = update_pathx5_by_cam_x3(new_target, heading)
#         print("!! path succ switched target fr", t0, "to", t1)

#     else:
#         print("!! not enough cam info to replan path")

    
#     call_resume_service()

#     return new_target, heading



def control_switch_path_no_stop(new_target, heading):

    if new_target:
        # pass
        # t0, t1 = update_pathx5_by_cam(new_target, heading)
        t0, t1 = update_pathx5_by_cam_x3(new_target, heading)
        print("!! path succ switched target fr", t0, "to", t1)

    else:
        print("!! not enough cam info to replan path")

 

# def control_switch_path2_no_stop(target_l, target_r):
    
#     new_target = [(target_l[0]+target_r[0])/2, (target_l[1]+target_r[1])/2]

#     if new_target:
#         # pass
#         # t0, t1 = update_pathx5_by_cam(new_target, heading)
        # t0, t1 = update_pathx5_by_cam_x3(new_target, heading)
#         print("!! path succ switched target fr", t0, "to", t1)

#     else:
#         print("!! not enough cam info to replan path")



def control_switch_path_no_stop_mul(tracker, lr):
    
    # new_target,heading = tracker.get_recent_est_mul_avg(lr)
    # new_target,heading = tracker.get_recent_est_mul_avg_rm_outliers(lr)
    new_target,heading = tracker.get_recent_est_mul_avg_rm_outliers_dy_buffer(lr)

    if new_target:
        # pass
        # t0, t1 = update_pathx5_by_cam(new_target, heading)
        t0, t1 = update_pathx5_by_cam_x3(new_target, heading)
        print("!! path succ switched target fr", t0, "to", t1)

    else:
        print("!! not enough cam info to replan path")

    return new_target,heading

 

# def control_switch_path2_no_stop_mul(tracker):

#     new_target,heading = tracker.get_recent_est_by_both_mul_avg()

#     if new_target:
#         # pass
#         # t0, t1 = update_pathx5_by_cam(new_target, heading)
#         t0, t1 = update_pathx5_by_cam_x3(new_target, heading)
#         print("!! path succ switched target fr", t0, "to", t1)

#     else:
#         print("!! not enough cam info to replan path")

#     return new_target,heading



######################################################################


# for rec_mid_in, emulate the turning
# Define the index range (60 to 220 inclusive)

# ts = 30 # for left in
# te = 150
# ts = 60 # for middle in
# te = 200
# ts = 55 # for right in
# te = 220
# ts = 90 # test0205
# te = 430
# indices = np.arange(ts, te+1)
# # Generate values that change linearly from 90 to 0
# values = np.linspace(90, 0, len(indices))
# # Combine into a dictionary or DataFrame
# mapping = dict(zip(indices, values))



## streaming

## offline
# video_path = "rec1.mp4" 
# video_path = "rec2c.mp4" 
# video_path = "rec_left_in.mp4" 
# video_path = "rec_mid_in.mp4" 
# video_path = "rec_right_in.mp4"
# video_path = "off_close_line.mp4" 
# video_path = "off_mid_line.mp4" 
# video_path = "off_far_line.mp4" 
# video_path = "test11.mp4"
# video_path = "test2.mp4"
# video_path = "test0205.mp4" # 534 frame, 157 gps/heading read
# video_path = "cam_0216.mp4" 
video_path = "test11.mp4"


# video_path = "./0209/g1/cam_rec1.mp4" # 1986 frame, 230~1636 move
# video_path = "./0209/g2/cam_rec2.mp4" # 3106 frame , 280~2580 move

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)


## not good at fixing the buffer caused frame jump issue
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cap.set(cv2.CAP_PROP_FPS, 15)
# for _ in range(3):
#     cap.grab()
# ret, frame = cap.retrieve()


# Set lower resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

# Optional: lower FPS
# cap.set(cv2.CAP_PROP_FPS, 30)

## change exposure
# For some drivers/backends (like V4L2), use a specific value to turn off auto exposure
# Common values are 1 (manual) or 3 (auto) for v4l2, and 0.25 (manual) or 0.75 (auto) for others
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
# Windows (DSHOW): Values are often negative indices ranging from 0 (longest exposure) to around -13 (shortest exposure).
# Linux (V4L2): Values can be a direct representation of exposure time (e.g., 0.1 for 1/10s) or an absolute value that needs to be determined for your specific camera.
# cap.set(cv2.CAP_PROP_EXPOSURE, 0.1) # Example value, adjust based on camera

print("exposure", cap.get(cv2.CAP_PROP_EXPOSURE))
# exit()


if not cap.isOpened():
    print("no camera found")
    exit()

frame_skip = 1 #2
count = 0
tracker = LaneTracker(max_history=500) 




## output mp4 format
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if metadata is missing
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video_output = cv2.VideoWriter('lane_detection_output.mp4', fourcc, fps, (w, h))
# video_output = cv2.VideoWriter('lane_detection_output.mp4', fourcc, fps, (1920, 1080))

valid_r_count = 0
valid_l_count = 0
valid_lr_count = 0
decision_has_made = False
decision_made_by = None

## load interpulated path
xyh = load_interp_path()
static_target = tracker.get_static_target()
# print(static_target)
# exit()

# yolo_model = YOLO('yolov8n.pt')
# yolo_model = YOLO('yolo11n.pt')
# yolo_model = YOLO('yolo26n.pt')  # Optimized for edge deployment

# yolo_seg_model = YOLO('yolo11n-seg.pt') 
# yolo_seg_model = YOLO('yolo26n-seg.pt') 

person_clear_time = None

while True:

    ## read/load stream
    ret, frame = cap.read()

    if not ret:
        print("stream reading fail")
        break

        # print("Stream lost. Reconnecting...")
        # cap.release()
        # time.sleep(1)
        # cap = cv2.VideoCapture(0)  # or your stream URL
        # continue

    count += 1
    if count % frame_skip != 0:
        continue   # skip processing

    print("frame", count)

    ## image preprocessing
    frame = cv2.resize(frame, (w, h))

    # # Undistort with alpha (crop vs extend black)
    # alpha = 0.0 # this is what the original provide wxh to estimate fov (~98, ~82)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    #     K, dist_coeffs, (w, h), alpha, (w, h)
    # )
    

    # scale = 0.8  # <1 → wider view
    # newcameramtx[0,0] *= scale  # fx
    # newcameramtx[1,1] *= scale  # fy

    # frame = cv2.undistort(frame, K, dist_coeffs, None, newcameramtx)

    # ## new fov after undistort
    # fx_new = newcameramtx[0, 0]
    # fy_new = newcameramtx[1, 1]

    # hfov_new = 2 * np.arctan(w / (2 * fx_new))
    # vfov_new = 2 * np.arctan(h / (2 * fy_new))

    # hfov_new_deg = np.degrees(hfov_new)
    # vfov_new_deg = np.degrees(vfov_new)

    # print("New HFOV:", hfov_new_deg)
    # print("New VFOV:", vfov_new_deg)
    # exit()

    ## basic
    frame = cv2.undistort(frame, K, dist_coeffs)


    # alpha = 1.0  # 1 = keep full FOV
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    #     K, dist_coeffs, (w, h), alpha, (w, h)
    # )

    # # Optional: further widen FOV by scaling focal length
    # scale = 0.85  # <1 → wider view
    # newcameramtx[0, 0] *= scale  # fx
    # newcameramtx[1, 1] *= scale  # fy

    # # Undistort the frame
    # frame = cv2.undistort(frame, K, dist_coeffs, None, newcameramtx)

    # # -------------------------
    # # Compute new FOV
    # # -------------------------
    # fx_new = newcameramtx[0, 0]
    # fy_new = newcameramtx[1, 1]

    # hfov_new = 2 * np.arctan(w / (2 * fx_new))
    # vfov_new = 2 * np.arctan(h / (2 * fy_new))

    # hfov_new_deg = np.degrees(hfov_new)
    # vfov_new_deg = np.degrees(vfov_new)

    # print("New HFOV:", hfov_new_deg)
    # print("New VFOV:", vfov_new_deg)


    
    

    
    ## init vehicle status
    v2w_heading = 0 # v2p for camera stream ui, v2w for execution
    current_gps = None
    gps0 = None
 
    ## remove, only for testing
    gps0 = (42.517299, -83.045387) ## ref (e.g. init point)
    current_gps = (42.517319, -83.045271)
    v2w_heading = 0
    vehicle_cur_pos_meter = None

    ## load veh status
    try:
        v2w_heading, gps0, current_gps = load_veh_status()
        vehicle_cur_pos_meter = gps_to_meter(gps0, current_gps)     
        print("veh sensed status (head, lat, lon, meter pos)", v2w_heading, current_gps, vehicle_cur_pos_meter)
        
        ## search v2w heading by the recent location fr the interpulated path, avoid delay from can heading reading  
        search_ele = search_heading_by_loc(vehicle_cur_pos_meter[0], vehicle_cur_pos_meter[1], xyh)
        
        ## search the closed point on the phone planned path (assume strict follow path)
        v2w_heading_planned = search_ele[2]
        # print("veh status est interpulated heading", v2w_heading)

        ## prediction based on delta tracking history
        # if len(tracker.get_history_list['veh_heading'])>0:
        #     v2w_heading = tracker.pred_heading
        # print("veh sensing delay, delta_H, and predicted heading", tracker.delay, tracker.delta_heading, tracker.pred_heading)
        print("veh sensed heading, planned heading, sensing delay, delta_H, steering, and predicted heading, current timing", v2w_heading, v2w_heading_planned, tracker.delay, tracker.delta_heading, tracker.steer, tracker.pred_heading, tracker.curr_timing)

        
    except:
        print("veh status loading fail")


                
   
    
    ### JJJJJ ###

    ori_img = frame.copy()

    if YOLO_DET and count%1==0:
        # YOLO Person Detection
        try:
            person_in_zone = False
            results = yolo_model(frame, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id == 0:  # 0 is the class for 'person' in COCO
                        conf = box.conf[0].item()
                        
                        if conf > 0.5:
                            # Extract bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            # Calculate center-bottom of bounding box
                            bx = (x1 + x2) // 2
                            by = y2
                            
                            # Project pixel to world to get distance
                            rw_coords = pixel_to_world(bx, by)
                            if rw_coords:
                                # rw_coords is (xw, yw, 0.0), where yw is forward distance in meters
                                yw = rw_coords[1]
                                
                                if yw < 6.0:
                                    person_in_zone = True
                                    # Show alert if proximity < 6m
                                    alert_text = f"ALERT: Person {yw:.1f}m"
                                    cv2.putText(frame, alert_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                                    if emg_stop_tag == True:
                                        call_stop_service()
                                        emg_stop_tag = False
                            
                            # Draw box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"Person {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Resume service if person exited zone and vehicle was stopped
            if person_in_zone:
                person_clear_time = None
            elif not emg_stop_tag:
                if person_clear_time is None:
                    person_clear_time = time.time()
                else:
                    elapsed = time.time() - person_clear_time
                    if elapsed >= 2.0:
                        call_resume_service()
                        emg_stop_tag = True
                        person_clear_time = None
                    else:
                        # Show "Person Left" on screen during the 1-second delay
                        alert_text = f"PERSON LEFT, res in {2.0 - elapsed:.1f}s"
                        cv2.putText(frame, alert_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        except Exception as e:
            print(f"YOLO detection failed: {e}")
    


    # if YOLO_SEG and count%1==0:
    #     # YOLO Person Detection
    #     try:
    #         person_in_zone = False
    #         results = yolo_seg_model(frame, verbose=False)
    #         # print(results)
    #         # exit()
         
    #     except Exception as e:
    #         print(f"YOLO segmentation failed: {e}")
        

    ## basic add center of image point     
    cv2.circle(frame, (w//2, h//2), 10, (0,0,0), -1)
    
    # grid and mask system
    frame, project_points, mask = est_distance_grid(frame)

    ## ROI mask
    mask_colored = np.zeros_like(frame)
    mask_colored[:] = (0, 255, 0)  # BGR for Green
    mask_overlay = cv2.bitwise_and(mask_colored, mask_colored, mask=mask)
    frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.2, 0)
    
    ##########################

    # detection here
    dis2target = -1
    try:
        dis2target = ((vehicle_cur_pos_meter[0]-static_target[0])**2+(vehicle_cur_pos_meter[1]-static_target[1])**2)**0.5
    except:
        pass

    ## reset and ready for 2nd round detection when out of range
    if dis2target>Det_dis_reset:
        valid_r_count = 0
        valid_l_count = 0
        valid_lr_count = 0
        decision_has_made = False
        decision_made_by = None

    # if v2w_heading>Det_heading_trigger[0] and v2w_heading>Det_heading_trigger[0]:
    # if v2w_heading>Det_heading_trigger[0] and v2w_heading>Det_heading_trigger[0] and dis2target>Det_distance_trigger[0] and dis2target<Det_distance_trigger[1] and tracker.delta_heading<Delta_heading_thr:
        # print("&&&&&&&&&&", dis2target, v2w_heading, vehicle_cur_pos_meter, static_target)
    if True:
    # if tracker.delta_heading<Delta_heading_thr:
    # !! based on the current vehicle gps to the target_static, and the current heading to the target heading, to find when to start lane detection 

        ## edge & Lane detection
        # edges, lines = edge_det_mask_range(ori_img, hough_thr, blur_thr, mask)  # Traditional CV
        edges, lines = edge_det_yolop(ori_img, hough_thr, blur_thr, mask)  # YOLOP ML
        # edges, lines = edge_det_mask_range_otsu(ori_img, hough_thr, blur_thr, mask)
        # edges, lines = edge_det_mask_range_otsu_rm_dark(ori_img, hough_thr, blur_thr, mask)
        # edges, lines = edge_det_mask_range_otsu_rm_dark_dyn_hough(ori_img, hough_thr, blur_thr, mask)
        # edges, lines = edge_det_adaptive_thresh(ori_img, hough_thr, blur_thr, mask)
        # edges, lines = edge_det_adaptive_thresh_dyn_hough(ori_img, hough_thr, blur_thr, mask)
        
        frame = add_edge_on_img(frame, edges)
        frame = draw_all_hough_lines(frame, lines) ## deb

        # Project static target onto image plane to separate left/right lines
        target_center_x = w // 2
        try:
            # v2w_heading = 330 # 0
            if static_target is not None and vehicle_cur_pos_meter is not None and v2w_heading is not None:
                delta_E = static_target[0] - vehicle_cur_pos_meter[0]
                delta_N = static_target[1] - vehicle_cur_pos_meter[1]
                # h_rad = np.deg2rad(v2w_heading) ## JJJ heading
                h_rad = np.deg2rad(tracker.pred_heading)

                # Note: Navigation heading (N=0, E=90)
                # Inverse of parking_center_to_vehicle:
                # E = sin(h)*y_v + cos(h)*x_v
                # N = cos(h)*y_v - sin(h)*x_v

                # Therefore:
                x_v = np.cos(h_rad) * delta_E - np.sin(h_rad) * delta_N
                y_v = np.sin(h_rad) * delta_E + np.cos(h_rad) * delta_N
                x_c = x_v
                y_c = y_v - vehicle_length / 2
                pt = project_point_with_fov_to_img(x_c, y_c)
                if pt is not None:
                    target_center_x = max(0, min(w, pt[0]))
                    # print("static t trans", static_target, vehicle_cur_pos_meter, v2w_heading, x_c, y_c)
                    # print("project static target on img X", pt, target_center_x)
                    if not decision_has_made:
                        cv2.circle(frame, (int(target_center_x), int(pt[1])), 10, (0, 0, 255), 3)
        
        except Exception as e:
            print("Failed to project static target to image:", e)

        # ll,rl,mid_p,err_x, err_h = find_LR2(lines, w, h, angle_thr_deg)     
        # ll,rl,mid_p,err_x, err_h = find_LR2_target(lines, w, h, angle_thr_deg, target_center_x = None) ## target center None, default w/2
        ll,rl,mid_p,err_x, err_h = find_LR2_target(lines, w, h, angle_thr_deg, target_center_x = target_center_x) ## target center based on current veh status and static sensed spot
        
        # if rl is None and ll:
        #     rl = ll
        #     ll = None

        # focuse on right line only instead (if left not visible and right lay on the left side from overshooting)

        # ll = None
        # rl = find_PL(lines, w, h, angle_thr_deg)

        ## add left/right parking lane to UI
        # frame, lane_ep = add_lanes_on_img(frame, ll, rl, mask, project_points)
        ## !!! try find connected max support line after find the intersection points between line equation and mask border 
        # frame, lane_ep = add_lanes_on_img_with_endpoints(frame, ll, rl, mask, project_points, edges, 1, 15)
        
        lane_ep = find_lanes_on_img_with_endpoints_seg_check(frame, ll, rl, mask, project_points, edges, 3, 10) # prev 1,15 
        # print(lane_ep)

        

        ## ref point detection
        anchor_points_rw = find_anchor_points(lane_ep)    


        ## est per frame 
        pc = None
        pc2veh = None
        pc2world = None
        target_heading = None
    
        pc_l = None
        pc2veh_l = None
        pc2world_l = None
        target_heading_l = None

        pc_img = None
        pa_img = []
        pa_img_l = []


        ## parking spot center estimation #####
        v2p_heading = 0 ## also test e.g. 359!!!! 

        ## 1st start with right lane close point
        if rbp_valid:
            try:
                # pc_img, pa_img, pc, pc2veh, pc2world, target_heading = target_est_by_rs(anchor_points_rw, frame, v2w_heading, vehicle_cur_pos_meter)            
                pc_img, pa_img, pc, pc2veh, pc2world, target_heading = target_est_by_rs(anchor_points_rw, frame, tracker.pred_heading, vehicle_cur_pos_meter)            
                
                ### JJJ
                ## check ref point is within the valid range (near parking spot)
                est_offset = ((pc2world[0]-static_target[0])**2 + (pc2world[1]-static_target[1])**2)**0.5
                print("phone vs camera target diff (R)", est_offset)
                if est_offset>sta_dyn_thr:
                    rbp_valid = False

                if rbp_valid:
                    ## draw out here
                    frame = draw_lane_lines(frame, lane_ep, mask)
                    frame = draw_parking_center(frame, pc_img, pc[0], pc[1], color=(255, 0, 0), by='right')
                    frame = draw_parking_area(frame, pa_img, color=(255, 0, 0))
                    valid_r_count += 1
                # exit()
            except:
                pass

        ## 2nd 
        if rtp_valid:
            pass
        
        ## 3rd start with left lane close point (if left lane visible in fov, close point is better for ref est)
        if lbp_valid:
            try:
                # pc_img_l, pa_img_l, pc_l, pc2veh_l, pc2world_l, target_heading_l = target_est_by_ls(anchor_points_rw, frame, v2w_heading, vehicle_cur_pos_meter)
                pc_img_l, pa_img_l, pc_l, pc2veh_l, pc2world_l, target_heading_l = target_est_by_ls(anchor_points_rw, frame, tracker.pred_heading, vehicle_cur_pos_meter)
                
                ### JJJ
                est_offset = ((pc2world_l[0]-static_target[0])**2 + (pc2world_l[1]-static_target[1])**2)**0.5
                print("phone vs camera target diff (L)", est_offset)
                if est_offset>sta_dyn_thr:
                    lbp_valid = False
                    
                if lbp_valid:
                    frame = draw_lane_lines(frame, lane_ep, mask)
                    frame = draw_parking_center(frame, pc_img_l, pc_l[0], pc_l[1], color=(255, 0, 255), by='left')
                    frame = draw_parking_area(frame, pa_img_l, color=(255, 0, 255))
                    
                    valid_l_count += 1
            except:
                pass

        ## 4th
        if ltp_valid:
            pass
            # pc_l, pa_img_l, pc2veh_l, pc2world_l, target_heading_l = target_est_by_le(anchor_points_rw, frame, v2w_heading, vehicle_cur_pos_meter)


        ## both lane partically visible
        # if ltp_valid and rbp_valid and pc and pc_l:
        #     # print("####### both lanes visible, heading sensed", target_heading, target_heading_l)
        #     # print("####### both lanes visible, location sensed", pc, pc_l)
        #     dis_vary = ((pc[0]-pc_l[0])**2+(pc[1]-pc_l[1])**2)**0.5
        #     if dis_vary<LR_vary_thr:
        #         valid_lr_count += 1

        



        frame = draw_distance_labels(frame, lane_ep, anchor_points_rw)

        ## debug show ref point
        frame = draw_ref_indicator(frame, rbp_valid, rtp_valid, lbp_valid, ltp_valid)
       

        tracker.update(anchor_points_rw, pc, pc_l, pc2veh, pc2veh_l, pc2world, pc2world_l, current_gps, v2w_heading, target_heading, target_heading_l, rbp_valid, lbp_valid)
        # tracker.check_consistance(window_size=10, std_threshold=0.15)
        tracker.update_delta_heading()
        # print("recent delta heading (check turning finish):", tracker.delta_heading)

        # tracker.check_stop()

        ## decision
        ## problem: v2w heading delay, world coordinate won't be stable
        ## when stop the vehicle, wait to update the recent vehicle heading, re-estimate the target
        ## if FOV good enough, wait till both lane shows and consist with each other, then stop and replan 
        
        
        ### JJJJJ ###
        ## whoever detect first, jump high to avoid a 2nd stop decision
        '''
        ## tracker by right lane 
        if valid_r_count==decision_point and PATH_REP:
            # control_switch_path_no_stop(pc2world, target_heading)
            # targ1,h1 = control_switch_path(tracker, "r") ## stop 2s
            targ2,h2 = control_switch_path_no_stop_mul(tracker, "r")
            # print("comp loc stop, no stop, mul-dec", targ1, pc2world, targ2)
            # print("comp head stop, no stop, mul-dec", h1, target_heading, h2)
            valid_r_count += 100 
            valid_l_count += 100
            # exit()
        

        ## tracker by left lane
        if valid_l_count==decision_point and PATH_REP:
            # control_switch_path_no_stop(pc2world_l, target_heading_l)
            # targ1,h1 = control_switch_path(tracker, "l") ## stop 2s
            targ2,h2 = control_switch_path_no_stop_mul(tracker, "l")
            # print("comp loc stop, no stop, mul-dec", targ1, pc2world_l, targ2)
            # print("comp head stop, no stop, mul-dec", h1, target_heading_l, h2)
            valid_l_count += 100
            valid_r_count += 100
        #     exit()
            
        ## tracker by both lanes when 2x est are consistent
        # if valid_lr_count==decision_point2:
            # control_switch_path2(tracker)
            # control_switch_path2_no_stop(pc2world, pc2world_l)
            # control_switch_path2_no_stop_mul(tracker)
            # valid_lr_count += 1
        '''

        ## also test the buffer window opt
        ## instead of counting, use minimum distance to trigger the decision
        ## when distance < 6m, whenever valid count over threshold, trigger the re-plan
        if PATH_REP and (not decision_has_made) and valid_r_count>decision_point and (not dis2target==-1) and dis2target<Decision_make_distance:
            targ2,h2 = control_switch_path_no_stop_mul(tracker, "r")
            decision_has_made = True
            decision_made_by = "L-Lane"
                        
        elif PATH_REP and (not decision_has_made) and valid_l_count>decision_point and (not dis2target==-1) and dis2target<Decision_make_distance:
            targ2,h2 = control_switch_path_no_stop_mul(tracker, "l")
            decision_has_made = True
            decision_made_by = "R-Lane"


    frame = draw_decision_tag(frame, decision_has_made, decision_made_by)
        
    cv2.imshow('stream', frame)
    video_output.write(frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
    # if cv2.waitKey(100) & 0xFF==ord('q'): ## 10ms per frame
        break


cap.release()
cv2.destroyAllWindows()








####### figure for debugging
# Access the history tracking data of anchor points
full_path = tracker.get_history_list()

with open(track_fn, 'w') as f:
    json.dump(full_path, f, indent=4)

# plot_lane_history(full_path)
# # plot_pc_history(full_path, 'park_cen')
# plot_pc_history(full_path, 'park_cen_veh')
# # plot_pc_history(full_path, 'park_cen_world')

# filter_full_path = lane_history_filtered(full_path)
# # plot_lane_history(filter_full_path)
# plot_pc_history(filter_full_path, 'park_cen_veh')


# ## compare by camera coordinate first
# # park_center_r = full_path["park_cen"]     
# # park_center_l = full_path["park_cen_l"]  
# park_center_r = full_path["park_cen_veh"]     
# park_center_l = full_path["park_cen_veh_l"]   
# comp_lr_loc_est(park_center_r, park_center_l)


# park_h_r = full_path["park_heading_world"]    
# park_h_l = full_path["park_heading_world_l"]   
# comp_lr_h_est(park_h_r, park_h_l)

print("valid right count", valid_r_count)
print("valid left count", valid_l_count)
print("valid both count", valid_lr_count)




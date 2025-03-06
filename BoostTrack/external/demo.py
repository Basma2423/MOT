import cv2
import csv
import numpy as np
import json
import torch
import os
import time
import pandas as pd
from collections import defaultdict

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

from tracker.boost_track import BoostTrack
from default_settings import GeneralSettings, BoostTrackPlusPlusSettings

TRAINING_DATA_PATH = os.path.join(CURRENT_PATH, "../../data/tracking/train")
TEST_DATA_PATH = os.path.join(CURRENT_PATH, "../../data/tracking/test/01")

def load_detector(version=8):

    from ultralytics import YOLO

    if version == 8:
        model = YOLO('yolov8s.pt')
    else:
        model = YOLO('yolo11n.pt')
    
    return model

def detect_persons(detector, frame, tag):
    """
    Input:
        detector: YOLO model
        frame: Input image frame
        tag: Frame identifier
    
    Output:
        Detections in format [x1, y1, x2, y2, confidence, class]
    """

    results = detector(frame, classes=0)  # Class 0 is person in COCO
    
    person_dets = []
    
    if len(results) > 0:
        result = results[0]
        
        # Extract boxes
        if hasattr(result, 'boxes'):
            # YOLOv8 format
            boxes = result.boxes.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i]
                confidence = boxes.conf[i]
                class_id = boxes.cls[i]
                person_dets.append([x1, y1, x2, y2, confidence, class_id])
    
    return np.array(person_dets)

def process_frame(frame, frame_count, detector, tracker, output_data, tag):
    """
    Process a single frame with the tracker and update output data
    """
    print(f"Processing frame {frame_count}")
    
    # Get detections for this frame
    detections = detect_persons(detector, frame, tag)
    
    # Convert frame to tensor format for the tracker
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    
    start_time = time.time()
    # Update tracker and get tracking results
    tracks = tracker.update(detections, img_tensor, frame, tag)
    process_time = time.time() - start_time
    
    # Format tracked objects according to the required output format
    tracked_objects = []
    for track in tracks:
        x1, y1, x2, y2, track_id, conf = track
        width = x2 - x1
        height = y2 - y1
        
        obj = {
            'tracked_id': float(track_id),
            'x': int(x1),
            'y': int(y1),
            'w': int(width),
            'h': int(height),
            'confidence': float(conf)
        }
        tracked_objects.append(obj)
    
    # Add to output data
    output_data.append({
        'ID': frame_count-1,
        'Frame': float(frame_count),
        'Objects': tracked_objects,
        'Objective': 'tracking'
    })
    
    # Visualize tracks on frame
    visualization_frame = frame.copy()
    for obj in tracked_objects:
        x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        track_id = obj['tracked_id']
        cv2.rectangle(visualization_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(visualization_frame, f"ID: {int(track_id)}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Tracking', visualization_frame)
    cv2.waitKey(1)
    
    return process_time



def track_persons_in_image_sequence(image_folder, output_file, yolo_version=8, mode="train"):
    """
    Track persons in a sequence of images and output tracking results in the specified format
    """
    import glob
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg'))) + \
                 sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    # Initialize detector
    detector = load_detector(yolo_version)
    
    # Initialize tracker
    folder_name = os.path.basename(os.path.normpath(image_folder))
    tracker = BoostTrack(video_name=folder_name)
    
    # Initialize output data
    output_data = []
    
    frame_count = 0
    total_time = 0
    
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error reading image {img_path}")
            continue
        
        frame_count += 1
        tag = f"{folder_name}:{frame_count}"
        
        process_time = process_frame(frame, frame_count, detector, tracker, output_data, tag)
        total_time += process_time
    
    tracker.dump_cache()
    
    write_output_to_csv(output_data, output_file, mode)
    
    print(f"Tracking completed: {frame_count} images processed")
    print(f"Time spent: {total_time:.3f}s, FPS: {frame_count / (total_time + 1e-9):.2f}")
    
    return output_data


def write_output_to_csv(output_data, output_file="submission.csv", mode="train", existing_file="submission_file.csv", start_row=430):
    
    # Tracking Results
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ID", "frame", "objects", "objective"])  # Header row
        
        indx = 0
        for item in output_data:
            writer.writerow([indx, float(item["Frame"]), json.dumps(item["Objects"]), item["Objective"]])
            indx += 1

    # Face ReID Results
    if mode == "test" and existing_file:
        df = pd.read_csv(existing_file)

        if start_row < len(df):
            df = df.iloc[start_row:]
            df.to_csv(output_file, mode='a', index=False, header=False)

 
    unique_ids = set()
    for item in output_data:
        for obj in item["Objects"]:
            unique_ids.add(obj["tracked_id"])

    print(f"Number of unique people detected: {len(unique_ids)}")
    print(f"Tracking results saved to {output_file}")

def setup_tracker_settings(use_embedding=True, use_ecc=True, use_rich_s=True, use_sb=True, use_vt=True):
    """
    Configure tracker settings based on parameters
    """
    GeneralSettings.values['use_embedding'] = use_embedding  # Visual embedding
    GeneralSettings.values['use_ecc'] = use_ecc  # Camera motion compensation
    BoostTrackPlusPlusSettings.values['use_rich_s'] = use_rich_s  # Use rich similarity (not just IoU)
    BoostTrackPlusPlusSettings.values['use_sb'] = use_sb  # Use soft detection confidence boost
    BoostTrackPlusPlusSettings.values['use_vt'] = use_vt  # Use varying threshold

def load_ground_truth(gt_file):
    """
    Load ground truth data ==> MOT format
    
    Ground truth format:
    Frame,Track ID,X,Y,Width,Height,Confidence,Class,Visibility
    
    Returns:
        dict: Ground truth data organized by frame
    """
    gt_data = defaultdict(list)
    
    try:
        # Try to read with pandas for more robust handling
        df = pd.read_csv(gt_file, header=None)
        
        for index, row in df.iterrows():
            frame = int(row[0])
            track_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            width = float(row[4])
            height = float(row[5])
            confidence = float(row[6])
            class_id = int(row[7])
            visibility = float(row[8]) if len(row) > 8 else 1.0
            
            gt_data[frame].append({
                'track_id': track_id,
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'confidence': confidence,
                'class': class_id,
                'visibility': visibility
            })
    except Exception as e:
        print(f"Error reading ground truth file with pandas: {e}")
        # Fallback to manual parsing
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    if line.startswith('Frame') or not line.strip():
                        continue
                    
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        print(f"Warning: Malformed line in ground truth file: {line}")
                        continue
                    
                    frame = int(parts[0])
                    track_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])
                    confidence = float(parts[6])
                    class_id = int(parts[7])
                    visibility = float(parts[8]) if len(parts) > 8 else 1.0
                    
                    gt_data[frame].append({
                        'track_id': track_id,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'confidence': confidence,
                        'class': class_id,
                        'visibility': visibility
                    })
        except Exception as e:
            print(f"Error reading ground truth file manually: {e}")
    
    return gt_data

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    box format: [x, y, width, height]
    """
    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Determine intersection coordinates
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    # Check if there's an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas of both bounding boxes
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

def evaluate_tracking(tracking_results, ground_truth, iou_threshold=0.5):
    """
    Evaluate tracking results against ground truth
    
    Args:
        tracking_results: List of dictionaries with tracking results
        ground_truth: Dictionary with ground truth data by frame
        iou_threshold: IoU threshold for considering a detection a match
    
    Returns:
        dict: Evaluation metrics
    """
    # Initialize counters
    total_gt = 0
    total_predicted = 0
    total_matches = 0
    
    # Count total ground truth objects
    for frame in ground_truth:
        total_gt += len(ground_truth[frame])
    
    # Create a frame-indexed version of tracking results
    tracking_by_frame = {}
    for item in tracking_results:
        frame = item['Frame']
        tracking_by_frame[frame] = item['Objects']
        total_predicted += len(item['Objects'])
    
    # Evaluate each frame
    id_matches = {}  # To track ID consistency
    
    for frame in ground_truth:
        gt_objects = ground_truth[frame]
        pred_objects = tracking_by_frame.get(frame, [])
        
        # Skip if no predictions for this frame
        if not pred_objects:
            continue
        
        # Match ground truth objects to predictions
        for gt_obj in gt_objects:
            best_iou = 0
            best_match = None
            
            for pred_obj in pred_objects:
                # Get IoU between ground truth and prediction
                gt_box = [gt_obj['x'], gt_obj['y'], gt_obj['width'], gt_obj['height']]
                pred_box = [pred_obj['x'], pred_obj['y'], pred_obj['w'], pred_obj['h']]
                
                iou = calculate_iou(gt_box, pred_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_obj
            
            # If we found a match above the threshold
            if best_iou >= iou_threshold and best_match is not None:
                total_matches += 1
                
                # Track ID consistency
                gt_id = gt_obj['track_id']
                pred_id = best_match['tracked_id']
                
                if gt_id not in id_matches:
                    id_matches[gt_id] = {}
                
                if pred_id not in id_matches[gt_id]:
                    id_matches[gt_id][pred_id] = 0
                
                id_matches[gt_id][pred_id] += 1
    
    # Calculate ID consistency - for each ground truth ID, find the most matched predicted ID
    id_consistency = 0
    id_switches = 0
    
    for gt_id, pred_ids in id_matches.items():
        max_matches = max(pred_ids.values())
        total_matches_for_id = sum(pred_ids.values())
        id_consistency += max_matches / total_matches_for_id if total_matches_for_id > 0 else 0
        id_switches += len(pred_ids) - 1  # Number of different IDs minus 1
    
    # Average ID consistency
    avg_id_consistency = id_consistency / len(id_matches) if id_matches else 0
    
    # Calculate metrics
    precision = total_matches / total_predicted if total_predicted > 0 else 0
    recall = total_matches / total_gt if total_gt > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'total_ground_truth': total_gt,
        'total_predictions': total_predicted,
        'true_positives': total_matches,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'id_switches': id_switches,
        'id_consistency': avg_id_consistency
    }
    
    return metrics


def visualize_tracking(image_folder, tracking_results, mode="train", output_video_path="tracking_results.mp4", ground_truth=None, iou_threshold=0.5):
    """
    Visualize tracking results compared to ground truth
    
    Args:
        image_folder: Folder containing image sequence
        tracking_results: List of dictionaries with tracking results
        ground_truth: Dictionary with ground truth data by frame
        output_video_path: Path to save visualization video
        iou_threshold: IoU threshold for considering a detection a match
    """
    import glob
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg'))) + \
                 sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    # Create frame-indexed version of tracking results
    tracking_by_frame = {}
    for item in tracking_results:
        frame = item['Frame']
        tracking_by_frame[frame] = item['Objects']

    if mode == "test":
        output_video_path = "test_tracking_results.mp4"
    
    # Initialize video writer
    first_frame = cv2.imread(image_files[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
    
    # Process each frame
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error reading image {img_path}")
            continue
        
        frame_num = i + 1  # Assuming 1-indexed frames in ground truth

        if mode != "test":

            # Draw ground truth boxes in green
            if frame_num in ground_truth:
                for gt_obj in ground_truth[frame_num]:
                    x, y = int(gt_obj['x']), int(gt_obj['y'])
                    w, h = int(gt_obj['width']), int(gt_obj['height'])
                    track_id = gt_obj['track_id']
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"GT-{track_id}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tracking results boxes in red
        if frame_num in tracking_by_frame:
            for track_obj in tracking_by_frame[frame_num]:
                x, y = track_obj['x'], track_obj['y']
                w, h = track_obj['w'], track_obj['h']
                track_id = int(track_obj['tracked_id'])
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"T-{track_id}", (x, y+h+15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        video_writer.write(frame)
        
        # Display the frame
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Visualization saved to {output_video_path}")


def try_on_training(yolo_version=8):

    # Extract sequence name from path
    seq_name = os.path.basename(os.path.normpath(TRAINING_DATA_PATH))
    # If the path is like "../../data/tracking/train/02", extract "02"

    print(f'Seq. Name: {seq_name}')

    TRAINING_DATA_PATH = os.path.join(CURRENT_PATH, seq_name)
    
    # Paths
    gt_folder = os.path.join(TRAINING_DATA_PATH, "gt")
    images_path = os.path.join(TRAINING_DATA_PATH, "img1")
    output_file = f"{seq_name}_tracking_results.csv"

    # Run tracking
    print(f"Running tracking with YOLOv{yolo_version}...")
    tracking_results = track_persons_in_image_sequence(images_path, output_file, yolo_version)

    results = evaluate_tracking(tracking_results, load_ground_truth(os.path.join(gt_folder, "gt.txt")))
    print("Results:")
    print(f"Total ground truth objects: {results['total_ground_truth']}")
    print(f"Total predicted objects: {results['total_predictions']}")
    print(f"True positives: {results['true_positives']}")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall: {results['recall']*100:.2f}%")
    print(f"F1 Score: {results['f1_score']*100:.2f}%")
    print(f"ID Switches: {results['id_switches']}")
    print(f"ID Consistency: {results['id_consistency']*100:.2f}%")
    
    
    # Run TrackEval evaluation
    # print("Running TrackEval evaluation...")
    # tracker_name = f"BoostTrack_YOLOv{yolo_version}"
    # results, messages = evaluate_using_trackeval(
    #     tracking_results, 
    #     gt_folder, 
    #     tracker_name=tracker_name, 
    #     seq_name="Fawry"
    # )
    
    # # Print summary of results
    # if results:
    #     # Get the first dataset, tracker, and first combined sequence result
    #     dataset



def test(yolo_version=8):

    # Paths
    mode = "test"
    images_path = os.path.join(TEST_DATA_PATH, "img1")
    output_file = f"{mode}_tracking_results.csv"
    
    # Run tracking
    print(f"Running tracking with YOLOv{yolo_version}...")
    tracking_results = track_persons_in_image_sequence(images_path, output_file, yolo_version, mode=mode)

    print("Generating visualization...")
    output_video_path = f"{mode}_tracking.mp4"
    visualize_tracking(images_path, tracking_results, mode=mode, output_video_path=output_video_path)


if __name__ == "__main__":

    setup_tracker_settings(
        use_embedding=True,
        use_ecc=True,
        use_rich_s=True,
        use_sb=True,
        use_vt=True,
    )

    yolo_version=11
    
    # try_on_training(yolo_version)
    test(yolo_version)
import cv2
import numpy as np
import json
import torch
import os
import time
import pandas as pd
from collections import defaultdict
import glob

from tracker.boost_track import BoostTrack
from default_settings import GeneralSettings, BoostTrackPlusPlusSettings
import utils

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(CURRENT_PATH, "../../data/tracking/train")
TEST_DATA_PATH = os.path.join(CURRENT_PATH, "../../data/tracking/test/01")
CSV_FILES_PATH = os.path.join(CURRENT_PATH, "CSV_files")


def load_detector(detector_type=9):

    from ultralytics import YOLO

    if detector_type == 8:
        model = YOLO('../models/yolov8s.pt')

    elif detector_type == 9:
        model = YOLO('../models/yolov9c.pt')
    
    elif detector_type == 'pretrained_on_MOT20':
        model = YOLO('../models/yolov9c_trained.pt')

    elif detector_type == 'ours':
        model = YOLO('../models/best.pt')

    elif detector_type == 11:
        model = YOLO('../models/yolo11n.pt')

    elif detector_type == 'deim':       # not working totally
        from models.deim.tools.inference.torch_inf import TorchInference
        model = TorchInference(config="models/deim/configs/deim_dfine/dfine_hgnetv2_x_coco.yml", 
                               resume="models/deim/model.pth", 
                               device="cuda:0")

    else:
        model = YOLO('../models/yolov9c.pt')

    model.eval()
    
    return model


def detect_persons(detector, frame, detector_type=9, tag=None):

    person_dets = []

    results = detector(frame, classes=0)  # Class 0 is 'person'
        
    if len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes'):
            boxes = result.boxes.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i]
                confidence = boxes.conf[i]
                class_id = boxes.cls[i]
                person_dets.append([x1, y1, x2, y2, confidence, class_id])

    return np.array(person_dets)


def process_frame(frame, frame_count, detector, tracker, output_data, detector_type=9, tag=None):

    print(f"Processing frame {frame_count}")
    
    # detect
    detections = detect_persons(detector, frame, detector_type, tag)

    # convert to tensor format
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    
    start_time = time.time()

    # track
    tracks = tracker.update(detections, img_tensor, frame, tag)
    tlwhs, ids, confs = utils.filter_targets(tracks, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])
    process_time = time.time() - start_time
    
    # format
    tracked_objects = []
    for tlwh, track_id, conf in zip(tlwhs, ids, confs):

        x1, y1, w, h = tlwh
        
        obj = {
            'tracked_id': float(track_id),
            'x': int(x1),
            'y': int(y1),
            'w': int(w),
            'h': int(h),
            'confidence': float(conf)
        }
        tracked_objects.append(obj)
    
    # add to output data
    output_data.append({
        'ID': frame_count-1,
        'Frame': float(frame_count),
        'Objects': tracked_objects,
        'Objective': 'tracking'
    })
    
    # # visualize
    # visualization_frame = frame.copy()
    # for obj in tracked_objects:
    #     x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
    #     track_id = obj['tracked_id']
    #     cv2.rectangle(visualization_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(visualization_frame, f"ID: {int(track_id)}", (x, y-10), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # # display
    # cv2.imshow('Tracking', visualization_frame)
    # cv2.waitKey(1)
    
    return process_time



def track_persons_in_image_sequence(image_folder, output_file, detector_type=9, mode="train"):

    import glob
    
    # get all image files
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg'))) + \
                 sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    # initialize detector
    detector = load_detector(detector_type)
    
    # initialize tracker
    folder_name = os.path.basename(os.path.normpath(image_folder))
    tracker = BoostTrack(video_name=folder_name)
    
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
        
        process_time = process_frame(frame, frame_count, detector, tracker, output_data, detector_type, tag)
        total_time += process_time
    
    tracker.dump_cache()

    original_submission_file = os.path.join(CSV_FILES_PATH, "submission_file.csv")
    output_submission_file   = os.path.join(CSV_FILES_PATH, output_file)
    
    generate_submission_file(output_data, original_submission_file, output_submission_file)
    
    print(f"Tracking completed: {frame_count} images processed")
    print(f"Time spent: {total_time:.3f}s, FPS: {frame_count / (total_time + 1e-9):.2f}")
    
    return output_data



def generate_submission_file(output_data, original_file, output_file):
    df_original = pd.read_csv(original_file, dtype=str)

    # frame → objects
    tracking_data = {int(item["Frame"]): item["Objects"] for item in output_data}

    for i in range(len(df_original)):
        frame = int(float(df_original.loc[i, "frame"]))

        if frame in tracking_data:
            df_original.at[i, "objects"] = json.dumps(tracking_data[frame], separators=(',', ':'))
            df_original.at[i, "objective"] = "tracking"

    df_original.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Submission file saved successfully to {output_file}")

def merge_results(submission_file, tracking_csv_file, face_reid_csv_file, output_csv_file):
    df_submission = pd.read_csv(submission_file, dtype=str)
    df_tracking = pd.read_csv(tracking_csv_file, dtype=str)
    df_face_reid = pd.read_csv(face_reid_csv_file, dtype=str)
    
    assert len(df_submission) == len(df_tracking) == len(df_face_reid), "CSV files must have the same length"
    
    df_merged = df_submission.copy()
    
    df_merged.loc[df_tracking['objective'] == 'tracking', 'objects'] = df_tracking['objects']
    df_merged.loc[df_tracking['objective'] == 'tracking', 'objective'] = 'tracking'
    
    df_merged.loc[df_face_reid['objective'] == 'face_reid', 'objects'] = df_face_reid['objects']
    df_merged.loc[df_face_reid['objective'] == 'face_reid', 'objective'] = 'face_reid'
    
    df_merged.to_csv(output_csv_file, index=False, encoding='utf-8')
    print(f"Merged file saved successfully to {output_csv_file}")


def setup_tracker_settings(use_embedding=True, reid_type='fastreid', num_classes=2012, reid_path=None, use_ecc=True, use_rich_s=True, use_sb=True, use_vt=True):

    GeneralSettings.values['use_embedding'] = use_embedding         # reid for visual embedding
    GeneralSettings.values['reid_type'] = reid_type                 # "fastreid" or "our_trained_osnet" or None for generalized osnet
    GeneralSettings.values['num_classes'] = num_classes             # number of classes for reid
    GeneralSettings.values['reid_path'] = reid_path                 # use our pretrained weights
    GeneralSettings.values['use_ecc'] = use_ecc                     # camera motion compensation
    BoostTrackPlusPlusSettings.values['use_rich_s'] = use_rich_s    # use rich similarity (not just IoU)
    BoostTrackPlusPlusSettings.values['use_sb'] = use_sb            # use soft detection confidence boost
    BoostTrackPlusPlusSettings.values['use_vt'] = use_vt            # use varying threshold


def load_ground_truth(gt_file): # Load ground truth data ==> MOT format
    
    # Input:    Lines of (Frame, Track ID, X, Y, Width, Height, Confidence, Class, Visibility)
    # Output:   dict(Frame: ground truth)
    
    gt_data = defaultdict(list)
    
    try:
        # Try to read with pandas
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
        # Manual Parse
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


def visualize_tracking(image_folder, tracking_results, mode="train", output_video_path="tracking_results.mp4", ground_truth=None, iou_threshold=0.5):
    
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
        
        frame_num = i + 1

        if mode != "test":  # test data does not have ground truth

            # draw ground truth boxes in green
            if frame_num in ground_truth:
                for gt_obj in ground_truth[frame_num]:
                    x, y = int(gt_obj['x']), int(gt_obj['y'])
                    w, h = int(gt_obj['width']), int(gt_obj['height'])
                    track_id = gt_obj['track_id']
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"GT-{track_id}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # draw tracking results boxes in red
        if frame_num in tracking_by_frame:
            for track_obj in tracking_by_frame[frame_num]:
                x, y = track_obj['x'], track_obj['y']
                w, h = track_obj['w'], track_obj['h']
                track_id = int(track_obj['tracked_id'])
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"T-{track_id}", (x, y+h+15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # write frame to video
        video_writer.write(frame)
        
        # display the frame
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Visualization saved to {output_video_path}")

def convert_to_mot_format(df, output_file):
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            frame_id = row["frame"]
            for obj in row["objects"]:
                tracked_id = int(obj["tracked_id"])
                x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
                confidence = obj.get("confidence", 1.0)
                f.write(f"{frame_id},{tracked_id},{x},{y},{w},{h},{confidence},-1,-1,-1\n")


def evaluate_tracking_results(tracking_results_file, gt_file=None, iou_threshold=0.5):
    results_file_csv = os.path.join(CSV_FILES_PATH, tracking_results_file)
    tracking_results = pd.read_csv(results_file_csv, dtype=str)
    df_tracking = tracking_results[tracking_results["objective"] == "tracking"]
    df_tracking.loc[:, "objects"] = df_tracking["objects"].apply(json.loads)

    TRACK_EVAL_PATH = os.path.join(CURRENT_PATH, 'external', 'TrackEval')
    TRACKER_OUTPUT_PATH = os.path.join(TRACK_EVAL_PATH, 'data', 'trackers', 'mot_challenge', 'MOT20-train', 'ABN_Track', 'data')
    output_file_path = os.path.join(TRACKER_OUTPUT_PATH, 'MOT20-01.txt')

    print(f"Converting tracking results to MOT format: {output_file_path}")
    convert_to_mot_format(df_tracking, output_file_path)
        

def test(detector_type=9):

    # paths
    mode = "test"
    images_path = os.path.join(TEST_DATA_PATH, "img1")
    output_file = f"{mode}_tracking_results.csv"
    
    # track
    print(f"Running tracking with {detector_type}...")
    tracking_results = track_persons_in_image_sequence(images_path, output_file, detector_type, mode=mode)

    # visualize results
    # print("Generating visualization...")
    # output_video_path = f"{mode}_tracking.mp4"
    # visualize_tracking(images_path, tracking_results, mode=mode, output_video_path=output_video_path)

    evaluate_tracking_results(output_file)

if __name__ == "__main__":

    detector_type = 'pretrained_on_MOT20'               # 8, 9, 11, 'ours', 'pretrained_on_MOT20'
    reid_type =  'our_trained_osnet'                             # 'fastreid', 'our_trained_osnet',  None for generalized osnet

    # general osnet
    # reid_path = None                                      # the generalized models by torch reid will be used
    
    # trained osnet
    reid_path = '/home/floubsy2423/Downloads/model.pth'
    osnet_num_classes = 2011
    
    # fastreid
    # reid_path = '/home/floubsy2423/Downloads/model_final.pth'


    setup_tracker_settings(
        use_embedding=True,
        reid_type=reid_type,
        num_classes=osnet_num_classes,
        reid_path=reid_path,
        use_ecc=True,
        use_rich_s=True,
        use_sb=True,
        use_vt=True,
    )
    
    test(detector_type)                     # test and generate a submission file

    submission_file = os.path.join(CSV_FILES_PATH, "submission_file.csv")
    tracking_csv_file = os.path.join(CSV_FILES_PATH, "tracking.csv")
    face_reid_csv_file = os.path.join(CSV_FILES_PATH, "fast_reid.csv")
    output_csv_file = os.path.join(CSV_FILES_PATH, "final_submission.csv")

    merge_results(submission_file, tracking_csv_file, face_reid_csv_file, output_csv_file)

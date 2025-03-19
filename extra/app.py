from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from sort import Sort

def process_video(input_path, output_path, model_path=None, conf_threshold=0.25):
    # Load YOLO model
    if model_path:
        model_yolo = YOLO(model_path)
        print(f"Loaded custom model from {model_path}")
    else:
        model_yolo = YOLO('yolov8n.pt')
        print("Loaded default YOLOv8n model")
    
    # Initialize SORT tracker
    mot_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Open video file
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise ValueError(f"Error: Could not open video at {input_path}")
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        # YOLO detection
        results = model_yolo(frame, conf=conf_threshold)
        
        # Prepare detections for SORT
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            detections = np.hstack((boxes, scores.reshape(-1, 1)))
        
        # Update SORT tracker
        tracked_objects = mot_tracker.update(detections if len(detections) > 0 else np.empty((0, 5)))
        
        # Process each tracked object
        annotated_frame = frame.copy()
        for obj in tracked_objects:
            obj = obj.astype(int)
            x1, y1, x2, y2, track_id = obj
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID
            text = f"ID: {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1 - 10), 
                         (0, 0, 0), -1)
            
            cv2.putText(annotated_frame, text, 
                       (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 2)
        
        # Write output frame
        output_video.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time
            remaining_frames = total_frames - frame_count
            estimated_time = remaining_frames / frames_per_second if frames_per_second > 0 else 0
            
            print(f"Progress: {frame_count}/{total_frames} frames "
                  f"({(frame_count/total_frames*100):.1f}%) | "
                  f"Processing speed: {frames_per_second:.1f} FPS | "
                  f"Est. time remaining: {estimated_time:.1f} seconds")
    
    # Release resources
    video.release()
    output_video.release()
    
    print(f"\nVideo processing complete. Output saved to {output_path}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

def main():
    input_video = "test_video.mp4"
    output_video = "output.mp4"
    custom_model_path = "license_plate_detector.pt"
    
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        return
    
    try:
        process_video(
            input_path=input_video,
            output_path=output_video,
            model_path=custom_model_path,
            conf_threshold=0.3
        )
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
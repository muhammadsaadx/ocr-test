from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from PIL import Image
from sort import Sort
import cv2
import os
import logging
import numpy as np
import time
import pandas as pd

# Configure logging
logging.getLogger("transformers").setLevel(logging.ERROR)
Image.warnings.simplefilter('ignore')

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union != 0 else 0

class OCRTracker:
    def __init__(self):
        # Initialize models
        self.trocr_processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-base-printed', 
            cache_dir=os.environ.get("MODEL_CACHE")
        )
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-printed',
            cache_dir=os.environ.get("MODEL_CACHE")
        )
        self.yolo_model = YOLO('license_plate_detector.pt')
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Tracking state and data storage
        self.id_to_ocr = {}
        self.ocr_to_id = {}  # Maps OCR text to a unique display ID
        self.next_ocr_id = 1  # Next available ID for new OCR texts
        self.frame_count = 0
        self.start_time = time.time()
        self.tracking_data = []
        self.all_tracking_data = []  # Store all detections including frame information
        self.results_df = None
        self.all_results_df = None  # New dataframe for all detections

    def perform_ocr(self, cropped_image):
        """Perform OCR on a cropped license plate image"""
        try:
            pixel_values = self.trocr_processor(
                cropped_image, return_tensors="pt"
            ).pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            return self.trocr_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return None

    def process_frame(self, frame):
        """Process a single video frame"""
        # Detect license plates
        results = self.yolo_model(frame, conf=0.3)
        detections = []
        ocr_texts = []

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cropped_img = frame[y1:y2, x1:x2]
                
                if cropped_img.size == 0:
                    continue
                
                # Perform OCR
                pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                ocr_text = self.perform_ocr(pil_img)
                ocr_texts.append(ocr_text if ocr_text else "Unknown")
                detections.append([x1, y1, x2, y2, score])

        # Update tracker
        detections_np = np.array(detections) if detections else np.empty((0, 5))
        tracked_objects = self.tracker.update(detections_np)

        # Update OCR mappings
        current_ocr_mapping = {}
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            current_box = (x1, y1, x2, y2)
            
            best_iou = 0
            best_idx = -1
            for idx, det in enumerate(detections_np):
                det_box = det[:4].astype(int)
                iou = calculate_iou(current_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx != -1 and best_iou > 0.5:
                current_ocr_mapping[track_id] = ocr_texts[best_idx]

        # Update global OCR mapping
        for track_id, ocr_text in current_ocr_mapping.items():
            self.id_to_ocr[track_id] = ocr_text

        return tracked_objects

    def draw_annotations(self, frame, tracked_objects):
        """Draw tracking annotations on the frame"""
        annotated_frame = frame.copy()
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            ocr_text = self.id_to_ocr.get(track_id, "Unknown")

            # Get or assign display ID based on OCR text
            if ocr_text in self.ocr_to_id:
                display_id = self.ocr_to_id[ocr_text]
            else:
                display_id = self.next_ocr_id
                self.ocr_to_id[ocr_text] = display_id
                self.next_ocr_id += 1

            # Create combined display text
            display_text = f"ID {display_id}: {ocr_text}"

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1 - 10),
                (0, 0, 0), -1
            )

            # Put combined text
            cv2.putText(
                annotated_frame, display_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

        return annotated_frame

    def process_video(self, input_path, output_path):
        """Process complete video"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            tracked_objects = self.process_frame(frame)
            annotated_frame = self.draw_annotations(frame, tracked_objects)
            out.write(annotated_frame)

            # Collect tracking data
            current_time = self.frame_count / fps
            frame_number = self.frame_count
            
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, obj)
                ocr_text = self.id_to_ocr.get(track_id, "Unknown")
                display_id = self.ocr_to_id.get(ocr_text, track_id)

                self.tracking_data.append({
                    'display_id': display_id,
                    'ocr_text': ocr_text,
                    'timestamp': current_time,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
                
                # Add to all_tracking_data with additional frame information
                self.all_tracking_data.append({
                    'display_id': display_id,
                    'track_id': track_id,
                    'ocr_text': ocr_text,
                    'frame_number': frame_number,
                    'timestamp': current_time,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })

            # Progress reporting
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_time
                processed_fps = self.frame_count / elapsed
                remaining = (total_frames - self.frame_count) / processed_fps
                print(f"Processed {self.frame_count}/{total_frames} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Remaining: {remaining:.1f}s")

        # Create dataframes
        self.results_df = pd.DataFrame(self.tracking_data)
        self.all_results_df = pd.DataFrame(self.all_tracking_data)
        
        # Cleanup
        cap.release()
        out.release()
        print(f"Processing complete. Output saved to {output_path}")

def main():
    # Configure environment
    os.makedirs(".model", exist_ok=True)
    os.environ["MODEL_CACHE"] = ".model"
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Process video
    tracker = OCRTracker()
    tracker.process_video("test_video.mp4", "output.mp4")
    
    # Save and display summarized results
    if tracker.results_df is not None and not tracker.results_df.empty:
        # Get most frequent entry per display_id
        tracker.results_df = tracker.results_df.groupby('display_id').agg({
            'ocr_text': lambda x: x.mode()[0],
            'timestamp': 'count'  # Count occurrences
        }).rename(columns={'timestamp': 'count'}).sort_values('count', ascending=False)
        tracker.results_df = tracker.results_df.reset_index()[['display_id', 'ocr_text']]
        tracker.results_df.to_csv('tracking_results.csv', index=False)
        
        print("\nTracking results summary:")
        print(tracker.results_df)
        print(f"\nSaved results to tracking_results.csv")
    
    # Save and display all tracking results
    if tracker.all_results_df is not None and not tracker.all_results_df.empty:
        # Save all tracking results to CSV
        tracker.all_results_df.to_csv('all_tracking_results.csv', index=False)
        
        # Create a version with just the IDs and OCR texts
        all_tracking_summary = tracker.all_results_df[['display_id', 'track_id', 'ocr_text', 'frame_number', 'timestamp']]
        print("\nAll tracking results sample (first 5 rows):")
        print(all_tracking_summary.head())
        print(f"\nTotal detections: {len(tracker.all_results_df)}")
        print(f"\nSaved all results to all_tracking_results.csv")

if __name__ == "__main__":
    main()
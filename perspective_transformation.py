import numpy as np
import supervision as sv
from inference import get_model

def initialize_field_detection_model(api_key, model_id):
    """Initialize the field detection model using Roboflow's API."""
    return get_model(model_id=model_id, api_key=api_key)

def detect_field_keypoints(model, frame, confidence_threshold=0.5):
    """Detect field keypoints in a frame using the provided model."""
    result = model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)
    
    # Filter out low confidence detections
    filter_mask = key_points.confidence[0] > confidence_threshold
    frame_reference_points = key_points.xy[0][filter_mask]
    pitch_indices = np.where(filter_mask)[0]
    
    return frame_reference_points, pitch_indices
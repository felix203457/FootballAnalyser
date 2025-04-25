import pickle
import cv2
import numpy as np
import os

class CameraMovementEstimator:
    def __init__(self, frame):
        # Parameters for optical flow
        self.lk_params = dict(
            winSize = (21, 21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01)
        )
        
        # Get frame dimensions
        self.height, self.width = frame.shape[:2]
        self.frame_center = (self.width // 2, self.height // 2)
        
        # Create first frame grayscale
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create mask for feature detection
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:80] = 1
        mask_features[:, self.width-80:] = 1
        mask_features[0:80, :] = 1
        mask_features[self.height-80:, :] = 1
        
        # Feature detection parameters
        self.features = dict(
            maxCorners = 300,
            qualityLevel = 0.25,
            minDistance = 10,
            blockSize = 7,
            mask = mask_features
        )
        
        # Noise filtering parameters
        self.dead_zone = 2.0           # Dead zone for tiny movements (pixels)
        self.stability_count = 0       # Counter for stable frames
        self.stable_threshold = 3      # Reduced threshold - only need 3 stable frames
        self.is_stable = False         # Explicit stability flag
        
        # Refresh rate (in frames)
        self.refresh_rate = 30
        
        # Smoothing factor
        self.alpha = 0.5
        
        # Buffer for movement history
        self.movement_history_x = []
        self.movement_history_y = []
        self.history_size = 5
    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """Adjust position data to account for camera movement"""
        accumulated_x = 0
        accumulated_y = 0
        
        for frame_num, movement_data in enumerate(camera_movement_per_frame):
            # Unpack movement data - if it's a tuple or list with 3 elements
            if isinstance(movement_data, (list, tuple)) and len(movement_data) >= 3:
                mov_x, mov_y, _ = movement_data
            else:
                # Handle old format or unexpected data
                mov_x, mov_y = movement_data[0], movement_data[1]
            
            # Accumulate camera movement
            accumulated_x += mov_x
            accumulated_y += mov_y
            
            for object_type, object_tracks in tracks.items():
                if frame_num < len(object_tracks):
                    for track_id, track_info in object_tracks[frame_num].items():
                        if 'position' in track_info:
                            position = track_info['position']
                            position_adjusted = (
                                position[0] - accumulated_x,
                                position[1] - accumulated_y
                            )
                            tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted
    
    def _filter_movement(self, movement_x, movement_y):
        """Filter out noise in detected movement"""
        # Add to history
        self.movement_history_x.append(movement_x)
        self.movement_history_y.append(movement_y)
        
        # Keep history at specified size
        if len(self.movement_history_x) > self.history_size:
            self.movement_history_x.pop(0)
            self.movement_history_y.pop(0)
        
        # Calculate movement magnitude
        magnitude = np.sqrt(movement_x**2 + movement_y**2)
        
        # Apply dead zone for tiny movements
        if magnitude < self.dead_zone:
            # If movement is small, increment stability counter
            self.stability_count += 1
            if self.stability_count >= self.stable_threshold:
                # Mark as stable when we have several consecutive stable frames
                self.is_stable = True
                return 0, 0  # Return exactly zero movement when stable
        else:
            # Reset stability counter if we detect significant movement
            self.stability_count = 0
            self.is_stable = False
        
        # Return filtered movement values
        if self.is_stable:
            return 0, 0
        else:
            return movement_x, movement_y
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """Calculate camera movement with improved noise filtering"""
        # Try to read from stub if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        # Initialize camera movement array and stability states array
        camera_movement = []
        stability_states = []
        
        # Reset movement history and stability tracking
        self.movement_history_x = []
        self.movement_history_y = []
        self.stability_count = 0
        self.is_stable = False
        
        # Convert first frame to grayscale
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Detect features in first frame
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        # Add initial frame with zero movement
        camera_movement.append([0, 0, 1.0])
        stability_states.append(True)  # First frame is always "stable"
        
        # Handle case of no features
        if old_features is None:
            # Fill with zeros and return
            camera_movement = [[0, 0, 1.0]] * len(frames)
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(camera_movement, f)
            return camera_movement
        
        frames_since_refresh = 0
        
        # Process remaining frames
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            new_features, status, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )
            
            # Handle tracking failure
            if new_features is None or not np.any(status):
                camera_movement.append(camera_movement[-1])
                stability_states.append(stability_states[-1])
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray.copy()
                continue
            
            # Filter tracked features
            status_mask = status.ravel() == 1
            if not np.any(status_mask):
                camera_movement.append(camera_movement[-1])
                stability_states.append(stability_states[-1])
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray.copy()
                continue
            
            old_valid = old_features[status_mask]
            new_valid = new_features[status_mask]
            
            # Calculate displacements
            if len(old_valid) == 0:
                camera_movement.append(camera_movement[-1])
                stability_states.append(stability_states[-1])
                old_gray = frame_gray.copy()
                continue
            
            # Calculate displacements for all tracked features
            displacements = new_valid - old_valid
            
            # Get movement vector statistics
            dx_values = displacements[:, 0, 0]
            dy_values = displacements[:, 0, 1]
            
            # Use robust statistics
            median_dx = np.median(dx_values)
            median_dy = np.median(dy_values)
            
            # Calculate MAD for outlier detection
            mad_x = np.median(np.abs(dx_values - median_dx))
            mad_y = np.median(np.abs(dy_values - median_dy))
            
            # Filter outliers
            inlier_mask_x = np.abs(dx_values - median_dx) < 2.5 * mad_x
            inlier_mask_y = np.abs(dy_values - median_dy) < 2.5 * mad_y
            inlier_mask = inlier_mask_x & inlier_mask_y
            
            # Recalculate median using only inliers
            if np.sum(inlier_mask) > 10:
                camera_movement_x = np.median(dx_values[inlier_mask])
                camera_movement_y = np.median(dy_values[inlier_mask])
            else:
                camera_movement_x = median_dx
                camera_movement_y = median_dy
            
            # Apply noise filtering and get stability status
            filtered_x, filtered_y = self._filter_movement(camera_movement_x, camera_movement_y)
            
            # Apply temporal smoothing
            if len(camera_movement) > 0:
                filtered_x = self.alpha * filtered_x + (1-self.alpha) * camera_movement[-1][0]
                filtered_y = self.alpha * filtered_y + (1-self.alpha) * camera_movement[-1][1]
            
            # Store movement data and stability state
            camera_movement.append([filtered_x, filtered_y, 1.0])
            stability_states.append(self.is_stable)
            
            # Refresh features periodically
            frames_since_refresh += 1
            if frames_since_refresh >= self.refresh_rate:
                new_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                if new_features is not None:
                    old_features = new_features
                else:
                    old_features = new_valid
                frames_since_refresh = 0
            else:
                old_features = new_valid
            
            # Update old_gray for next iteration
            old_gray = frame_gray.copy()
        
        # Convert to structure expected by rest of code
        result = list(zip(camera_movement, stability_states))
        
        # Save to stub if path provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """Draw camera movement information on frames"""
        output_frames = []
        accumulated_x = 0
        accumulated_y = 0
        
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            # Create overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 120), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            
            # Get movement values
            if frame_num < len(camera_movement_per_frame):
                if len(camera_movement_per_frame[frame_num]) >= 3:
                    x_movement, y_movement, _ = camera_movement_per_frame[frame_num]
                else:
                    x_movement, y_movement = camera_movement_per_frame[frame_num][0], camera_movement_per_frame[frame_num][1]
            else:
                x_movement, y_movement = 0, 0
            
            # Accumulate movement
            accumulated_x += x_movement
            accumulated_y += y_movement
            
            # Calculate movement magnitude
            magnitude = np.sqrt(x_movement**2 + y_movement**2)
            
            # Determine if this frame is stable (exactly 0 movement indicates stability)
            is_stable = (magnitude < 0.01)
            status = "STABLE" if is_stable else "MOVING"
            color = (0, 128, 0) if is_stable else (0, 0, 200)
            
            # Draw movement text
            frame = cv2.putText(
                frame, 
                f"Camera Status: {status}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                color, 
                2
            )
            
            frame = cv2.putText(
                frame, 
                f"Movement X: {x_movement:.2f} px", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 0), 
                2
            )
            
            frame = cv2.putText(
                frame, 
                f"Movement Y: {y_movement:.2f} px", 
                (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 0), 
                2
            )
            
            # Add frame to output
            output_frames.append(frame)
            
        return output_frames
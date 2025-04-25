from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2

from video_utils import get_centre_of_bbox, get_foot_position, get_width_of_bbox


class Tracker:
    """
    A tracking class to handle object detection, tracking, annotation,
    and ball possession visualization for football match analysis.
    """

    def __init__(self, model_path):
        # Load detection model (YOLOv8) and ByteTrack tracker
        print("[INFO] Initializing detection model and tracker...")
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        print("[INFO] Model and tracker initialized successfully.")

    def detect_frames(self, frames):
        """
        Perform batch detection on input frames using the YOLO model.

        Args:
            frames (list): List of video frames (numpy arrays)

        Returns:
            List of detections for each frame
        """
        print("[INFO] Starting batch object detection...")
        batch_size = 20
        detections = []

        try:
            for i in range(0, len(frames), batch_size):
                print(f"[INFO] Detecting frames {i} to {i + batch_size}...")
                batch_result = self.model.predict(frames[i:i + batch_size], imgsz=1280, device=0)
                detections += batch_result
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")

        print("[INFO] Detection complete.")
        return detections

    def add_position_to_tracks(self, tracks):
        """
        Add positional data (center or foot) to each tracked object.
        """
        print("[INFO] Adding positional coordinates to tracked objects...")
        try:
            for obj_type, obj_tracks in tracks.items():
                for frame_idx, track in enumerate(obj_tracks):
                    for track_id, info in track.items():
                        bbox = info['bbox']
                        if obj_type == 'ball':
                            position = get_centre_of_bbox(bbox)
                        else:
                            position = get_foot_position(bbox)
                        tracks[obj_type][frame_idx][track_id]['position'] = position
        except Exception as e:
            print(f"[ERROR] Failed to add positions: {e}")

    def interpolate_ball_positions(self, ball_positions):
        """
        Smooth and fill gaps in ball positions using cubic interpolation.
        """
        print("[INFO] Interpolating missing ball positions...")
        try:
            ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
            df_ball = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
            df_ball = df_ball.interpolate(method="cubic", limit_direction="both").bfill()
            return [{1: {"bbox": bbox}} for bbox in df_ball.to_numpy().tolist()]
        except Exception as e:
            print(f"[ERROR] Interpolation failed: {e}")
            return ball_positions

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Main tracking pipeline: detects objects, tracks them, and categorizes them.

        Returns:
            dict: Tracked objects separated by type (players, ball, etc.)
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[INFO] Loading tracks from cached stub at: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        print("[INFO] Running detection and tracking pipeline...")
        detections = self.detect_frames(frames)
        tracks = {"players": [], "goalkeepers": [], "referees": [], "ball": []}

        try:
            for frame_idx, detection in enumerate(detections):
                print(f"[INFO] Processing frame {frame_idx + 1}/{len(detections)}...")
                cls_names = detection.names
                cls_ids = {v: k for k, v in cls_names.items()}

                supervision_det = sv.Detections.from_ultralytics(detection)
                tracked_objects = self.tracker.update_with_detections(supervision_det)

                # Initialize frame containers
                for key in tracks:
                    tracks[key].append({})

                for obj in tracked_objects:
                    bbox = obj[0].tolist()
                    class_id = obj[3]
                    track_id = obj[4]

                    if class_id == cls_ids['player']:
                        tracks["players"][frame_idx][track_id] = {"bbox": bbox}
                    elif class_id == cls_ids['goalkeeper']:
                        tracks["goalkeepers"][frame_idx][track_id] = {"bbox": bbox}
                    elif class_id == cls_ids['referee']:
                        tracks["referees"][frame_idx][track_id] = {"bbox": bbox}

                for obj in supervision_det:
                    bbox = obj[0].tolist()
                    class_id = obj[3]
                    if class_id == cls_ids['ball']:
                        tracks["ball"][frame_idx][1] = {"bbox": bbox}
        except Exception as e:
            print(f"[ERROR] Tracking failed: {e}")

        if stub_path:
            print(f"[INFO] Saving tracks to stub file: {stub_path}")
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None, ref=False):
        """
        Draw an ellipse and optional ID tag below a bounding box.
        """
        y2 = int(bbox[3])
        x_center, _ = get_centre_of_bbox(bbox)
        width = get_width_of_bbox(bbox)

        # Draw ellipse around the base of the player
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw ID rectangle below ellipse
        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = x_center - rect_w // 2
            x2_rect = x_center + rect_w // 2
            y1_rect = (y2 - rect_h // 2) + 15
            y2_rect = (y2 + rect_h // 2) + 15

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            label = "Ref" if ref else str(track_id)
            offset = 12 if not ref else 10
            if isinstance(track_id, int) and track_id > 99:
                offset -= 10

            cv2.putText(
                frame, label,
                (int(x1_rect + offset), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
        return frame

    def draw_triangles(self, frame, bbox, color):
        """
        Draw a small triangle above a bounding box to indicate ball possession.
        """
        y = int(bbox[1])
        x, _ = get_centre_of_bbox(bbox)
        triangle = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_idx, control_array):
        """
        Display a real-time ball possession percentage overlay on the frame.
        """
        try:
            overlay = frame.copy()
            cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            control_history = control_array[:frame_idx + 1]
            team_1_frames = np.sum(control_history == 1)
            team_2_frames = np.sum(control_history == 2)

            total_frames = team_1_frames + team_2_frames
            team_1_pct = (team_1_frames / total_frames) * 100 if total_frames else 0
            team_2_pct = (team_2_frames / total_frames) * 100 if total_frames else 0

            cv2.putText(frame, f"Team 1 Ball Control: {team_1_pct:.2f}%", (1400, 900),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Team 2 Ball Control: {team_2_pct:.2f}%", (1400, 950),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        except Exception as e:
            print(f"[ERROR] Failed to draw team control overlay: {e}")
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Annotate video frames with tracking info, ball possession, and labels.

        Returns:
            List of annotated frames
        """
        print("[INFO] Starting annotation of video frames...")
        output_frames = []

        try:
            for idx, frame in enumerate(video_frames):
                frame = frame.copy()

                players = tracks["players"][idx]
                goalkeepers = tracks["goalkeepers"][idx]
                referees = tracks["referees"][idx]
                balls = tracks["ball"][idx]

                for pid, pdata in players.items():
                    color = pdata.get("team_color", (0, 0, 255))
                    frame = self.draw_ellipse(frame, pdata["bbox"], color, pid)
                    if pdata.get("has_ball", False):
                        frame = self.draw_triangles(frame, pdata["bbox"], (0, 0, 255))

                for gid, gdata in goalkeepers.items():
                    color = gdata.get("team_color", (0, 255, 255))
                    frame = self.draw_ellipse(frame, gdata["bbox"], color, gid)

                for _, rdata in referees.items():
                    frame = self.draw_ellipse(frame, rdata["bbox"], (0, 0, 100), ref=True)

                for _, bdata in balls.items():
                    frame = self.draw_triangles(frame, bdata["bbox"], (0, 0, 255))

                frame = self.draw_team_ball_control(frame, idx, team_ball_control)
                output_frames.append(frame)
        except Exception as e:
            print(f"[ERROR] Annotation failed on frame {idx}: {e}")

        print("[INFO] Annotation complete.")
        return output_frames

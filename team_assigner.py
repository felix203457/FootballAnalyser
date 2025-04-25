from typing import Dict, List
import numpy as np
import cv2
import torch
from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans
from collections import Counter

class TeamAssigner:
    def __init__(self, device='cpu', batch_size=16, num_frames=90):
        """
        Initialize the TeamAssigner with simple multi-frame analysis.

        Args:
            device (str): The device to run the model on ('cpu' or 'cuda').
            batch_size (int): The batch size for processing images.
            num_frames (int): Number of frames to analyze (3 seconds at 30fps = 90 frames)
        """
        self.device = device
        self.batch_size = batch_size
        self.num_frames = num_frames

        self.team_colors = {}
        self.player_team_dict = {}

        # Load the pre-trained vision model
        self.features_model = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224').to(device)
        self.processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')

        # Initialize UMAP for dimensionality reduction and KMeans for clustering
        self.reducer = umap.UMAP(n_components=3)
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)

        # Flag to indicate if initialization is complete
        self.is_initialized = False

    def _extract_features(self, crops):
        """Extract features from a list of image crops using the pre-trained model"""
        if not crops:
            return np.array([])

        # Convert OpenCV images to PIL format
        pil_crops = []
        for crop in crops:
            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crops.append(rgb_crop)

        # Process crops in batches
        features = []
        for i in range(0, len(pil_crops), self.batch_size):
            batch = pil_crops[i:i+self.batch_size]

            with torch.no_grad():
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                batch_features = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                features.append(batch_features)

        return np.vstack(features) if features else np.array([])

    def _enhance_color(self, color):
        """
        Enhance a color to make it more vibrant and bright.

        Args:
            color: A BGR color tuple (B, G, R)

        Returns:
            Enhanced BGR color tuple
        """
        # Convert to numpy array for processing
        color_array = np.array([[color]], dtype=np.uint8)

        # Convert BGR to HSV
        hsv_color = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)[0][0]

        # Get HSV components
        h, s, v = hsv_color

        # Check if this is close to white (low saturation, high value)
        if s < 50 and v > 150:
            # For white/light colors, make pure white
            return (255, 255, 255)

        # Check if this is close to black (low value)
        if v < 70:
            # For dark colors, just brighten but keep hue
            v = 200
            s = min(255, s + 100)
        else:
            # For colored uniforms, enhance saturation and value
            s = min(255, s + 100)  # Increase saturation
            v = min(255, v + 50)   # Increase brightness

        # Create enhanced HSV color
        enhanced_hsv = np.array([[[h, s, v]]], dtype=np.uint8)

        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)[0][0]

        # Ensure integer values
        return tuple(map(int, enhanced_bgr))

    def _calculate_team_color(self, crops):
        """Calculate the average color for a team based on player crops"""
        if not crops:
            return (0, 0, 255)  # Default to red

        # Flatten all crops and calculate mean color
        all_pixels = np.vstack([crop.reshape(-1, 3) for crop in crops])
        mean_color = np.mean(all_pixels, axis=0).astype(int)

        # Convert to BGR tuple
        bgr_color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

        # Enhance the color to make it more vibrant
        enhanced_color = self._enhance_color(bgr_color)

        return enhanced_color

    def initialize_from_multiple_frames(self, video_frames, all_tracks):
        """
        Initialize team assignments using multiple frames.

        Args:
            video_frames: List of video frames to analyze
            all_tracks: List of player tracking data for each frame
        """
        print(f"Initializing team assignments using {min(self.num_frames, len(video_frames))} frames...")

        # Store player crops and IDs from multiple frames
        all_player_crops = {}  # player_id -> list of crops
        frame_count = min(self.num_frames, len(video_frames))

        # Collect crops for each player across frames
        for frame_idx in range(frame_count):
            frame = video_frames[frame_idx]
            players = all_tracks["players"][frame_idx]

            for player_id, player_data in players.items():
                bbox = player_data["bbox"]

                # Extract crop
                crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                # Skip tiny crops
                if crop.shape[0] <= 1 or crop.shape[1] <= 1:
                    continue

                if player_id not in all_player_crops:
                    all_player_crops[player_id] = []

                all_player_crops[player_id].append(crop)

        # Process players with enough samples
        valid_player_ids = []
        valid_features = []

        for player_id, crops in all_player_crops.items():
            # Only use players that appear in multiple frames
            if len(crops) < 3:
                continue

            # Use a random sample of crops (up to 5) for efficiency
            sample_size = min(5, len(crops))
            sample_indices = np.random.choice(len(crops), sample_size, replace=False)
            sample_crops = [crops[i] for i in sample_indices]

            # Extract features
            features = self._extract_features(sample_crops)

            if len(features) == 0:
                continue

            # Use median feature for stability
            median_feature = np.median(features, axis=0)
            valid_features.append(median_feature)
            valid_player_ids.append(player_id)

        if len(valid_features) < 2:
            print("Not enough valid players for team assignment")
            self.team_colors[1] = (0, 0, 255)  # Red
            self.team_colors[2] = (0, 255, 0)  # Green
            return

        # Reduce dimensions and cluster
        reduced_features = self.reducer.fit_transform(np.array(valid_features))
        cluster_labels = self.kmeans.fit_predict(reduced_features)

        # Create initial team assignments
        for i, player_id in enumerate(valid_player_ids):
            team_id = cluster_labels[i] + 1  # Convert 0/1 to 1/2
            self.player_team_dict[player_id] = team_id

        # Assign team colors
        team1_crops = []
        team2_crops = []

        for player_id in valid_player_ids:
            if player_id in self.player_team_dict:
                if self.player_team_dict[player_id] == 1:
                    team1_crops.extend(all_player_crops[player_id])
                else:
                    team2_crops.extend(all_player_crops[player_id])

        self.team_colors[1] = self._calculate_team_color(team1_crops)
        self.team_colors[2] = self._calculate_team_color(team2_crops)

        # Use consistent assignments from frame-by-frame analysis
        self._refine_team_assignments(video_frames, all_tracks, frame_count)

        self.is_initialized = True
        print("Team assignment initialization complete.")

    def _refine_team_assignments(self, video_frames, all_tracks, frame_count):
        """Refine team assignments by analyzing each player across frames"""
        player_frame_teams = {}  # player_id -> list of team assignments

        # Collect team predictions for each player in each frame
        for frame_idx in range(frame_count):
            frame = video_frames[frame_idx]
            players = all_tracks["players"][frame_idx]

            for player_id, player_data in players.items():
                if player_id not in player_frame_teams:
                    player_frame_teams[player_id] = []

                bbox = player_data["bbox"]
                crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                # Skip tiny crops
                if crop.shape[0] <= 1 or crop.shape[1] <= 1:
                    continue

                features = self._extract_features([crop])

                if len(features) == 0:
                    continue

                reduced_features = self.reducer.transform(features)
                predicted_team = self.kmeans.predict(reduced_features)[0] + 1

                player_frame_teams[player_id].append(predicted_team)

        # Determine most frequent team assignment for each player
        for player_id, team_predictions in player_frame_teams.items():
            if not team_predictions:
                continue

            # Get most common team assignment
            team_counts = Counter(team_predictions)
            most_common_team = team_counts.most_common(1)[0][0]

            # Update player team dictionary
            self.player_team_dict[player_id] = most_common_team

    def assign_team_color(self, frame, player_detections):
        """Legacy method for compatibility; team assignments require initialization first"""
        if not self.is_initialized:
            print("Warning: Team assignments not initialized. Use initialize_from_multiple_frames first.")
            # Temporary assignment
            self.team_colors[1] = (0, 0, 255)  # Red
            self.team_colors[2] = (0, 255, 0)  # Green

    def get_player_team(self, frame, player_bbox, player_id):
        """Get team assignment for a player"""
        # If already assigned, return the existing assignment
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # For new players, assign team based on feature similarity
        crop = frame[int(player_bbox[1]):int(player_bbox[3]), int(player_bbox[0]):int(player_bbox[2])]

        # Skip tiny crops
        if crop.shape[0] <= 1 or crop.shape[1] <= 1:
            return 1  # Default team

        features = self._extract_features([crop])

        if len(features) == 0:
            return 1  # Default team

        # Use trained models to predict team
        try:
            reduced_features = self.reducer.transform(features)
            team_id = self.kmeans.predict(reduced_features)[0] + 1  # Convert 0/1 to 1/2

            # Store for future reference
            self.player_team_dict[player_id] = team_id

            return team_id
        except Exception as e:
            print(f"Error predicting team: {e}")
            return 1  # Default team
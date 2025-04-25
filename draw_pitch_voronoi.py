import cv2
import numpy as np
import supervision as sv
from pitch_config import SoccerPitchConfiguration
from draw_pitch_voronoi import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram
)

class VoronoiDiagramGenerator:
    def __init__(self, config, padding=50, scale=0.1):
        """
        Initialize a Voronoi diagram visualizer for football match analysis.

        Args:
            config: Configuration object defining the soccer pitch dimensions
            padding: Additional space (in pixels) around the pitch boundaries
            scale: Scale factor to resize the pitch (useful for high-res or low-res output)
        """
        self.config = config
        self.padding = padding
        self.scale = scale
        
    def generate_diagram(self, team1_coords, team2_coords, ball_coord=None):
        """
        Generates a Voronoi diagram over a soccer pitch, based on player positions.

        Args:
            team1_coords: (np.array) Pitch coordinates of Team 1 players (e.g., home team)
            team2_coords: (np.array) Pitch coordinates of Team 2 players (e.g., away team)
            ball_coord: (np.array or None) Coordinates of the ball, if available

        Returns:
            final_image: A rendered image of the pitch with Voronoi zones and annotations
        """

        # === Step 1: Create the blank pitch background with lines and boundaries ===
        base_pitch = draw_pitch(
            config=self.config,
            padding=self.padding,
            scale=self.scale
        )

        # === Step 2: Overlay Voronoi regions for both teams ===
        # This fills the pitch with colored cells representing each player's "area of control"
        final_image = draw_pitch_voronoi_diagram(
            config=self.config,
            team_1_xy=team1_coords,                         # Coordinates of players from Team 1
            team_2_xy=team2_coords,                         # Coordinates of players from Team 2
            team_1_color=sv.Color.from_hex('#1E90FF'),      # Dodger blue for Team 1 zones
            team_2_color=sv.Color.from_hex('#FF69B4'),      # Soft pink for Team 2 zones
            opacity=0.45,                                   # Slight transparency to see pitch lines beneath
            padding=self.padding,
            scale=self.scale,
            pitch=base_pitch                                # Base pitch to overlay onto
        )

        # === Step 3: Add Team 1 player markers (circles) on top of their Voronoi zones ===
        if team1_coords.size > 0:
            final_image = draw_points_on_pitch(
                config=self.config,
                xy=team1_coords,
                face_color=sv.Color.from_hex('#1E90FF'),     # Match the zone color
                edge_color=sv.Color.BLACK,                   # Add a black outline for visibility
                radius=16,                                   # Circle radius in pixels
                padding=self.padding,
                scale=self.scale,
                pitch=final_image                            # Draw on top of Voronoi diagram
            )

        # === Step 4: Add Team 2 player markers similarly ===
        if team2_coords.size > 0:
            final_image = draw_points_on_pitch(
                config=self.config,
                xy=team2_coords,
                face_color=sv.Color.from_hex('#FF69B4'),     # Match Team 2 color
                edge_color=sv.Color.BLACK,
                radius=16,
                padding=self.padding,
                scale=self.scale,
                pitch=final_image
            )

        # === Step 5: Add the ball to the pitch (if coordinates are provided) ===
        if ball_coord is not None and ball_coord.size > 0:
            final_image = draw_points_on_pitch(
                config=self.config,
                xy=ball_coord,
                face_color=sv.Color.from_hex('#F5F5F5'),     # Light gray/white ball color
                edge_color=sv.Color.BLACK,
                radius=10,                                   # Slightly smaller than players
                padding=self.padding,
                scale=self.scale,
                pitch=final_image
            )

        # Return the fully rendered image with all overlays
        return final_image

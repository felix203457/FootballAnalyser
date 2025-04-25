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
        Setup the VoronoiDiagramGenerator with custom pitch details.
        
        Args:
            config: Soccer pitch layout configuration
            padding: Extra spacing around pitch edges (in pixels)
            scale: Proportional scaling factor for pitch rendering
        """
        self.config = config
        self.padding = padding
        self.scale = scale
        
    def generate_diagram(self, team1_xy, team2_xy, ball_xy=None):
        """
        Create a visual Voronoi diagram overlaying player positions on a pitch.
        
        Args:
            team1_xy: Array of pitch coordinates for players of team 1
            team2_xy: Array of pitch coordinates for players of team 2
            ball_xy: Optional pitch coordinate for the ball
            
        Returns:
            A rendered image showing the Voronoi map and player locations
        """
        # Step 1: Draw the empty pitch
        base = draw_pitch(
            config=self.config,
            padding=self.padding,
            scale=self.scale
        )
        
        # Step 2: Apply Voronoi region coloring over the pitch
        output = draw_pitch_voronoi_diagram(
            config=self.config,
            team_1_xy=team1_xy,
            team_2_xy=team2_xy,
            team_1_color=sv.Color.from_hex('#00BFFF'),  # Cyan-like shade
            team_2_color=sv.Color.from_hex('#FF1493'),  # Bright pink
            opacity=0.5,
            padding=self.padding,
            scale=self.scale,
            pitch=base
        )
        
        # Step 3: Add circles for team 1 players
        if team1_xy.size > 0:
            output = draw_points_on_pitch(
                config=self.config,
                xy=team1_xy,
                face_color=sv.Color.from_hex('#00BFFF'),
                edge_color=sv.Color.BLACK,
                radius=16,
                padding=self.padding,
                scale=self.scale,
                pitch=output
            )
        
        # Step 4: Add circles for team 2 players
        if team2_xy.size > 0:
            output = draw_points_on_pitch(
                config=self.config,
                xy=team2_xy,
                face_color=sv.Color.from_hex('#FF1493'),
                edge_color=sv.Color.BLACK,
                radius=16,
                padding=self.padding,
                scale=self.scale,
                pitch=output
            )
        
        # Step 5: Add the ball marker, if position is provided
        if ball_xy is not None and ball_xy.size > 0:
            output = draw_points_on_pitch(
                config=self.config,
                xy=ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                padding=self.padding,
                scale=self.scale,
                pitch=output
            )
        
        return output

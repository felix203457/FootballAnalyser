import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class FieldTiltAnalyzer:
    def __init__(self, config, padding=50, scale=0.1):
        """
        Initialize the FieldTiltAnalyzer to calculate and visualize field tilt.
        
        Args:
            config: Soccer pitch configuration
            padding: Padding around the pitch in pixels
            scale: Scale factor for the pitch
        """
        self.config = config
        self.padding = padding
        self.scale = scale
        self.field_tilt_history = []
        self.frame_indices = []
        self.ball_possession_history = {1: [], 2: []}
        
    def calculate_field_tilt(self, team1_xy, team2_xy, pitch_dims=None):
        """
        Calculate field tilt based on the territorial control from Voronoi diagrams.
        
        Args:
            team1_xy: Positions of team 1 players in pitch coordinates
            team2_xy: Positions of team 2 players in pitch coordinates
            pitch_dims: Tuple of (width, length) of the pitch in cm
            
        Returns:
            Field tilt value between -100 (full team 2 dominance) and 100 (full team 1 dominance)
        """
        if pitch_dims is None:
            pitch_width = self.config.width
            pitch_length = self.config.length
        else:
            pitch_width, pitch_length = pitch_dims
            
        # Scale factors for importance weighting (attacking third matters more)
        attack_weight = 1.5
        middle_weight = 1.0
        defense_weight = 0.5
        
        # Create a grid of points covering the pitch
        grid_density = 100  # Number of points in each dimension
        x_points = np.linspace(0, pitch_length, grid_density)
        y_points = np.linspace(0, pitch_width, grid_density)
        xx, yy = np.meshgrid(x_points, y_points)
        grid_points = np.column_stack((xx.flatten(), yy.flatten()))
        
        # Skip calculation if either team has no players
        if len(team1_xy) == 0 or len(team2_xy) == 0:
            return 0.0
            
        # Calculate distances from each grid point to each player
        team1_control = np.zeros(len(grid_points), dtype=bool)
        team2_control = np.zeros(len(grid_points), dtype=bool)
        
        for i, point in enumerate(grid_points):
            dist_to_team1 = np.min([np.linalg.norm(point - player) for player in team1_xy])
            dist_to_team2 = np.min([np.linalg.norm(point - player) for player in team2_xy])
            
            if dist_to_team1 < dist_to_team2:
                team1_control[i] = True
            else:
                team2_control[i] = True
        
        # Apply weighting based on pitch thirds
        weights = np.ones(len(grid_points))
        
        # Team 1 attacks from left to right
        for i, point in enumerate(grid_points):
            x = point[0]
            # Attacking third for team 1
            if x > 2*pitch_length/3:
                weights[i] = attack_weight if team1_control[i] else defense_weight
            # Defending third for team 1
            elif x < pitch_length/3:
                weights[i] = defense_weight if team1_control[i] else attack_weight
            # Middle third
            else:
                weights[i] = middle_weight
        
        # Calculate weighted control
        team1_weighted = np.sum(weights * team1_control)
        team2_weighted = np.sum(weights * team2_control)
        
        total_weight = team1_weighted + team2_weighted
        
        # Calculate field tilt (-100 to 100)
        if total_weight > 0:
            field_tilt = 100 * (team1_weighted - team2_weighted) / total_weight
        else:
            field_tilt = 0.0
            
        return field_tilt
        
    def determine_ball_possession(self, ball_xy, team1_xy, team2_xy):
        """
        Determine ball possession based on proximity to players after coordinate transformation.
        
        Args:
            ball_xy: Position of the ball in pitch coordinates
            team1_xy: Positions of team 1 players in pitch coordinates
            team2_xy: Positions of team 2 players in pitch coordinates
            
        Returns:
            Team number (1 or 2) in possession of the ball, or previous possession if inconclusive
        """
        if len(ball_xy) == 0:
            # No ball detected, return previous possession or default
            return self.ball_possession_history[1][-1] if self.ball_possession_history[1] else 1
        
        ball_pos = ball_xy[0]  # Get the first (and only) ball position
        
        # Find closest player to ball
        min_dist = float('inf')
        possession_team = 0
        
        for player_pos in team1_xy:
            dist = np.linalg.norm(ball_pos - player_pos)
            if dist < min_dist:
                min_dist = dist
                possession_team = 1
                
        for player_pos in team2_xy:
            dist = np.linalg.norm(ball_pos - player_pos)
            if dist < min_dist:
                min_dist = dist
                possession_team = 2
        
        # Distance threshold for possession (700cm = 7 meters)
        if min_dist > 700:
            # Ball not close enough to any player, use previous possession
            possession_team = (self.ball_possession_history[1][-1] if self.ball_possession_history[1] else 1)
            
        return possession_team
    
    def update(self, frame_idx, team1_xy, team2_xy, ball_xy=None):
        """
        Update field tilt and ball possession history with new frame data.
        
        Args:
            frame_idx: Frame index or timestamp
            team1_xy: Positions of team 1 players in pitch coordinates
            team2_xy: Positions of team 2 players in pitch coordinates
            ball_xy: Position of the ball in pitch coordinates
        """
        # Calculate field tilt
        tilt = self.calculate_field_tilt(team1_xy, team2_xy)
        
        # Determine ball possession
        if ball_xy is not None:
            possession = self.determine_ball_possession(ball_xy, team1_xy, team2_xy)
        else:
            # Use previous possession or default
            possession = (self.ball_possession_history[1][-1] if self.ball_possession_history[1] else 1)
        
        # Update histories
        self.field_tilt_history.append(tilt)
        self.frame_indices.append(frame_idx)
        
        # Update possession counters
        for team in [1, 2]:
            if team == possession:
                self.ball_possession_history[team].append(1)
            else:
                self.ball_possession_history[team].append(0)
    
    def generate_field_tilt_chart(self, output_path, window_size=90):
        """
        Generate a field tilt chart visualization similar to the reference image.
        
        Args:
            output_path: Path to save the output image
            window_size: Window size for moving average smoothing
            
        Returns:
            Path to the saved chart image
        """
        # Apply moving average smoothing
        if len(self.field_tilt_history) > window_size:
            smoothed_tilt = np.convolve(
                self.field_tilt_history, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            # Pad the beginning to maintain the same length
            padding = np.ones(window_size-1) * self.field_tilt_history[0]
            smoothed_tilt = np.concatenate((padding, smoothed_tilt))
        else:
            smoothed_tilt = self.field_tilt_history
        
        # Create figure with specific size
        fig = Figure(figsize=(12, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Set up grid
        ax.grid(True, linestyle=':', alpha=0.7, color='#cccccc')
        ax.set_axisbelow(True)
        
        # Set limits
        ax.set_ylim(-100, 100)
        ax.set_xlim(0, len(self.frame_indices))
        
        # Set labels
        ax.set_ylabel('Field Tilt')
        ax.set_xlabel('Frame')
        
        # Create x-axis ticks
        max_frame = len(self.frame_indices)
        tick_step = max(1, max_frame // 6)  # Divide into 6 sections
        ax.set_xticks(np.arange(0, max_frame, tick_step))
        ax.set_xticklabels([f"{idx}" for idx in range(0, max_frame, tick_step)])
        
        # Create y-axis ticks
        ax.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
        
        # Plot center line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Create positive and negative areas
        positive_mask = smoothed_tilt >= 0
        negative_mask = smoothed_tilt < 0
        
        # Plot positive area (black)
        if any(positive_mask):
            ax.fill_between(
                range(len(smoothed_tilt)), 
                0, 
                [v if v >= 0 else 0 for v in smoothed_tilt], 
                color='black', 
                alpha=0.9
            )
        
        # Plot negative area (red)
        if any(negative_mask):
            ax.fill_between(
                range(len(smoothed_tilt)), 
                0, 
                [v if v < 0 else 0 for v in smoothed_tilt], 
                color='red', 
                alpha=0.9
            )
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Background color
        fig.patch.set_facecolor('#f0f0f0')
        ax.set_facecolor('#f0f0f0')
        
        # Save figure
        fig.tight_layout()
        canvas.draw()
        
        # Convert to OpenCV image
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(output_path, image)
        
        return output_path
    
    def calculate_possession_stats(self):
        """
        Calculate possession statistics based on the corrected ball possession tracking.
        
        Returns:
            Dictionary with possession percentages for each team
        """
        team1_possession = sum(self.ball_possession_history[1])
        team2_possession = sum(self.ball_possession_history[2])
        total_frames = len(self.frame_indices)
        
        if total_frames > 0:
            team1_pct = (team1_possession / total_frames) * 100
            team2_pct = (team2_possession / total_frames) * 100
        else:
            team1_pct = 50.0
            team2_pct = 50.0
            
        return {
            1: team1_pct,
            2: team2_pct
        }
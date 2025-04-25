import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import cv2
import requests
from io import StringIO


class MatchMomentumAnalyzer:
    def __init__(self, config, window_size=90, decay_rate=0.1, sigma=2):
        """
        Initialize the Match Momentum Analyzer using xT values.
        
        Args:
            config: Soccer pitch configuration
            window_size: Size of the rolling window in frames
            decay_rate: Rate at which older events lose importance
            sigma: Smoothing factor for the momentum curve
        """
        self.config = config
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.sigma = sigma
        
        # Store data for momentum calculation
        self.frames = []
        self.team1_momentum_data = []
        self.team2_momentum_data = []
        self.ball_positions = []
        self.team_possession = []
        self.prev_ball_positions = {}  # Store previous ball positions by team
        
        # Load xT grid from GitHub
        self.xT = self._load_xt_grid()
        if self.xT is not None:
            self.xT_rows, self.xT_cols = self.xT.shape
        else:
            print("Warning: Could not load xT grid, using fallback calculation")
        
    def _load_xt_grid(self):
        """
        Load the xT grid from GitHub.
        
        Returns:
            NumPy array of xT values or None if loading fails
        """
        try:
            url = "https://raw.githubusercontent.com/AKapich/WorldCup_App/main/app/xT_Grid.csv"
            response = requests.get(url)
            if response.status_code == 200:
                xT = pd.read_csv(StringIO(response.text), header=None)
                return np.array(xT)
            else:
                print(f"Error loading xT grid: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"Error loading xT grid: {e}")
            return None
    
    def add_frame_data(self, frame_num, team1_positions, team2_positions, ball_position, possession_team):
        """
        Add data from a frame for momentum calculation.
        
        Args:
            frame_num: Frame number
            team1_positions: Positions of team 1 players in pitch coordinates
            team2_positions: Positions of team 2 players in pitch coordinates
            ball_position: Position of the ball in pitch coordinates
            possession_team: Team currently in possession (1 or 2)
        """
        self.frames.append(frame_num)
        
        # Get ball position
        current_ball_pos = None
        if ball_position is not None and len(ball_position) > 0:
            current_ball_pos = ball_position[0]
        else:
            # Use previous position or default
            if self.ball_positions:
                current_ball_pos = self.ball_positions[-1]
            else:
                current_ball_pos = [self.config.length / 2, self.config.width / 2]
        
        # Store ball position
        self.ball_positions.append(current_ball_pos)
        
        # Store possession
        self.team_possession.append(possession_team)
        
        # Track ball movement for xT calculation
        prev_pos = self.prev_ball_positions.get(possession_team)
        if prev_pos is None:
            self.prev_ball_positions[possession_team] = current_ball_pos
            
        # Calculate team1 threat score
        team1_threat = self._calculate_threat_score(
            team1_positions, current_ball_pos, 1, possession_team, prev_pos
        )
        self.team1_momentum_data.append(team1_threat)
        
        # Calculate team2 threat score
        team2_threat = self._calculate_threat_score(
            team2_positions, current_ball_pos, 2, possession_team, prev_pos
        )
        self.team2_momentum_data.append(team2_threat)
        
        # Update previous position for the team in possession
        if possession_team in (1, 2):
            self.prev_ball_positions[possession_team] = current_ball_pos
    
    def _calculate_xt_value(self, position):
        """
        Calculate the xT value for a given position.
        
        Args:
            position: [x, y] position on the pitch
            
        Returns:
            xT value from the grid or fallback value if grid not loaded
        """
        if self.xT is None:
            # Fallback calculation
            normalized_x = position[0] / self.config.length
            return normalized_x * 0.1  # Simple linear increase toward goal
        
        # Convert position to grid coordinates
        x, y = position
        
        # Normalize coordinates to grid size
        x_norm = x / self.config.length
        y_norm = y / self.config.width
        
        # Convert to bin indices
        x_bin = min(self.xT_cols - 1, max(0, int(x_norm * self.xT_cols)))
        y_bin = min(self.xT_rows - 1, max(0, int(y_norm * self.xT_rows)))
        
        # Get xT value
        return self.xT[y_bin][x_bin]
    
    def _calculate_threat_score(self, team_positions, ball_position, team_id, possession_team, prev_ball_position):
        """
        Calculate a threat score for a team in the current frame using xT values.
        
        Args:
            team_positions: Positions of team players in pitch coordinates
            ball_position: Position of the ball in pitch coordinates
            team_id: Team ID (1 or 2)
            possession_team: Team currently in possession (1 or 2)
            prev_ball_position: Previous ball position for this team
            
        Returns:
            Threat score between 0 and 1
        """
        if team_positions.size == 0:
            return 0.0
            
        # Check if this team has possession
        has_possession = team_id == possession_team
        if not has_possession:
            return 0.0  # Only calculate xT for the team in possession
            
        # Get current xT value
        current_xt = self._calculate_xt_value(ball_position)
        
        # If we have a previous position, calculate the xT difference
        if prev_ball_position is not None:
            prev_xt = self._calculate_xt_value(prev_ball_position)
            xt_diff = max(0, current_xt - prev_xt)  # Only count positive xT changes
        else:
            xt_diff = 0
            
        # Cap the xT value at 0.1 as in the reference code
        xt_value = min(0.1, current_xt * 0.3 + xt_diff * 0.7)
        
        # Add factors for player positioning
        # Count players in attacking third
        attacking_third_start = 2 * self.config.length / 3 if team_id == 1 else 0
        attacking_third_end = self.config.length if team_id == 1 else self.config.length / 3
        
        players_in_attacking_third = sum(1 for pos in team_positions 
                                         if (attacking_third_start <= pos[0] <= attacking_third_end))
        
        # Normalize by total players
        attacking_player_factor = 0.2 * (players_in_attacking_third / max(1, len(team_positions)))
        
        # Combine factors
        threat_score = xt_value + attacking_player_factor
        
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, threat_score))
    
    def calculate_momentum(self):
        """
        Calculate momentum throughout the match.
        
        Returns:
            DataFrame with frame numbers and momentum values
        """
        if not self.frames:
            return pd.DataFrame({'frame': [], 'momentum': []})
        
        # Convert lists to arrays for easier calculation
        frames = np.array(self.frames)
        team1_threat = np.array(self.team1_momentum_data)
        team2_threat = np.array(self.team2_momentum_data)
        
        # Calculate weighted threat scores with decay
        team1_momentum = []
        team2_momentum = []
        momentum = []
        
        for i, frame in enumerate(frames):
            # Calculate window start
            window_start = max(0, i - self.window_size)
            
            # Get data for the window
            window_frames = frames[window_start:i+1]
            window_team1 = team1_threat[window_start:i+1]
            window_team2 = team2_threat[window_start:i+1]
            
            # Calculate decay weights
            time_diffs = frame - window_frames
            weights = np.exp(-self.decay_rate * time_diffs)
            
            # Calculate weighted sum
            if len(weights) > 0:
                team1_weighted = np.sum(weights * window_team1) / np.sum(weights)
                team2_weighted = np.sum(weights * window_team2) / np.sum(weights)
            else:
                team1_weighted = 0.0
                team2_weighted = 0.0
            
            team1_momentum.append(team1_weighted)
            team2_momentum.append(team2_weighted)
            
            # Calculate momentum as difference between teams
            momentum.append(team1_weighted - team2_weighted)
            
        # Create DataFrame
        momentum_df = pd.DataFrame({
            'frame': frames,
            'momentum': momentum,
            'team1_momentum': team1_momentum,
            'team2_momentum': team2_momentum
        })
        
        return momentum_df
    
    def generate_momentum_chart(self, output_path, team1_name="Team 1", team2_name="Team 2"):
        """
        Generate a momentum chart visualization similar to the reference code.
        
        Args:
            output_path: Path to save the output image
            team1_name: Name of team 1
            team2_name: Name of team 2
            
        Returns:
            Path to the saved chart image
        """
        # Calculate momentum
        momentum_df = self.calculate_momentum()
        
        if len(momentum_df) == 0:
            print("No data for momentum calculation")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        # Set up axis styling
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)
            
        # Set limits and ticks
        max_frame = momentum_df['frame'].max()
        tick_interval = max(1, max_frame // 6)
        ax.set_xticks(np.arange(0, max_frame + 1, tick_interval))
        ax.margins(x=0)
        ax.set_ylim(-0.8, 0.8)
        
        # Apply smoothing
        momentum_df['smoothed_momentum'] = gaussian_filter1d(momentum_df['momentum'], sigma=self.sigma)
        
        # Plot momentum line
        ax.plot(momentum_df['frame'], momentum_df['smoothed_momentum'], color='white')
        
        # Fill areas
        ax.axhline(0, color='white', linestyle='--', linewidth=0.5)
        ax.fill_between(momentum_df['frame'], momentum_df['smoothed_momentum'], 
                         where=(momentum_df['smoothed_momentum'] > 0), 
                         color='blue', alpha=0.5, interpolate=True)
        ax.fill_between(momentum_df['frame'], momentum_df['smoothed_momentum'], 
                         where=(momentum_df['smoothed_momentum'] < 0), 
                         color='red', alpha=0.5, interpolate=True)
        
        # Add labels
        ax.set_xlabel('Frame', color='white', fontsize=15, fontweight='bold', fontfamily='Monospace')
        ax.set_ylabel('Momentum (xT-based)', color='white', fontsize=15, fontweight='bold', fontfamily='Monospace')
        ax.set_title(f'Match Momentum (xT-based)\n{team1_name} vs {team2_name}', 
                     color='white', fontsize=20, fontweight='bold', fontfamily='Monospace', pad=10)
        
        # Add team labels
        team1_text = ax.text(max_frame * 0.1, 0.7, team1_name, fontsize=12, ha='center', 
                             fontfamily="Monospace", fontweight='bold', color='white')
        team1_text.set_bbox(dict(facecolor='blue', alpha=0.5, edgecolor='white', boxstyle='round'))
        
        team2_text = ax.text(max_frame * 0.1, -0.7, team2_name, fontsize=12, ha='center', 
                             fontfamily="Monospace", fontweight='bold', color='white')
        team2_text.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='white', boxstyle='round'))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Convert to OpenCV image
        img = cv2.imread(output_path)
        
        return output_path
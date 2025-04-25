import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import List, Optional, Tuple

from draw_pitch_voronoi import draw_pitch


class TeamHeatmapGenerator:
    def __init__(self, config, padding=50, scale=0.1, kernel_size=50):
        """
        Initialize the TeamHeatmapGenerator for creating team position heatmaps.
        
        Args:
            config: Soccer pitch configuration
            padding: Padding around the pitch in pixels
            scale: Scale factor for the pitch
            kernel_size: Size of the Gaussian kernel for heatmap generation
        """
        self.config = config
        self.padding = padding
        self.scale = scale
        self.kernel_size = kernel_size
        
        # Set up pitch dimensions
        self.pitch_width = int(config.width * scale)
        self.pitch_length = int(config.length * scale)
        self.pitch_img_width = self.pitch_width + 2 * padding
        self.pitch_img_length = self.pitch_length + 2 * padding
        
        # Initialize position accumulation arrays
        self.team1_positions = []
        self.team2_positions = []
        
        # Set up color maps
        self.team1_colormap = plt.cm.YlOrRd  # Yellow-Orange-Red for team 1
        self.team2_colormap = plt.cm.Blues   # Blues for team 2
    
    def add_frame_positions(self, team1_xy: np.ndarray, team2_xy: np.ndarray):
        """
        Add player positions from a frame to the accumulator.
        
        Args:
            team1_xy: Positions of team 1 players in pitch coordinates
            team2_xy: Positions of team 2 players in pitch coordinates
        """
        if team1_xy.size > 0:
            self.team1_positions.extend(team1_xy.tolist())
        
        if team2_xy.size > 0:
            self.team2_positions.extend(team2_xy.tolist())
    
    def _generate_heatmap_data(self, positions: List[List[float]], sigma=20) -> np.ndarray:
        """
        Generate a heatmap from position data using Gaussian kernels.
        
        Args:
            positions: List of [x, y] player positions in pitch coordinates
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            2D numpy array representing the heatmap
        """
        if not positions:
            return np.zeros((self.pitch_img_width, self.pitch_img_length))
        
        # Create empty heatmap
        heatmap = np.zeros((self.pitch_img_width, self.pitch_img_length))
        
        # Add each position to the heatmap
        for pos in positions:
            # Scale position to image coordinates
            x, y = pos
            x_scaled = int(x * self.scale) + self.padding
            y_scaled = int(y * self.scale) + self.padding
            
            # Skip positions outside the image bounds
            if (0 <= x_scaled < self.pitch_img_length and 
                0 <= y_scaled < self.pitch_img_width):
                heatmap[y_scaled, x_scaled] += 1
        
        # Apply Gaussian filter
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def _apply_colormap(self, heatmap: np.ndarray, colormap) -> np.ndarray:
        """
        Apply a colormap to a heatmap.
        
        Args:
            heatmap: 2D numpy array representing the heatmap
            colormap: Matplotlib colormap
            
        Returns:
            BGR image with the colormap applied
        """
        # Apply colormap
        colored_heatmap = (colormap(heatmap) * 255).astype(np.uint8)
        
        # Convert RGBA to BGR for OpenCV
        colored_heatmap_bgr = cv2.cvtColor(colored_heatmap, cv2.COLOR_RGBA2BGR)
        
        return colored_heatmap_bgr
    
    def generate_heatmap_image(self, output_path: str):
        """
        Generate and save a combined heatmap image for both teams.
        
        Args:
            output_path: Path to save the output image
            
        Returns:
            Path to the saved heatmap image
        """
        # Draw the base pitch
        pitch = draw_pitch(
            config=self.config,
            padding=self.padding,
            scale=self.scale
        )
        
        # Generate heatmaps
        team1_heatmap = self._generate_heatmap_data(self.team1_positions)
        team2_heatmap = self._generate_heatmap_data(self.team2_positions)
        
        # Create separate team heatmaps for overlay
        team1_colored = self._apply_colormap(team1_heatmap, self.team1_colormap)
        team2_colored = self._apply_colormap(team2_heatmap, self.team2_colormap)
        
        # Create masks for each team's significant presence
        team1_mask = team1_heatmap > 0.3  # Only show areas with significant presence
        team2_mask = team2_heatmap > 0.3
        
        # Create final image
        final_image = pitch.copy()
        
        # Overlay team 1 heatmap
        for i in range(3):  # BGR channels
            final_image[:, :, i] = np.where(
                team1_mask,
                cv2.addWeighted(final_image[:, :, i], 0.4, team1_colored[:, :, i], 0.6, 0),
                final_image[:, :, i]
            )
        
        # Overlay team 2 heatmap (with blending in overlapping areas)
        for i in range(3):  # BGR channels
            final_image[:, :, i] = np.where(
                team2_mask,
                cv2.addWeighted(final_image[:, :, i], 0.4, team2_colored[:, :, i], 0.6, 0),
                final_image[:, :, i]
            )
        
        # Add legend
        final_image = self._add_legend(final_image)
        
        # Save the image
        cv2.imwrite(output_path, final_image)
        
        return output_path
    
    def _add_legend(self, image: np.ndarray) -> np.ndarray:
        """
        Add a legend to the heatmap.
        
        Args:
            image: BGR image to add the legend to
            
        Returns:
            BGR image with legend
        """
        # Create legend area at the bottom
        legend_height = 60
        legend_img = np.ones((legend_height, image.shape[1], 3), dtype=np.uint8) * 255
        
        # Draw team 1 color bar
        team1_bar_x = int(image.shape[1] * 0.25)
        team1_text_x = int(image.shape[1] * 0.15)
        bar_width = 100
        bar_height = 20
        
        # Create team 1 gradient
        for i in range(bar_width):
            intensity = i / bar_width
            color = self.team1_colormap(intensity)
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            cv2.rectangle(
                legend_img,
                (team1_bar_x + i, 20),
                (team1_bar_x + i + 1, 20 + bar_height),
                color_bgr,
                -1
            )
        
        # Draw team 2 color bar
        team2_bar_x = int(image.shape[1] * 0.65)
        team2_text_x = int(image.shape[1] * 0.55)
        
        # Create team 2 gradient
        for i in range(bar_width):
            intensity = i / bar_width
            color = self.team2_colormap(intensity)
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            cv2.rectangle(
                legend_img,
                (team2_bar_x + i, 20),
                (team2_bar_x + i + 1, 20 + bar_height),
                color_bgr,
                -1
            )
        
        # Add labels
        cv2.putText(
            legend_img,
            "Team 1",
            (team1_text_x, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        cv2.putText(
            legend_img,
            "Team 2",
            (team2_text_x, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Add "Low" and "High" text
        cv2.putText(
            legend_img,
            "Low",
            (team1_bar_x - 30, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        cv2.putText(
            legend_img,
            "High",
            (team1_bar_x + bar_width + 5, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        cv2.putText(
            legend_img,
            "Low",
            (team2_bar_x - 30, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        cv2.putText(
            legend_img,
            "High",
            (team2_bar_x + bar_width + 5, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        # Combine with original image
        combined = np.vstack((image, legend_img))
        
        return combined
    
    def clear(self):
        """Clear accumulated position data."""
        self.team1_positions = []
        self.team2_positions = []
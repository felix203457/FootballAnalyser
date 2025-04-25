import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import threading
import PIL.Image, PIL.ImageTk
import os
from pathlib import Path
from pass_tracker import compute_pass_events, draw_pass_markers_on_pitch
from perspective_transformation import detect_field_keypoints, initialize_field_detection_model
from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from cam_estimator import CameraMovementEstimator
from pitch_config import SoccerPitchConfiguration
from create_voronoi import VoronoiDiagramGenerator
from field_tilt import FieldTiltAnalyzer
from heatmap import TeamHeatmapGenerator
from momentum_tracker import MatchMomentumAnalyzer
from view_transformer import ViewTransformer
from video_utils import *

ROBOFLOW_API_KEY = 'your_key_here'
PATHOFMODEL = 'set_model_path'

class ModernButton(tk.Button):
    """Custom button class with modern styling"""
    def __init__(self, master=None, **kwargs):
        # Extract custom parameters
        bg_color = kwargs.pop('bg_color', '#2C2F33')
        fg_color = kwargs.pop('fg_color', '#FFFFFF')
        hover_color = kwargs.pop('hover_color', '#5865F2')
        width = kwargs.pop('width', None)
        height = kwargs.pop('height', None)
        
        # Configure default styling
        kwargs['bg'] = bg_color
        kwargs['fg'] = fg_color
        kwargs['activebackground'] = hover_color
        kwargs['activeforeground'] = '#FFFFFF'
        kwargs['relief'] = tk.FLAT
        kwargs['borderwidth'] = 0
        kwargs['padx'] = kwargs.get('padx', 15)
        kwargs['pady'] = kwargs.get('pady', 8)
        
        # Initialize the button
        super().__init__(master, **kwargs)
        
        # Bind hover events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        # Set fixed width/height if specified
        if width:
            self.config(width=width)
        if height:
            self.config(height=height)
    
    def _on_enter(self, e):
        self['background'] = self['activebackground']
        
    def _on_leave(self, e):
        self['background'] = '#2C2F33'

class SoccerAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Soccer Match Analysis")
        self.root.geometry("1000x700")
        
        # Set up dark theme colors
        self.colors = {
            'bg_dark': '#1E1E1E',        # Main background
            'bg_medium': '#2C2F33',      # Frames background
            'bg_light': '#36393F',       # Controls background
            'text_bright': '#FFFFFF',    # Bright text
            'text_muted': '#B9BBBE',     # Muted text
            'accent': '#5865F2',         # Primary accent color
            'accent_hover': '#4752C4',   # Accent hover color
            'success': '#57F287',        # Success color
            'warning': '#FEE75C',        # Warning color
            'error': '#ED4245'           # Error color
        }
        
        # Configure the root window
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Initialize variables
        self.video_path = None
        self.output_paths = None
        self.video_playing = False
        self.video_cap = None
        
        # Create the input screen
        self.setup_input_screen()
    
    def setup_input_screen(self):
        # Clear any existing widgets
        self.clear_screen()
        
        # Main frame
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(pady=50, fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(main_frame, text="SOCCER MATCH ANALYSIS", 
                        font=("Helvetica", 24, "bold"), 
                        bg=self.colors['bg_dark'], 
                        fg=self.colors['text_bright'])
        title.pack(pady=20)
        
        # Content box
        content_box = tk.Frame(main_frame, bg=self.colors['bg_medium'], padx=30, pady=30)
        content_box.pack(padx=40, pady=10)
        
        # App logo/icon (can be replaced with an actual logo)
        logo_frame = tk.Frame(content_box, bg=self.colors['bg_medium'], width=80, height=80)
        logo_frame.pack(pady=10)
        # Create a canvas for the circular logo background
        logo_canvas = tk.Canvas(logo_frame, width=80, height=80, 
                              bg=self.colors['bg_medium'], 
                              highlightthickness=0)
        logo_canvas.pack()
        logo_canvas.create_oval(5, 5, 75, 75, fill=self.colors['accent'], outline="")
        logo_canvas.create_text(40, 40, text="⚽", font=("Helvetica", 24), fill=self.colors['text_bright'])
        
        # Instructions
        instructions = tk.Label(content_box, 
                             text="Select a soccer match video to analyze", 
                             font=("Helvetica", 12),
                             bg=self.colors['bg_medium'], 
                             fg=self.colors['text_muted'])
        instructions.pack(pady=20)
        
        # File selection button
        select_btn = ModernButton(content_box, 
                                text="SELECT VIDEO", 
                                command=self.select_video,
                                font=("Helvetica", 11, "bold"),
                                bg_color=self.colors['bg_light'],
                                hover_color=self.colors['accent'])
        select_btn.pack(pady=15)
        
        # Selected file frame
        file_frame = tk.Frame(content_box, bg=self.colors['bg_light'], padx=15, pady=10)
        file_frame.pack(fill=tk.X, pady=15)
        
        file_label_title = tk.Label(file_frame, text="SELECTED FILE:", 
                                  font=("Helvetica", 9), 
                                  bg=self.colors['bg_light'], 
                                  fg=self.colors['text_muted'])
        file_label_title.pack(anchor='w')
        
        self.file_label = tk.Label(file_frame, text="No file selected", 
                                 font=("Helvetica", 10), 
                                 bg=self.colors['bg_light'], 
                                 fg=self.colors['text_bright'])
        self.file_label.pack(anchor='w', pady=5)
        
        # Process button (initially disabled)
        self.process_btn = ModernButton(content_box, 
                                      text="PROCESS VIDEO", 
                                      command=self.process_video,
                                      state=tk.DISABLED, 
                                      font=("Helvetica", 11, "bold"),
                                      bg_color=self.colors['bg_light'],
                                      hover_color=self.colors['accent'])
        self.process_btn.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(content_box, text="", 
                                   font=("Helvetica", 10), 
                                   bg=self.colors['bg_medium'], 
                                   fg=self.colors['text_muted'])
        self.status_label.pack(pady=10)
    
    def select_video(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if file_path:
            self.video_path = file_path
            self.file_label.config(text=f"{Path(file_path).name}", fg=self.colors['accent'])
            self.process_btn.config(state=tk.NORMAL)
    
    def process_video(self):
        if not self.video_path:
            return
            
        # Disable process button and update status
        self.process_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing... This may take a while.", fg=self.colors['warning'])
        self.root.update()
        
        # Run processing in a separate thread to avoid freezing GUI
        def process_thread():
            try:
                # Call the main function with the selected video
                self.output_paths = main(self.video_path)
                
                # Update GUI in main thread
                self.root.after(0, self.show_results_menu)
            except Exception as e:
                # Handle errors
                self.root.after(0, lambda: self.show_error(str(e)))
        
        threading.Thread(target=process_thread).start()
    
    def show_error(self, error_msg):
        messagebox.showerror("Error", f"An error occurred during processing:\n{error_msg}")
        self.process_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Processing failed. Please try again.", fg=self.colors['error'])
    
    def show_results_menu(self):
        # Clear screen
        self.clear_screen()
        
        # Create results menu
        results_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header bar
        header = tk.Frame(results_frame, bg=self.colors['bg_medium'], height=60)
        header.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        title = tk.Label(header, text="ANALYSIS RESULTS", 
                       font=("Helvetica", 16, "bold"),
                       bg=self.colors['bg_medium'], 
                       fg=self.colors['text_bright'],
                       padx=20, pady=15)
        title.pack(side=tk.LEFT)
        
        # Content area
        content_area = tk.Frame(results_frame, bg=self.colors['bg_dark'])
        content_area.pack(padx=40, pady=10, fill=tk.BOTH, expand=True)
        
        # Results grid - divide into two columns
        left_col = tk.Frame(content_area, bg=self.colors['bg_dark'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_col = tk.Frame(content_area, bg=self.colors['bg_dark'])
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Create result cards
        outputs = [
            ("Annotated Video", "Review player tracking and team assignments", 
             lambda: self.show_video(self.output_paths["annotated_video"], "Annotated Video")),
            
            ("Voronoi Diagram", "Visualize player control areas on the field", 
             lambda: self.show_video(self.output_paths["voronoi_video"], "Voronoi Diagram")),
            
            ("Field Tilt Analysis", "See which team is dominating field position", 
             lambda: self.show_image(self.output_paths["field_tilt_chart"], "Field Tilt Analysis")),
            
            ("Team Position Heatmap", "Analyze team positioning patterns", 
             lambda: self.show_image(self.output_paths["heatmap"], "Team Position Heatmap")),
            
            ("Pass Markers", "Review passing patterns and key areas", 
             lambda: self.show_image(self.output_paths["passing_markers"], "Pass Markers")),
            
            ("Match Momentum Chart", "Track momentum shifts throughout the match", 
             lambda: self.show_image(self.output_paths["momentum_chart"], "Match Momentum"))
        ]
        
        # Create cards in alternating columns
        for i, (title, desc, command) in enumerate(outputs):
            if i % 2 == 0:
                self._create_result_card(left_col, title, desc, command)
            else:
                self._create_result_card(right_col, title, desc, command)
        
        # Footer with new analysis button
        footer = tk.Frame(results_frame, bg=self.colors['bg_dark'], pady=20)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        
        restart_btn = ModernButton(footer, text="START NEW ANALYSIS", 
                                 command=self.setup_input_screen,
                                 font=("Helvetica", 11, "bold"),
                                 bg_color=self.colors['bg_medium'],
                                 hover_color=self.colors['accent'])
        restart_btn.pack(pady=15)
    
    def _create_result_card(self, parent, title, description, command):
        """Create a card-style button for results"""
        card = tk.Frame(parent, bg=self.colors['bg_medium'], padx=15, pady=15)
        card.pack(fill=tk.X, pady=10)
        
        title_label = tk.Label(card, text=title.upper(), 
                             font=("Helvetica", 12, "bold"),
                             bg=self.colors['bg_medium'], 
                             fg=self.colors['text_bright'])
        title_label.pack(anchor='w')
        
        desc_label = tk.Label(card, text=description, 
                            font=("Helvetica", 10),
                            bg=self.colors['bg_medium'], 
                            fg=self.colors['text_muted'],
                            wraplength=300, justify='left')
        desc_label.pack(anchor='w', pady=5)
        
        view_btn = ModernButton(card, text="VIEW", 
                              command=command,
                              font=("Helvetica", 10),
                              bg_color=self.colors['bg_light'],
                              hover_color=self.colors['accent'])
        view_btn.pack(anchor='e', pady=(5, 0))
        
        # Make the whole card clickable
        for widget in [card, title_label, desc_label]:
            widget.bind("<Button-1>", lambda e, cmd=command: cmd())
            widget.bind("<Enter>", lambda e, c=card: c.config(bg=self.colors['bg_light']))
            widget.bind("<Leave>", lambda e, c=card: c.config(bg=self.colors['bg_medium']))
    
    def clear_screen(self):
        # Stop any playing videos
        if self.video_playing:
            self.video_playing = False
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
        
        # Clear all widgets
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def setup_viewer(self, title):
        self.clear_screen()
        
        # Main frame
        viewer_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        
        # Navigation bar
        nav_bar = tk.Frame(viewer_frame, bg=self.colors['bg_medium'])
        nav_bar.pack(fill=tk.X)
        
        # Back button
        back_btn = ModernButton(nav_bar, text="← BACK", 
                              command=self.show_results_menu,
                              font=("Helvetica", 10),
                              bg_color=self.colors['bg_medium'],
                              hover_color=self.colors['accent'])
        back_btn.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Title
        title_label = tk.Label(nav_bar, text=title.upper(), 
                             font=("Helvetica", 14, "bold"),
                             bg=self.colors['bg_medium'], 
                             fg=self.colors['text_bright'])
        title_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Content frame
        content_frame = tk.Frame(viewer_frame, bg=self.colors['bg_dark'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        return content_frame
    
    def show_video(self, video_path, title):
        content_frame = self.setup_viewer(title)
        
        # Create video display with border
        video_container = tk.Frame(content_frame, bg=self.colors['bg_medium'], bd=1)
        video_container.pack(pady=10)
        
        video_label = tk.Label(video_container, bg=self.colors['bg_dark'])
        video_label.pack(padx=2, pady=2)
        
        # Controls frame
        controls = tk.Frame(content_frame, bg=self.colors['bg_dark'])
        controls.pack(pady=15)
        
        play_btn = ModernButton(controls, text="PLAY/PAUSE", 
                              command=lambda: self.toggle_play_pause(),
                              font=("Helvetica", 10),
                              bg_color=self.colors['bg_medium'],
                              hover_color=self.colors['accent'])
        play_btn.pack(side=tk.LEFT, padx=10)
        
        restart_btn = ModernButton(controls, text="RESTART", 
                                 command=lambda: self.restart_video(),
                                 font=("Helvetica", 10),
                                 bg_color=self.colors['bg_medium'],
                                 hover_color=self.colors['accent'])
        restart_btn.pack(side=tk.LEFT, padx=10)
        
        # Open video
        self.video_cap = cv2.VideoCapture(video_path)
        self.video_playing = True
        
        # Function to update video frames
        def update_frame():
            if self.video_playing and self.video_cap and self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    # Convert frame for display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize to fit display
                    height, width = rgb_frame.shape[:2]
                    max_width = 900
                    max_height = 500
                    
                    if width > max_width or height > max_height:
                        scale = min(max_width / width, max_height / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
                    
                    # Update display
                    img = PIL.Image.fromarray(rgb_frame)
                    imgtk = PIL.ImageTk.PhotoImage(image=img)
                    video_label.imgtk = imgtk
                    video_label.config(image=imgtk)
                    
                    # Schedule next frame
                    video_label.after(30, update_frame)
                else:
                    # Restart video
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    video_label.after(30, update_frame)
        
        # Start video playback
        update_frame()
    
    def toggle_play_pause(self):
        self.video_playing = not self.video_playing
    
    def restart_video(self):
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def show_image(self, image_path, title):
        content_frame = self.setup_viewer(title)
        
        try:
            # Create image container with border
            img_container = tk.Frame(content_frame, bg=self.colors['bg_medium'], bd=1)
            img_container.pack(pady=10)
            
            # Load image
            img = PIL.Image.open(image_path)
            
            # Resize to fit display
            width, height = img.size
            max_width = 900
            max_height = 500
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
            
            # Display image
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            img_label = tk.Label(img_container, image=imgtk, bg=self.colors['bg_dark'])
            img_label.image = imgtk  # Keep reference
            img_label.pack(padx=2, pady=2)
            
            # Add caption
            caption = tk.Label(content_frame, text=f"{Path(image_path).name}", 
                             font=("Helvetica", 10),
                             bg=self.colors['bg_dark'], 
                             fg=self.colors['text_muted'])
            caption.pack(pady=5)
            
        except Exception as e:
            error_label = tk.Label(content_frame, text=f"Error loading image: {str(e)}", 
                                  font=("Helvetica", 12), 
                                  bg=self.colors['bg_dark'],
                                  fg=self.colors['error'])
            error_label.pack(pady=20)


# Include the main processing function from original code
def main(video_path):
    # Read video
    print(f"Reading video from {video_path}...")
    video_frames = read_video(video_path)
    
    if not video_frames:
        print("Error: Could not read video")
        raise Exception("Could not read video file")
    
    print(f"Successfully loaded {len(video_frames)} frames.")
    
    # Initialize Roboflow field detection model
    ROBOFLOW_API_KEY = 'VPdDhTq3F7b7jJhe6kOd'
    FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"
    print("Initializing field detection model...")
    field_detection_model = initialize_field_detection_model(ROBOFLOW_API_KEY, FIELD_DETECTION_MODEL_ID)
    
    # Initialize tracker
    print("Initializing object tracker...")
    tracker = Tracker(PATHOFMODEL)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path=None)
    tracker.add_position_to_tracks(tracks)
    
    # Initialize camera movement estimator
    print("Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=False, stub_path=None
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # Initialize team assigner
    print("Assigning teams to players...")
    team_assigner = TeamAssigner(device='cuda', num_frames=90)  # 3 seconds at 30fps
    team_assigner.initialize_from_multiple_frames(video_frames, tracks)
    
    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Process player tracking data
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # Assign ball to player and track passes
    print("Assigning ball to players...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)  # Default to team 1
    
    team_ball_control = np.array(team_ball_control)
    
    # Create base directory for outputs
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(base_dir, f"{base_name}_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create standard annotated video
    print("Creating annotated video...")
    output_video = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video = camera_movement_estimator.draw_camera_movement(output_video, camera_movement_per_frame)
    
    # Save standard annotated video
    output_video_path = os.path.join(output_dir, f"{base_name}_annotated.avi")
    print(f"Saving annotated video to {output_video_path}...")
    save_video(output_video, output_video_path)
    
    # Initialize soccer pitch configuration
    config = SoccerPitchConfiguration()
    
    # Initialize visualization components
    print("Initializing analysis tools...")
    voronoi_generator = VoronoiDiagramGenerator(config)
    field_tilt_analyzer = FieldTiltAnalyzer(config)
    heatmap_generator = TeamHeatmapGenerator(config)
    momentum_analyzer = MatchMomentumAnalyzer(config)
    
    # Dictionary to store transformers for visualization
    frame_transformers = {}
    
    # Create Voronoi diagram frames and calculate field tilt
    voronoi_frames = []
    print("Generating visualizations and analyzing data...")
    
    # Skip frames to speed up processing - process every 5th frame
#     FRAME_SKIP = 5
    
#     # Collect frames to process
#     frames_to_process = list(range(0, len(video_frames), FRAME_SKIP))
#     # Always include the last frame if not already included
#     if (len(video_frames) - 1) not in frames_to_process:
#         frames_to_process.append(len(video_frames) - 1)

    frames_to_process = list(range(0, len(video_frames)))
    print(f"Processing all {len(frames_to_process)} frames without skipping...")
    
    # Process only selected frames
    last_valid_voronoi = None
    processed_voronoi_frames = []
    
    for frame_num in frames_to_process:
        print(f"Processing frame {frame_num+1}/{len(video_frames)}", end='\r')
        frame = video_frames[frame_num]
        
        try:
            # Detect field keypoints
            frame_reference_points, pitch_indices = detect_field_keypoints(
                field_detection_model, frame, confidence_threshold=0.5
            )
            
            # Skip frame if not enough keypoints detected
            if len(frame_reference_points) < 4:
                print(f"\nSkipping frame {frame_num}: Not enough keypoints detected")
                continue
            
            # Get pitch reference points
            pitch_reference_points = np.array([config.vertices[i] for i in pitch_indices])
            
            # Initialize view transformer
            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )
            
            # Store transformer for visualization
            frame_transformers[frame_num] = transformer
            
            # Get player positions
            team1_positions = []
            team2_positions = []
            
            for player_id, player in tracks['players'][frame_num].items():
                if 'position' in player:
                    position = np.array([player['position']], dtype=np.float32)
                    
                    if player.get('team') == 1:
                        team1_positions.append(position[0])
                    else:
                        team2_positions.append(position[0])
            
            # Convert to numpy arrays
            team1_positions = np.array(team1_positions, dtype=np.float32)
            team2_positions = np.array(team2_positions, dtype=np.float32)
            
            # Transform positions to pitch coordinates
            if team1_positions.size > 0:
                team1_pitch_positions = transformer.transform_points(team1_positions)
            else:
                team1_pitch_positions = np.array([])
                
            if team2_positions.size > 0:
                team2_pitch_positions = transformer.transform_points(team2_positions)
            else:
                team2_pitch_positions = np.array([])
            
            # Get ball position
            if 1 in tracks['ball'][frame_num] and 'position' in tracks['ball'][frame_num][1]:
                ball_position = np.array([tracks['ball'][frame_num][1]['position']], dtype=np.float32)
                ball_pitch_position = transformer.transform_points(ball_position)
            else:
                ball_pitch_position = np.array([])
            
            # Add positions to heatmap generator
            heatmap_generator.add_frame_positions(team1_pitch_positions, team2_pitch_positions)
            
            # Update momentum analyzer
            current_possession_team = 1 if team_ball_control[frame_num] == 1 else 2
            momentum_analyzer.add_frame_data(
                frame_num,
                team1_pitch_positions,
                team2_pitch_positions,
                ball_pitch_position,
                current_possession_team
            )
            
            # Update field tilt analysis
            field_tilt_analyzer.update(
                frame_num, 
                team1_pitch_positions, 
                team2_pitch_positions, 
                ball_pitch_position
            )
            
            # Generate Voronoi diagram
            if team1_pitch_positions.size > 0 and team2_pitch_positions.size > 0:
                voronoi_diagram = voronoi_generator.generate_diagram(
                    team1_pitch_positions, team2_pitch_positions, ball_pitch_position
                )
                
                # Resize back to original size
                voronoi_diagram = cv2.resize(voronoi_diagram, (frame.shape[1], frame.shape[0]))
                
                last_valid_voronoi = voronoi_diagram.copy()
                processed_voronoi_frames.append((frame_num, voronoi_diagram))
            else:
                if last_valid_voronoi is not None:
                    processed_voronoi_frames.append((frame_num, last_valid_voronoi.copy()))
                else:
                    blank = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
                    processed_voronoi_frames.append((frame_num, blank))
                
        except Exception as e:
            print(f"\nError processing frame {frame_num}: {e}")
            if last_valid_voronoi is not None:
                processed_voronoi_frames.append((frame_num, last_valid_voronoi.copy()))
            else:
                blank = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
                processed_voronoi_frames.append((frame_num, blank))
    
    # Process Voronoi frames
    print("\nExpanding processed frames to match original video length...")
    full_voronoi_frames = []
    processed_frame_dict = dict(processed_voronoi_frames)
    processed_frame_numbers = sorted(list(processed_frame_dict.keys()))
    
    for i in range(len(video_frames)):
        if i in processed_frame_dict:
            full_voronoi_frames.append(processed_frame_dict[i])
        else:
            closest_frame_idx = min(processed_frame_numbers, key=lambda x: abs(x - i))
            full_voronoi_frames.append(processed_frame_dict[closest_frame_idx])
    
    # Compute passing markers
    print("\nGenerating pass markers...")
    pass_events = compute_pass_events(video_frames, tracks, frames_to_process, field_detection_model, config)
    
    # Generate all visualizations
    print("\nGenerating analysis outputs...")
    
    # Field tilt chart
    field_tilt_chart_path = os.path.join(output_dir, f"{base_name}_field_tilt.png")
    field_tilt_analyzer.generate_field_tilt_chart(field_tilt_chart_path)
    
    # Team heatmap
    heatmap_path = os.path.join(output_dir, f"{base_name}_team_heatmap.png")
    heatmap_generator.generate_heatmap_image(heatmap_path)
    
    # Pass markers
    passing_markers_path = os.path.join(output_dir, f"{base_name}_passing_markers.png")
    passing_markers_image = draw_pass_markers_on_pitch(config, pass_events)
    cv2.imwrite(passing_markers_path, passing_markers_image)
    
    # Match momentum chart
    momentum_chart_path = os.path.join(output_dir, f"{base_name}_momentum.png")
    momentum_analyzer.generate_momentum_chart(momentum_chart_path)
    
    # Calculate possession stats
    possession_stats = field_tilt_analyzer.calculate_possession_stats()
    print(f"\nBall Possession Stats:")
    print(f"Team 1: {possession_stats[1]:.2f}%")
    print(f"Team 2: {possession_stats[2]:.2f}%")
    
    # Save Voronoi diagram video
    print("\nSaving Voronoi diagram video...")
    voronoi_video_path = os.path.join(output_dir, f"{base_name}_voronoi.avi")
    save_video(full_voronoi_frames, voronoi_video_path)
    
    print("\nProcessing complete. Output files:")
    print(f"1. {output_video_path} - Original annotated video")
    print(f"2. {voronoi_video_path} - Voronoi diagram video")
    print(f"3. {field_tilt_chart_path} - Field tilt analysis chart")
    print(f"4. {heatmap_path} - Team position heatmap")
    print(f"5. {passing_markers_path} - Pass markers on pitch")
    print(f"6. {momentum_chart_path} - Match momentum chart (xT-based)")
    
    # Return dictionary of output paths
    return {
        "annotated_video": output_video_path,
        "voronoi_video": voronoi_video_path,
        "field_tilt_chart": field_tilt_chart_path,
        "heatmap": heatmap_path,
        "passing_markers": passing_markers_path,
        "momentum_chart": momentum_chart_path
    }

# Application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = SoccerAnalysisGUI(root)
    root.mainloop()
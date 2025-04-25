from draw_pitch_voronoi import draw_pitch
import cv2
import numpy as np

from perspective_transformation import detect_field_keypoints
from view_transformer import ViewTransformer

def compute_pass_events(video_frames, tracks, frames_to_process, field_detection_model, config, padding=50, scale=0.1):
    """
    Compute pass events based on ball possession changes.
    A pass is recorded when ball possession switches from one player to another on the same team.
    Returns a list of tuples: (start_pitch_position, end_pitch_position, team).
    """
    pass_events = []
    previous_possession = None  # Dict with keys: player_id, team, pitch_position

    for frame_num in frames_to_process:
        frame = video_frames[frame_num]
        # Detect field keypoints to compute the transformation
        frame_reference_points, pitch_indices = detect_field_keypoints(field_detection_model, frame, confidence_threshold=0.5)
        if len(frame_reference_points) < 4:
            continue  # Skip if not enough keypoints

        # Get corresponding pitch reference points from your pitch configuration vertices
        pitch_reference_points = np.array([config.vertices[i] for i in pitch_indices])
        try:
            transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)
        except Exception as e:
            continue

        # Find the player with ball possession in this frame
        current_possession = None
        for player_id, player in tracks['players'][frame_num].items():
            if player.get('has_ball', False):
                # Use player's position as the possession location
                ball_pos = np.array([player['position']], dtype=np.float32)
                pitch_pos = transformer.transform_points(ball_pos)[0]
                current_possession = {
                    'player_id': player_id,
                    'team': player['team'],
                    'pitch_position': pitch_pos
                }
                break

        if current_possession is None:
            continue

        if previous_possession is not None:
            # Check if the ball switched to a different player on the same team
            if previous_possession['player_id'] != current_possession['player_id']:
                if previous_possession['team'] == current_possession['team']:
                    # Record the pass event (start, end, team)
                    pass_events.append((
                        previous_possession['pitch_position'],
                        current_possession['pitch_position'],
                        current_possession['team']
                    ))
        previous_possession = current_possession

    return pass_events


def draw_pass_markers_on_pitch(config, pass_events, padding=50, scale=0.1):
    """
    Draws arrows on a pitch image for each pass event.
    Each arrow is drawn from the start to the end pitch position.
    Arrow color is determined by team (e.g. team 1 red, team 2 green).
    """
    # Create a base pitch image
    pitch = draw_pitch(
        config=config,
        padding=padding,
        scale=scale
    )

    for start, end, team in pass_events:
        # Choose color based on team (BGR format)
        color = (0, 0, 255) if team == 1 else (0, 255, 0)
        # Scale the pitch coordinates to image coordinates
        start_scaled = (int(start[0] * scale) + padding, int(start[1] * scale) + padding)
        end_scaled = (int(end[0] * scale) + padding, int(end[1] * scale) + padding)
        # Draw an arrow from the start to the end position
        cv2.arrowedLine(pitch, start_scaled, end_scaled, color, thickness=2, tipLength=0.2)

    return pitch





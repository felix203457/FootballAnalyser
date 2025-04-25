from video_utils import get_centre_of_bbox, measure_distance


class PlayerBallAssigner:
    """
    Class for assigning ball possession to the nearest player based on bounding box proximity.
    
    The algorithm calculates the center point of the ball and each player, then finds
    the player within a specified maximum distance who is closest to the ball.
    """
    
    def __init__(self):
        # Threshold (in pixels or spatial units) within which a player is considered
        # close enough to the ball to be assigned possession.
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, tracked_players, ball_bbox):
        """
        Assigns the ball to the closest player within a distance threshold.

        Args:
            tracked_players (dict): Dictionary of player objects.
                Each key is a player ID, and the value must include a 'bbox' (bounding box).
            ball_bbox (tuple): Bounding box coordinates of the ball (x1, y1, x2, y2)

        Returns:
            int: ID of the player closest to the ball (or -1 if no player is close enough)
        """
        
        # Step 1: Get the center point of the ball's bounding box
        ball_center = get_centre_of_bbox(ball_bbox)

        # Step 2: Initialize search for the closest player
        closest_distance = self.max_player_ball_distance
        closest_player_id = -1  # Default to -1 if no player is within threshold

        # Step 3: Loop through all tracked players
        for player_id, player_data in tracked_players.items():
            player_bbox = player_data['bbox']

            # Get the center of the player (e.g. midpoint of bounding box)
            player_center = get_centre_of_bbox(player_bbox)

            # Calculate Euclidean distance between player and ball centers
            distance_to_ball = measure_distance(player_center, ball_center)

            # If this player is closer than anyone seen so far, and within allowed range
            if distance_to_ball < closest_distance:
                closest_distance = distance_to_ball
                closest_player_id = player_id

        # Step 4: Return the player who is nearest to the ball (or -1 if no one close enough)
        return closest_player_id

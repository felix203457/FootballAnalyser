import cv2
def get_centre_of_bbox(bbox):
    x1, y1, x2, y2 = map(int, bbox)  # Ensure all values are integers
    return (x1 + x2) // 2, (y1 + y2) // 2  # Integer division for accuracy


def get_width_of_bbox(bbox):
    x1, x2 = map(int, (bbox[0], bbox[2]))  # Ensure integer values
    return x2 - x1

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]


def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)



def read_video(vid_path):
    frames = []
    cap = cv2.VideoCapture(vid_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {vid_path}")
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()  # Ensure resources are released
    return frames


def save_video(output_vid_frames, output_vid_path):
    if not output_vid_frames:
        print("Error: No frames to write to video.")
        return

    height, width = output_vid_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_vid_path, fourcc, 20.0, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create video file {output_vid_path}")
        return

    for frame in output_vid_frames:
        # Ensure frames are in uint8 format
        if frame.dtype != 'uint8':
            frame = cv2.convertScaleAbs(frame)
        out.write(frame)

    out.release()
    print(f"Video saved successfully: {output_vid_path}")

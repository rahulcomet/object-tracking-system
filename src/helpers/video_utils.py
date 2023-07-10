import cv2

def init_video_writer(output_path, width, height, fps, codec='XVID'):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

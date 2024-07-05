import cv2
import os

def extract_frames(video_path, output_folder, frame_interval):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Capture the video
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frames_to_extract = fps // frame_interval
    
    while True:
        success, image = video_capture.read()
        if not success:
            break
        
        if frame_count % frames_to_extract == 0:
            # Save frame as JPEG file
            frame_filename = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}frame{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, image)
            saved_frame_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    video_capture.release()
    print(f"Extracted {saved_frame_count} frames from {video_path}")

def process_videos(input_folder, output_folder, frame_interval):
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):  # Add other video formats if needed
            video_path = os.path.join(input_folder, filename)
            extract_frames(video_path, output_folder, frame_interval)

# Example usage
input_folder = "C:/Users/Aditi Bolakhe/Desktop/drowsy detection/DATASET_NEW/drowsy_yawn1/pixel_video"  # Replace with your input folder path
output_folder = 'C:/Users/Aditi Bolakhe/Desktop/video_extract'  # Replace with your desired output folder
frame_interval = 0.5  # 2 frames per second
process_videos(input_folder, output_folder, frame_interval)
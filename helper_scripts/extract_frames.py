import subprocess
import os
import glob

subject_id = 'v0302'

video_path = "../Data/" + subject_id + "/input.mov"  

# Output folder
output_folder = "../Data/"+subject_id+"/frames/"
os.makedirs(output_folder, exist_ok=True)

# Temporary naming pattern
temp_pattern = os.path.join(output_folder, "temp_%04d.png")

# FFmpeg command to extract frames at 5 FPS and resize to 1080x1080
ffmpeg_cmd = [
    "ffmpeg",
    "-i", video_path,       # Input video
    "-vf", "fps=10,scale=1080:1080",  # Set FPS to 5 and resize to 1080x1080
    "-q:v", "2",            # Quality setting (lower is better)
    temp_pattern            # Temporary output format (e.g., temp_0001.png)
]

# Run the FFmpeg command
subprocess.run(ffmpeg_cmd, check=True)

# Get the extracted frames and rename them starting from 0.png
frame_files = sorted(glob.glob(os.path.join(output_folder, "temp_*.png")))

for index, old_name in enumerate(frame_files):
    new_name = os.path.join(output_folder, f"{index}.png")  # Start from 0.png
    os.rename(old_name, new_name)

print(f"Frames extracted, resized to 1080x1080, and renamed starting from 0.png in {output_folder}/")
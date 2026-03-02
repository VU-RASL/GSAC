import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil
import subprocess
import os
from pathlib import Path
import numpy as np
import yaml

import cv2
def get_video(directory):
    # Common video file extensions
    video_extensions = {'.mp4', '.mov', '.avi'}

    # Get all video files in the directory
    video_files = [file for file in os.listdir(directory) 
                   if os.path.isfile(os.path.join(directory, file)) 
                   and os.path.splitext(file)[1].lower() in video_extensions]

    return video_files
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str)
    parser.add_argument('--gender', type=str)
    parser.add_argument('--start', type=str, default='')
    parser.add_argument('--stage', type =str, default='all')
    args = parser.parse_args()
    assert args.subject, "Please set subject."
    assert args.gender, "Please set gender."
    return args

args = parse_args()

subject_id = args.subject
gender = args.gender

############################################ Video to images #################################


if args.start =='video':
    ########################################### Video to images #################################
    subject_folder = './Data/'+subject_id
    video_name = get_video(subject_folder)
    if len(video_name) > 1 :
        print('MORE THAN ONE VIDEO DETECTED.')
    else:
        print(video_name)
        if not os.path.exists(subject_folder + '/frames/') :
            os.makedirs(subject_folder+'/frames/')  # Create the folder if it does not exist

        # if only raw video is provided.
            cmd = 'ffmpeg -i ' + subject_folder+'/'+video_name[0] + " -vf scale=720:1280,fps=5 "+ subject_folder+'/frames/%d.png  ' 
            result = os.system(cmd)
            print(' Frame extracted from '+ video_name[0])
        elif  not os.listdir(subject_folder+'/frames/'):
            cmd = 'ffmpeg -i ' + subject_folder+'/'+video_name[0] + " -vf scale=720:1280,fps=5 "+ subject_folder+'/frames/%d.png  ' 
            result = os.system(cmd)
            print(' Frame extracted from '+ video_name[0])


    # Input and output directories
    input_folder = subject_folder + '/frames/'
    output_folder = subject_folder + '/frames/'

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Target dimensions
    target_pad_size = (1280, 1280)  # Zero padding to this size
    final_size = (1080, 1080)       # Final resizing

    # Process each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping {filename}, couldn't read the image.")
                continue

            h, w = img.shape[:2]  # Original dimensions: 1280x720

            # Calculate padding (even padding on both left and right)
            pad_left = (target_pad_size[1] - w) // 2  # (1280 - 720) / 2 = 280
            pad_right = target_pad_size[1] - w - pad_left  # Ensure correct width

            # Apply zero padding
            padded_img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Resize to 1080x1080
            resized_img = cv2.resize(padded_img, final_size, interpolation=cv2.INTER_CUBIC)

            # Save the processed image
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, resized_img)

            print(f"Processed and saved: {save_path}")

    print("Batch processing completed!")

########################################### Preprocessor #####################################

subject_path = os.getcwd()+'/Data/'+subject_id + '/frames/'
print(subject_path)
print('ttttttttttt')
os.chdir('Preprocessor')

cmd = ' python run.py --root_path ' + subject_id+" "  + " --gender "+ gender
print(cmd)
result = os.system(cmd)
print( "#############################################")
print( " ")
print( "          Preprocessor Job Done.")
print( " ")
print( "#############################################")

############################################ Avatar #########################################

os.chdir('../')
new_data_root = os.getcwd() + '/Data/'+subject_id +"/train/"

os.chdir('Avatar')


# new_data_root = os.getcwd() + '/Data/'+subject_id
print(new_data_root)
# Path to yaml file
yaml_path = os.getcwd()  + "/configs/GSAC_custom.yaml"
print(yaml_path)
# Load yaml
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

# Modify train data_root
config["train_dataloader"]["params"]["data_root"] = new_data_root
config["val_dataloader"]["params"]["data_root"] = new_data_root

with open(yaml_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

cmd = ' python main.py --base=./configs/GSAC_custom.yaml  --gender ' +  gender + ' ' +  '--train_subject ' + subject_id

result = os.system(cmd)


print( "#############################################")
print( " ")
print( "          Avatar Creator Job Done.")
print( " ")
print( "#############################################")

# ############################################# Move result to correct place #########################################

# # MV the latest result to Data/Subject


# log_dir = os.getcwd()+ "/logs/GSAC_custom/"

# # Get all subdirectories inside the test-time directory
# subdirs = [d for d in Path(log_dir).iterdir() if d.is_dir()]

# # Find the latest folder based on modification time
# if subdirs:
#     latest_folder = max(subdirs, key=lambda d: d.stat().st_mtime)
#     print("Latest folder:", latest_folder)
# else:
#     print("No folders found in", log_dir)
# os.chdir('..')

# source_path = os.path.join(os.getcwd()+'/Avatar/',log_dir,latest_folder)
# target_path =os.path.join(os.getcwd() + '/Data/',subject_id)
# cmd = 'mv ' + source_path + ' ' + target_path
# result = os.system(cmd)



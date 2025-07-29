# GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration
Gaussian Splatting Avatar Creation Release 

This is official code release for GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration. This repo releases our end-to-end pipeline for gaussian splatting avatar creation for a monocular video in ~ 40 mins. Our pipeline incorporates a novel Gaussian splatting technique with
customized preprocessing, enabling detailed facial expression reconstruction and seamless integration with Unity-based VR/AR platforms. Additionally, we present a Unity-integrated Gaussian
Splatting Avatar Editor, offering a user-friendly environment for VR/AR application development. 
## Install
Clone the repo
~~~
git clone https://github.com/VU-RASL/GSAC.git
~~~
cd to the repo, then run
~~~
bash Avatar/docker/build.sh
bash Avatar/docker/run.sh
bash post-build.sh
pip install -r Avatar/docker/extra_require.txt
~~~
We recommand use docker image.
## Data Gathering 
Please record the video using your phone by rotating the yourself slowly and keeping the camera stable. The time would be about 20 secs. Please make sure there is only one human inside a video each of the time. 
Place the recorded video under inputs folder:
~~~

ROOT
    |__Data/
        |__{SUBJECT}/video.MOV
    |__Preprocessor/
    |__Avatar/

~~~
Or you can place the extracted frames you want to use from the video in the folder:
~~~

ROOT
    |__Data/
        |__{SUBJECT}/frames/
    |__Preprocessor/
    |__Avatar/

~~~
## Simplest Run
We provide a single bash script that you can easily run the pipeline end-to-end easily. If can not only create avatar from video input, but also can create from sliced frames from video. After docker is running and the data is pleased in correct place, navigate to ROOT folder of GSAC, run the command below:
~~~

python create_avatar.py --subject {SUBJECT} --gender {GENDER}  --start {video or image}

~~~
{SUBJECT} is the name of the folder of you data. {GENDER} is the gender of the recorded person.  

--start is optional. If you wish to create an avatar from a ~20 seconds video,   use  --start video  
Otherwise, leave it as blank.


Results will be saved in {SUBJECT} folder under Data/
~~~

ROOT
    |__Data/
        |__{SUBJECT}
        |________frames/
        |________train/
        |________Result/
    |__Preprocessor/
    |__Avatar/

~~~
-------------------------------------------------------------------------------------------------------------------------------------

Otherwise you can run the preprocessing and training separately.
## Data Preprocessing 
For capturing SMPLX parameters from video frames, inside the docker environment we created above, please run our preprocessing using commands below （Note: for infant mode, gender is INFANT）:
~~~
cd Preprocessor
python run.py --root_path {SUBJECT} --gender {GENDER}
~~~
The fitted results will be placed in train_data under {Subject} folder :
~~~

ROOT
    |__Data/
        |__{SUBJECT}/
            |__frames/
            |__train/
    |__Preprocessor/
    |__Avatar/

~~~
## Avatar Training 
After correctly downloading tools and setting up environment and data, you would be able to train the gaussian avatar using the following command in GSAC folder :
~~~
cd Avatar/
python main.py --base=./configs/GSAC_custom.yaml  --gender {GENDER} --train_subject {SUBJECT}
~~~
The training logs and results will be saved in {SUBJECT}_{START_TIME}
~~~

ROOT
    |__Data/
    |__Preprocessor/
    |__Avatar/
        |__logs/GSAC_custom/
            |__{SUBJECT}_{START_TIME}

~~~

-------------------------------------------------------------------------------------------------------------------------------------

## Results
The visual results will be available in Data/{SUBJECT}/Result/.

To visualize the avatar and animate it, please move state_dict.json and avatar.ply to training_asserts/ in [Unity Editor package](https://github.com/VU-RASL/UnityGaussianSplatting).

## IMPORTANT NOTICE
If you create an avatar of a person using this pipeline, please obtain permission from the individual before using it. If you use our pipeline, you AGREE  ALL use complies with legal and ethical standards.

## Citation:

If you find our work useful, please cite:

~~~
@misc{zhang2025gsacleveraginggaussiansplatting,
      title={GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration}, 
      author={Rendong Zhang and Alexandra Watkins and Nilanjan Sarkar},
      year={2025},
      eprint={2504.12999},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2504.12999}, 
}
~~~

```
cd BoostTrack
conda env create -f boost-track-env.yml
conda activate boostTrack
```

Download the weights from here: [Weights](https://drive.google.com/drive/folders/15hZcR4bW_Z9hEaXXjeWhQl_jwRKllauG)
and place them in `external/weights/`

The directory hierarchy would be:
```
|
|____data 
|      |_____tracking
|
|____MOT (this repo)
       |_____models
       |       |________yolov9c_trained.pt
       |
       |_____BoostTrack
               |________external
               |             |_______weights
               |             |
               |             | (rest of the directory)
               |             |
```

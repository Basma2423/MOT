
1. Download the weights from here: [weights](https://drive.google.com/drive/folders/1vYcMvm2VaHtYFJ_rtpR-rSmdG8MUGdFp?usp=sharing)
Place the models at `Detector` in `models/`
Place the `ReID` weights in `BoostTrack/external/weights/`

2. Run the Tracker
```
cd BoostTrack
conda env create -f boost-track-env.yml
conda activate boostTrack
python multiple_object_tracking.py (or run the file (e.g. in vs code) using the Python interpreter from the Conda environment.)
```

3. To evaluate the results:
```
python external/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT20 --METRICS HOTA
```
__**
Note: to match the competition's score,
Competition Score = (1.45562553679 * HOTA) - 0.31272426853
**__

The directory hierarchy would be:
```
|
|____data 
|      |_____tracking
|
|____tests
|      |_____det.txt // ground truth for the detector
|      |_____gt.txt // ground truth for the tracker
|
|____MOT (this repo)
       |_____models
       |       |________yolov9c_trained.pt (download from drive and place here)
       |       |________best.pt (fine-tuned on multiple datasets) (download from drive and place here)
       |
       |_____BoostTrack
               |________ multiple_object_tracking.py (main file)
               |________external
               |             |_______weights (download from drive and place them here)
               |             |
               |             |-------CSV_files (contains the generated result)
               |             |
               |             | (rest of the directory)
               |             |
```

# skellytracker

The tracking backend for freemocap. Collects different pose estimation tools and aggregates them using a consistent API. Can run pose estimation on images, webcams, and videos.

## Run skelly_tracker

Installation: `pip install skellytracker``
Then it can be run with `skellytracker`.

Running the basic `skellytracker` will open the first webcam port on your computer and run pose estimaiton in realtime with mediapipe holistic as a tracker. You can specify the tracker with `skellytracker TRACKER_NAME`, where `TRACKER_NAME` is the name of an available tracker. To view the names of all available trackers, see `RUN_ME.py`.

It will take some time to initialize the tracker the first time you run it, as it will likely need to download the model.

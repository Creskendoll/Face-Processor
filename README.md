# Face-Processor
A program which analyses images, and videos in order to get information about the faces within the frames. The program utilizes Microsoft Azure cognitive services using their API. At the current version the program is built to first cut frames out of a video file and analyse those frames for faces. Processing a video feed frame for frame is costly and since the web request takes time to result it is not possible. 

## Extracting the frames
First you have to have a video, preferably as a .mp4. Run the ``frameExtractor.py`` script by specifying the path to the video file. An example would be:

- ``python frameExtractor.py ./video.mp4``

This will save the images into the frames folder with the folder name being the video files name.

## Processing the frames
Add your Azure Cognitive Services subscription key to the ``index.py`` script and run it by specifying the folder for the frames.

- ``python index.py ./frames/video``

This will process all the images within the given folder recursively. The resulting .csv will be saved as ``data.csv`` to the root directory.

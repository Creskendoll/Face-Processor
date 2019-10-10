# Face-Processor
BAU capstone project for the term 2019-2020. This project aims to combine face recognition with cognitive functionality such as emotion and focus detection.    

The program utilizes the Microsoft Azure cognitive services API. The service returns predictions for emotions and several other features. [Refer here for more info on cognitive services.](https://docs.microsoft.com/en-us/azure/cognitive-services/face/index) 

## Project setup
You need Python 3 and pip to setup and run the project. Linux is the recommended OS.

On Ubuntu run this to install opencv:
- `sudo apt-get install python-opencv` 

If you're using Windows you might need to install C/C++ build tools in order to build opencv. [More info can be found here.](https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html)

To install the required dependencies navigate into the projects directory and run: 
- `python -m pip install -r req.txt`

## Running the project
In the current state you have to run the `main.py` file to run the project. You can do this by running this command:
- `python main.py`

This should open up a window with a video feed from your webcam and face outlines.
# K-Means 

A basic implementation of the K-means clustering algorithm 

### Setting up the environment 
For the program to run, you must ensure you have all necesary libraries installed on your machine. 

The only non standard library (not shipped with Python), is numpy. To install, simply run `pip3 install numpy`


### Running the Program
Run `python3 km.py` in the cloned directory. 

You will then be prompted for a path to the data file. Please provide the full path to the data you wish to evaluate. 

Next, you will be prompted for the number maximum of passes over the algorithm. This will prevent the program from running indefinitely if the desired accuracy is not met. Please only input an integer. 

### Performance 
The program was designed to a basic vanilla implementation of the k-means algorithm. The only improvements to accuracy included is a bounding to cluster size to ensure the data is not groupped entirely in a single general cluster (this would not be helpful). This program is provided as is and if greater accuracy is needed, please fork the repository and implement changes as you see fit. 
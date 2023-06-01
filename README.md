# REU Project
This repository contains code needed for the 2023 REU project on using a convolutional neural network (CNN) for severe weather prediction at timescales beyond a week. 

Synoptic_vars.m loads in Practically perfect hindcast data, ERA5 data, and OLR. ERA5 data is read in and interpolated to a resolution of 1.5x1.5. Data is also formatted to include only Nov-May.

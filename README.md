# Satellite-Imagery-Analysis

# Using Satellite Images to Identify Forest Types

## Overview
In a previous github repository I explored regression models in order to predict how much a plot of land would bid for in a timber auction through the Bureau of Land Management. One of the predictors this model was lacking was information on the species of trees. Since different tree species have different values, having this information would be a great addition to the prediction model. This project uses satellite imagery data from Sentinel Hub covering an area slightly southwest of Eugene, Oregon in the Siuslaw National Forest. Indices representing moisture levels, leaf chlorophyll depth, forest canopy density, and forest health were derived from the satellite imagery data. If this data can be used to identify what trees are in the satellite imagery, then tree species may be able to get added as a feature in the previously mentioned regression model and improve its predictive capabilities.

## Goals
The goal of this project is to identify forest types using clustering machine learning algorithms on satellite imagery data. If forest types can be identified, this information will be very useful to logging companies prospecting future timber lots.

## Data
The data used consists of satellite data from Sentinel Hub. The satellite data is from two polar-orbiting satellites that make up "The Copernicus Sentinel-2 Mission". Sentinel Hub was a good choice for me because of the Python API they offer for pulling data. I created a trial account in order to request data within coordinates of interest. Each pixel in the images represents a 72m X 72m area. The image used is from an area with both historically logged and uncut forests southwest of Eugene, Oregon. The image is 691 x 894 pixels and represents an area with a northwest corner of (44.5833 N, 124.0333 W) and a southeast corner of (44 N, 123.4167 W). This area is roughly 49 km from east to west, and 64 km from south to north. These two areas were chosen because these areas were explored in one of my previous github repositories.
## Repository Contents
---
<pre>
Data            : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/tree/main/Data>Data Files </a>

Code            : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/main/Notebooks/Data_Notebook_Final.ipynb>Data Cleaning Notebook </a>
                : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/main/Notebooks/EDA.ipynb>Data Visualization Notebook </a>
                : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/main/Notebooks/Modeling_Notebook.ipynb>Clustering Notebook </a>
                : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/main/Notebooks/sat_utils.py>Utilities File </a>
                
Report          : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/main/Report/Report.ipynb>Report Notebook</a>
</pre>
## Project Information
---
<b>Author: </b>David Harper <br>
<b>Language: </b>Python <br>
<b>Tools/IDE: </b>Jupyter Notebook and Spyder via Anaconda <br>
<b>Libraries: </b>numpy, pandas, plotly, matplotlib, math, sklearn, seaborn, mpu, geopy, sentinelhub, zipfile, os, datetime, requests, urllib, urllib3, scipy

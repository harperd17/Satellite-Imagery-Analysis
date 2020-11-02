# Satellite-Imagery-Analysis

# Using Satellite Images to Identify Forest Types

## Overview
In one of my previous [github repositories](https://github.com/harperd17/Timber-Sale-Vaulation), I built a mutliple linear regression model to predict the value of a timber plot with the Bureau of Land Management (BLM). One of the limitations of the model was the lack of features available in the data. In particular, one of the limitations was that there wasn't any information regarding the tree species on the timber plots. This project uses unsupervised learning techniques to find different groupings of trees in satellite images in order to take the place of physical land surveys. 

## Data
The data used consists of satellite images used from [Sentinel Hub](https://www.sentinel-hub.com/). I created a trial account and used their Python API in order to request images within coordinates of interest. Each pixel in the images represents a 72m X 72m area. The images used are from areas with historical logging west and southwest of Eugene, Oregon, and east of Pierce, Idaho. Elevation data was also used. The elevation was pulled from [USGS](https://nationalmap.gov/epqs/pqs.php?) using URL parsing through Python. Since the elevation data takes a long time to pull, getting an elevation for each pixel with the raw images resolution is unrealistic. Therefore, I downsized the images into a 360m X 360m resolution. The images used in analysis were of size <b> Insert the sizes </b>.
## Repository Contents
---
<pre>
Data            : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/master/Data>Data Files </a>
                 Raw data file found in 'Raw Data.zip' and data files used for visualization and modelling found in 'Manipulated Data.zip'

Code            : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/master/Notebooks/Data_Cleanup.ipynb>Data Cleaning Notebook </a>
                : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/master/Notebooks/Data_Visualizations.ipynb>Data Visualization Notebook </a>
                : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/master/Notebooks/Final_Classification_Modelling.ipynb>Classification Modelling Notebook </a>
                
Report          : <a href=https://github.com/harperd17/Satellite-Imagery-Analysis/blob/main/Report/Final%20Report_Notebook.ipynb>Report Notebook</a>
</pre>
## Project Information
---
<b>Author: </b>David Harper <br>
<b>Language: </b>Python <br>
<b>Tools/IDE: </b>Jupyter Notebook and Spyder via Anaconda <br>
<b>Libraries: </b>numpy, pandas, plotly, matplotlib, math, sklearn, statsmodels, seaborn

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:03:44 2020

@author: david
"""
import pandas as pd
import mpu
import math
import geopy
import geopy.distance
import numpy as np

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import SHConfig
import numpy as np
import pandas as pd
import requests
import urllib
import urllib3

from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest, to_utm_bbox
from sentinelhub import geo_utils


def plot_image(image, title, factor=1.0, clip_range = None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def checkSize(coords, resolution):
    """
    This function allows you to try out different bounaries and resolutions to make sure it will
    work with the Sentinel API (restricted to 2500x2500 or less). Also creates the objects necessary to make the image request.

    Parameters
    ----------
    coords : List containing the northwest corner and southeast corner coordinates of area box.
    resolution : How many meters each pixel should represent in satellite image.

    Returns
    -------
    bbox : An object that is needed for gathering satellite image from Sentinel API
    size : List with height and width of image in pixels. The limit is 2500 for each.

    """
    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    return bbox, size



def getSatelliteImage(box, size,folder,date_range,mosaick,config):
    
    """
    This function sends a request to SentinelHub through the API to get an image of a certain bounding.
    
    Parameters
    ----------
    box : box returne from checkSize.
    size : size returned from checkSize.
    folder : folder location to store image.
    date_range : list with a start and end date for image search.
    mosaick : what to filter the images based off.
    config : configuration set up with API.

    Returns
    -------
    Image which is a list with RGB values for each pixel.

    """
    
    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """
    
    request_true_color = SentinelHubRequest(
    data_folder=folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=(date_range[0], date_range[1]),
            mosaicking_order=mosaick
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=box,
    size=size,
    config=config
    )
    
    image = request_true_color.get_data(save_data=True)
    
    return image

def getCoordinates(coords, box, image):
    
    """
    This function gets the latitude and longitude coordinates for an image list.
    
    Parameters
    ----------
    coords : coordinate list used to make the box
    box : box returne from checkSize.
    image : the image received from getSatelliteImage

    Returns
    -------
    List containing the coordinates for each pixel.

    """
    
    #get the lower left and upper right coordinates of the box
    ll = box.lower_left
    ur = box.upper_right
    #figure out the length of all sides of the box in km
    left_side_distance = mpu.haversine_distance([coords[1],coords[0]],[ll[1],ll[0]])
    right_side_distance = mpu.haversine_distance([coords[3],coords[2]],[ur[1],ur[0]])
    top_distance = mpu.haversine_distance([coords[1],coords[0]],[ur[1],ur[0]])
    bottom_distance = mpu.haversine_distance([coords[3],coords[2]],[ll[1],ll[0]])
    #so the sides are the same distance, but the top and bottom are not, due to the curvature of the earth. 
    #this means that each pixel will have the same height, but will have different widths, depending on how close to the bottom
    #of the image they are. 
    pixel_height = left_side_distance/len(image[0])
    pixel_top_width = top_distance/len(image[0][0])
    pixel_bottom_width = bottom_distance/len(image[0][0])
    #next, I need to find how much the width needs to change per pixel as we go from top to bottom
    width_increment = (pixel_bottom_width-pixel_top_width)/len(image[0])
    #now to get the coordinates for each pixel
    coordinates =[]
    start = [coords[1],coords[0]]
    #loop through each level of height first, get a whole down and then increment the width
    for i in range(len(image[0])):
        if i%100 == 0:
            print(str(i)+" of "+str(len(image[0])))
        width = pixel_top_width + i*width_increment
        #now to loop through each pixel on x-axis
        row = []
        for j in range(len(image[0][0])):
            coor = geopy.distance.geodesic(kilometers=pixel_height*(i+.5)).destination(point=start,bearing=180)
            coor = geopy.distance.geodesic(kilometers=width*(j+0.5)).destination(point=coor,bearing=90)
            row.append(coor)
        coordinates.append(row)
        
    return coordinates

def condensePixels(image,coordinates,grid_width):
    """
    This function condenses the image from high pixel resolution to a lower pixel resolution.
    It creates subgrids of pixels and then summarizes these subgrids into only one pixel.

    Parameters
    ----------
    image : list of image information (rgb values) obtained from getSatelliteImage.
    coordinates : list of coorindates obtained from getCoordinates.
    grid_width : how wide and tall the grids for condensing should be.

    Returns
    -------
    compacted_data : list containing the condensed information.
    compacted_image : list containing only the rgb values, and in proper format to plot.

    """
    gridded_data = []
    for i in range(0,len(image[0])-grid_width,grid_width):
        grid_line = []
        for j in range(0,len(image[0][0])-grid_width,grid_width):
            reds = []
            greens = []
            blues = []
            lats = []
            lons = []
            for m in range(i,i+grid_width):
                for n in range(j,j+grid_width):
                    reds.append(image[0][m][n][0])
                    greens.append(image[0][m][n][1])
                    blues.append(image[0][m][n][2])
                    lats.append(coordinates[m][n][0])
                    lons.append(coordinates[m][n][1])
            grid_line.append([reds,greens, blues,lats,lons])
        gridded_data.append(grid_line)
        
    #Now to go through this gridded list and then create a dataframe that has the most common red, green, and blue per 100
    #pixel grid, and also the average latitude and average longitude
    compacted_data = []
    for i in range(len(gridded_data)):
        new_line = []
        for j in range(len(gridded_data[0])):
            reds_counts = pd.Series(gridded_data[i][j][0]).value_counts()
            common_red = np.uint8(pd.Series(reds_counts[reds_counts==max(reds_counts)].index).mean())
            greens_counts = pd.Series(gridded_data[i][j][1]).value_counts()
            common_green = np.uint8(pd.Series(greens_counts[greens_counts==max(greens_counts)].index).mean())
            blues_counts = pd.Series(gridded_data[i][j][2]).value_counts()
            common_blue = np.uint8(pd.Series(blues_counts[blues_counts==max(blues_counts)].index).mean())
            lat = pd.Series(gridded_data[i][j][3]).mean()
            lon = pd.Series(gridded_data[i][j][4]).mean()
            new_line.append(np.array([common_red, common_green, common_blue, lat, lon]))
        compacted_data.append(np.array(new_line))
        
    #now to create a new image list to test out what the compacted colors look like
    compacted_image = []
    for i in range(len(compacted_data)):
        line = []
        for j in range(len(compacted_data[0])):
            line.append(np.array([np.uint8(compacted_data[i][j][0]),np.uint8(compacted_data[i][j][1]),np.uint8(compacted_data[i][j][2])]))
        compacted_image.append(np.array(line))
    compacted_image = np.array(compacted_image)
    compacted_image = [compacted_image]
        
    return compacted_data, compacted_image

def getUSGSElevations(compacted_data):
    """
    This function gets the elevation for each pixel.

    Parameters
    ----------
    compacted_data : The data set returned from condensePixels.


    Returns
    -------
    elevations : list containing the elevations for each pixel.
    

    """
    #now to get the elevations
    elevations = []
    for i in range(len(compacted_data)):
        if i%10 == 0:
            print(i)
        line = []
        for j in range(len(compacted_data[0])):
            if j%10 == 0:
                print(j)
            el = elevation_function([compacted_data[i][j][3],compacted_data[i][j][4]])
            line.append(el)
        elevations.append(line)
        
    return elevations
        
    
    
def make_remote_request(url: str, params: dict):
    """
    Makes the remote request
    Continues making attempts until it succeeds
    """

    count = 1
    while True:
        try:
            response = requests.get((url + urllib.parse.urlencode(params)))
        except (OSError, urllib3.exceptions.ProtocolError) as error:
            print('\n')
            print('*' * 20, 'Error Occured', '*' * 20)
            print(f'Number of tries: {count}')
            print(f'URL: {url}')
            print(error)
            print('\n')
            count += 1
            continue
        break

    return response

def elevation_function(x):
    url = 'https://nationalmap.gov/epqs/pqs.php?'
    params = {'x': x[1],
              'y': x[0],
              'units': 'Meters',
              'output': 'json'}
    result = make_remote_request(url, params)
    return result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
    

def createFinalCompactedData(compacted_data,elevations):
    """
    This function creates a dataframe that combines the RGB data and the elevations data
    into a dataframe that can be used for analysis

    Parameters
    ----------
    compacted_data : list of compacted data returned from condensePixels.
    elevations : list of elevations from getUSGSElevations.

    Returns
    -------
    final_compacted_data : dataframe of merged data.

    """
    lats = []
    lons = []
    reds = []
    greens = []
    blues = []
    els = []
    for i in range(len(compacted_data)):
        for j in range(len(compacted_data[0])):
            reds.append(compacted_data[i][j][0])
            greens.append(compacted_data[i][j][1])
            blues.append(compacted_data[i][j][2])
            lats.append(compacted_data[i][j][3])
            lons.append(compacted_data[i][j][4])
            els.append(elevations[i][j])
    final_compacted_data = pd.DataFrame({'Lat':lats,'Lon':lons,'Elevation':els,'Red':reds,'Green':greens,'Blue':blues})   
    
    return final_compacted_data

"""-------------------------------------------------------------------------------------------------------------"""

def finalizeData(data, elevations, coords):
    """
    Merges the data file with the elevation and coordinate data and puts it into 2D

    Parameters
    ----------
    data : 3D array
        main data.
    elevations : 3D array
        elevation data.
    coords : 3D array
        coordinates data.

    Returns
    -------
    final_df : dataframe
        2D dataframe with the merges.

    """
    final_data = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            new_line = list(data[i][j])
            new_line.append(elevations[i][j])
            new_line.append(coords[i][j][0])
            new_line.append(coords[i][j][1])
            new_line.append(i)
            new_line.append(j)
            final_data.append(new_line)
        
    final_df = pd.DataFrame(final_data)
    final_df.columns = ['Blue','Green','Red','Red Edge','Band6','Band7','NIR','Band8A','Band9','Band10','SWIR1','Aerosol','SWIR2','Elevation','Lat','Lon','i','j']
    
    return final_df

"""-------------------------------------------------------------------------------------------------------------"""

def plot_image(image, title, factor=1.0, clip_range = None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def getSentinelElevation(bbox, size, date_ranges, config):
    
    
    evalscript_dem = '''
    //VERSION=3
    function setup() {
      return {
        input: ["DEM"],
        output:{
          id: "default",
          bands: 1,
          sampleType: SampleType.FLOAT32
        }
      }
    }

    function evaluatePixel(sample) {
      return [sample.DEM]
    }
    '''

    dem_request = SentinelHubRequest(
        evalscript=evalscript_dem,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.DEM,
                time_interval=(date_ranges[0], date_ranges[1]),
        )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=size,
        config=config
    )

    dem_data = dem_request.get_data()
    
    return dem_data

def getSecondFourBands(box, size,folder,date_range,mosaick,config):
    
    """
    This function sends a request to SentinelHub through the API to get an image of a certain bounding.
    
    Parameters
    ----------
    box : box returne from checkSize.
    size : size returned from checkSize.
    folder : folder location to store image.
    date_range : list with a start and end date for image search.
    mosaick : what to filter the images based off.
    config : configuration set up with API.

    Returns
    -------
    Image which is a list with RGB values for each pixel.

    """
    
    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B05","B06","B07","B08"]
            }],
            output: {
                bands: 4
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B05, sample.B06, sample.B07, sample.B08];
    }
    """
    
    request_true_color = SentinelHubRequest(
    data_folder=folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=(date_range[0], date_range[1]),
            mosaicking_order=mosaick
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=box,
    size=size,
    config=config
    )
    
    image = request_true_color.get_data(save_data=True)
    
    return image


def getThirdFourBands(box, size,folder,date_range,mosaick,config):
    
    """
    This function sends a request to SentinelHub through the API to get an image of a certain bounding.
    
    Parameters
    ----------
    box : box returne from checkSize.
    size : size returned from checkSize.
    folder : folder location to store image.
    date_range : list with a start and end date for image search.
    mosaick : what to filter the images based off.
    config : configuration set up with API.

    Returns
    -------
    Image which is a list with RGB values for each pixel.

    """
    
    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B8A","B09","B10","B11"]
            }],
            output: {
                bands: 4
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B8A, sample.B09, sample.B10, sample.B11];
    }
    """
    
    request_true_color = SentinelHubRequest(
    data_folder=folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=(date_range[0], date_range[1]),
            mosaicking_order=mosaick
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=box,
    size=size,
    config=config
    )
    
    image = request_true_color.get_data(save_data=True)
    
    return image


def getFirstAndLastBands(box, size,folder,date_range,mosaick,config):
    
    """
    This function sends a request to SentinelHub through the API to get an image of a certain bounding.
    
    Parameters
    ----------
    box : box returne from checkSize.
    size : size returned from checkSize.
    folder : folder location to store image.
    date_range : list with a start and end date for image search.
    mosaick : what to filter the images based off.
    config : configuration set up with API.

    Returns
    -------
    Image which is a list with RGB values for each pixel.

    """
    
    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B01","B12"]
            }],
            output: {
                bands: 2
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B01, sample.B12];
    }
    """
    
    request_true_color = SentinelHubRequest(
    data_folder=folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=(date_range[0], date_range[1]),
            mosaicking_order=mosaick
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=box,
    size=size,
    config=config
    )
    
    image = request_true_color.get_data(save_data=True)
    
    return image



def getPSRI(box, size, folder, date_range, mosaick, config):
    evalscript_true_color = """
    //VERSION=3

    let minVal = -0.2;
    let maxVal = 0.4;

    let viz = new HighlightCompressVisualizerSingle(minVal, maxVal);

    function evaluatePixel(samples) {
        let val = (samples.B06 > 0) ? (samples.B04 - samples.B02) / samples.B06 : JAVA_DOUBLE_MAX_VAL;
        return viz.process(val);
    }

    function setup() {
      return {
        input: [{
          bands: [
              "B02",
              "B04",
              "B06"
          ]
        }],
        output: { bands: 1 }
      }
    }
    """
    
    request_true_color = SentinelHubRequest(
    data_folder=folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=(date_range[0], date_range[1]),
            mosaicking_order=mosaick
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=box,
    size=size,
    config=config
    )
    
    image = request_true_color.get_data(save_data=True)
    
    return image

def colorCorrect(box, size,folder,date_range,mosaick,config):
    
    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B04","B03","B02"]
            }],
            output: {
                bands: 3
            }
        };
    }
    
    //== PARAMETERS ===========================
    var c0r = 0.036;   // amount of atmosphere we're compensating
    //var cManual = [0.039, 0.071, 0.121]; // manual white point
    //var cManual = [[0.039, 0.96], [0.071, 0.84], [0.121, 1.34]]; // manual black & white point
    var tx  = 0.2;    // ty/tx ~ contrast in dark areas
    var ty  = 0.4;    // (1-ty)/(1-tx) ~ contrast in light areas
    var max = 3.1;    // reflectance that will become white
    var sat = 1.3;    // saturation enhancement

    var debug = false; // Set to 'true' to highlight out-of-range values

    var atmRatios = [1, 2, 3.25]; // Rayleigh-derived consts for automated atmosphere offsets

    //== FUNCTIONS ============================
    var sRGBenc = C => C < 0.0031308 ? (12.92 * C) : (1.055 * Math.pow(C, 0.41666) - 0.055);

    // atmospheric adjustment
    var atm2p = (a, c0, c1) => (a - c0) / c1;

    var atm1p = (a, c0) => atm2p(a, c0, (1 - c0)**2);

    var atm = (a, ii) => (typeof cManual !== 'undefined')
        ? (cManual[ii] instanceof Array)
            ? atm2p(a, cManual[ii][0], cManual[ii][1])
            : atm1p(a, cManual[ii])
      : atm1p(a, c0r * atmRatios[ii]);

    //contrast enhancement
    var adjFun = (a, tx, ty, max) => {
      var ar = a / max;
      var txr = tx / max;
      var bot = (2 * txr - 1) * ar - txr;
      return ar * (1 + (txr - ty) * (1 - ar) / bot);
    };

    var adj = a => adjFun(a, tx, ty, max);

    var satEnh = rgbArr => {
      var avg = rgbArr.reduce((a, b) => a + b, 0) / rgbArr.length;
      return rgbArr.map(a => avg * (1 - sat) + a * sat);
    };

    var checkDebug = arr => {
      if (!debug) {
        return arr;
      }
      var maxC = Math.max.apply(null, arr);
      var minC = Math.min.apply(null, arr);

      return (minC < 0) // Highlight too dark pixels
         ? arr.map(a => a < 0 ? 1 : 0)
         : (maxC > 1) // Highlight too bright pixels
           ? (minC > 1)
             ? arr.map(a => (a - 1)/(maxC - 1))
             : arr.map(a => a > 1 ? 1 : 0)
           : arr;
    };

    

    function evaluatePixel(sample) {
        var rgb = satEnh([sample.B04,sample.B03,sample.B02].map(atm).map(adj));
        return checkDebug(rgb).map(sRGBenc);
    }
    """
    
    request_true_color = SentinelHubRequest(
    data_folder=folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=(date_range[0], date_range[1]),
            mosaicking_order=mosaick
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=box,
    size=size,
    config=config
    )
    
    image = request_true_color.get_data(save_data=True)
    
    return image

def getLAI(box, size,folder,date_range,mosaick,config):
    
    """
    This function sends a request to SentinelHub through the API to get an image of a certain bounding.
    
    Parameters
    ----------
    box : box returne from checkSize.
    size : size returned from checkSize.
    folder : folder location to store image.
    date_range : list with a start and end date for image search.
    mosaick : what to filter the images based off.
    config : configuration set up with API.

    Returns
    -------
    Image which is a list with RGB values for each pixel.

    """
    
    evalscript_true_color = """
    //VERSION=3 (auto-converted from 2)
    var degToRad = Math.PI / 180;
    
    function evaluatePixelOrig(samples) {
      var sample = samples[0];
      var b03_norm = normalize(sample.B03, 0, 0.253061520471542);
      var b04_norm = normalize(sample.B04, 0, 0.290393577911328);
      var b05_norm = normalize(sample.B05, 0, 0.305398915248555);
      var b06_norm = normalize(sample.B06, 0.006637972542253, 0.608900395797889);
      var b07_norm = normalize(sample.B07, 0.013972727018939, 0.753827384322927);
      var b8a_norm = normalize(sample.B8A, 0.026690138082061, 0.782011770669178);
      var b11_norm = normalize(sample.B11, 0.016388074192258, 0.493761397883092);
      var b12_norm = normalize(sample.B12, 0, 0.493025984460231);
      var viewZen_norm = normalize(Math.cos(sample.viewZenithMean * degToRad), 0.918595400582046, 1);
      var sunZen_norm  = normalize(Math.cos(sample.sunZenithAngles * degToRad), 0.342022871159208, 0.936206429175402);
      var relAzim_norm = Math.cos((sample.sunAzimuthAngles - sample.viewAzimuthMean) * degToRad)
    
      var n1 = neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
      var n2 = neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
      var n3 = neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
      var n4 = neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
      var n5 = neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
    
      var l2 = layer2(n1, n2, n3, n4, n5);
    
      var lai = denormalize(l2, 0.000319182538301, 14.4675094548151);
      return {
        default: [lai / 3]
      }
    }
    
    function neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
      var sum =
    	+ 4.96238030555279
    	- 0.023406878966470 * b03_norm
    	+ 0.921655164636366 * b04_norm
    	+ 0.135576544080099 * b05_norm
    	- 1.938331472397950 * b06_norm
    	- 3.342495816122680 * b07_norm
    	+ 0.902277648009576 * b8a_norm
    	+ 0.205363538258614 * b11_norm
    	- 0.040607844721716 * b12_norm
    	- 0.083196409727092 * viewZen_norm
    	+ 0.260029270773809 * sunZen_norm
    	+ 0.284761567218845 * relAzim_norm;
    
      return tansig(sum);
    }
    
    function neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
      var sum =
    	+ 1.416008443981500
    	- 0.132555480856684 * b03_norm
    	- 0.139574837333540 * b04_norm
    	- 1.014606016898920 * b05_norm
    	- 1.330890038649270 * b06_norm
    	+ 0.031730624503341 * b07_norm
    	- 1.433583541317050 * b8a_norm
    	- 0.959637898574699 * b11_norm
    	+ 1.133115706551000 * b12_norm
    	+ 0.216603876541632 * viewZen_norm
    	+ 0.410652303762839 * sunZen_norm
    	+ 0.064760155543506 * relAzim_norm;
    
      return tansig(sum);
    }
    
    function neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
      var sum =
    	+ 1.075897047213310
    	+ 0.086015977724868 * b03_norm
    	+ 0.616648776881434 * b04_norm
    	+ 0.678003876446556 * b05_norm
    	+ 0.141102398644968 * b06_norm
    	- 0.096682206883546 * b07_norm
    	- 1.128832638862200 * b8a_norm
    	+ 0.302189102741375 * b11_norm
    	+ 0.434494937299725 * b12_norm
    	- 0.021903699490589 * viewZen_norm
    	- 0.228492476802263 * sunZen_norm
    	- 0.039460537589826 * relAzim_norm;
    
      return tansig(sum);
    }
    
    function neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
      var sum =
    	+ 1.533988264655420
    	- 0.109366593670404 * b03_norm
    	- 0.071046262972729 * b04_norm
    	+ 0.064582411478320 * b05_norm
    	+ 2.906325236823160 * b06_norm
    	- 0.673873108979163 * b07_norm
    	- 3.838051868280840 * b8a_norm
    	+ 1.695979344531530 * b11_norm
    	+ 0.046950296081713 * b12_norm
    	- 0.049709652688365 * viewZen_norm
    	+ 0.021829545430994 * sunZen_norm
    	+ 0.057483827104091 * relAzim_norm;
    
      return tansig(sum);
    }
    
    function neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
      var sum =
    	+ 3.024115930757230
    	- 0.089939416159969 * b03_norm
    	+ 0.175395483106147 * b04_norm
    	- 0.081847329172620 * b05_norm
    	+ 2.219895367487790 * b06_norm
    	+ 1.713873975136850 * b07_norm
    	+ 0.713069186099534 * b8a_norm
    	+ 0.138970813499201 * b11_norm
    	- 0.060771761518025 * b12_norm
    	+ 0.124263341255473 * viewZen_norm
    	+ 0.210086140404351 * sunZen_norm
    	- 0.183878138700341 * relAzim_norm;
    
      return tansig(sum);
    }
    
    function layer2(neuron1, neuron2, neuron3, neuron4, neuron5) {
      var sum =
    	+ 1.096963107077220
    	- 1.500135489728730 * neuron1
    	- 0.096283269121503 * neuron2
    	- 0.194935930577094 * neuron3
    	- 0.352305895755591 * neuron4
    	+ 0.075107415847473 * neuron5;
    
      return sum;
    }
    
    function normalize(unnormalized, min, max) {
      return 2 * (unnormalized - min) / (max - min) - 1;
    }
    function denormalize(normalized, min, max) {
      return 0.5 * (normalized + 1) * (max - min) + min;
    }
    function tansig(input) {
      return 2 / (1 + Math.exp(-2 * input)) - 1; 
    }
    
    function setup() {
      return {
        input: [{
          bands: [
              "B03",
              "B04",
              "B05",
              "B06",
              "B07",
              "B8A",
              "B11",
              "B12",
              "viewZenithMean",
              "viewAzimuthMean",
              "sunZenithAngles",
              "sunAzimuthAngles"
          ]
        }],
        output: [
            {
              id: "default",
              sampleType: "AUTO",
              bands: 1
            }
        ]
      }
    }
    
    function evaluatePixel(sample, scene, metadata, customData, outputMetadata) {
      const result = evaluatePixelOrig([sample], [scene], metadata, customData, outputMetadata);
      return result[Object.keys(result)[0]];
    }

    """
    
    request_true_color = SentinelHubRequest(
    data_folder=folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=(date_range[0], date_range[1]),
            mosaicking_order=mosaick
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=box,
    size=size,
    config=config
    )
    
    image = request_true_color.get_data(save_data=True)
    
    return image

def getNDVI(B08, B04):
    NDVI = (B08-B04)/(B08+B04)
    end_NDVI = []
    for ndvi in NDVI:
        if (ndvi<-0.2):
            end_NDVI.append([0,0,0])
        elif (ndvi<-0.1):
            end_NDVI.append([1,0,0])
        elif (ndvi<0):
            end_NDVI.append([0.5,0.6,0])
        elif (ndvi<0.1):
            end_NDVI.append([0.4,0,0])
        elif (ndvi<0.2):
            end_NDVI.append([1,1,0.2])
        elif (ndvi<0.3):
            end_NDVI.append([0.8,0.8,0.2])
        elif (ndvi<0.4):
            end_NDVI.append([0.4,0.4,0])
        elif (ndvi<0.5):
            end_NDVI.append([0.2,1,1])
        elif (ndvi<0.6):
            end_NDVI.append([0.2,0.8,0.8])
        elif (ndvi<0.7):
            end_NDVI.append([0,0.4,0.4])
        elif (ndvi<0.8):
            end_NDVI.append([0.2,1,0.2])
        elif (ndvi<0.9):
            end_NDVI.append([0.2,0.8,0.2])
        else:
            end_NDVI.append([0,0.4,0])
            
    return NDVI, end_NDVI

def getEVI(B08, B04, B02):
    evi = 2.5*(B08-B04)/((B08+6*B04-7.5*B02)+1)
    end_EVI = []
    for EVI in evi:
        if (EVI<-1.1):
            end_EVI.append([0,0,0])
        elif (EVI<-0.2):
            end_EVI.append([0.75,0.75,0.75])
        elif (EVI<-0.1):
            end_EVI.append([0.86,0.86,0.86])
        elif (EVI<0):
            end_EVI.append([1,1,0.88])
        elif (EVI<0.025):
            end_EVI.append([1,0.98,0.8])
        elif (EVI<0.05):
            end_EVI.append([0.93,0.91,0.71])
        elif (EVI<0.075):
            end_EVI.append([0.87,0.85,0.61])
        elif (EVI<0.1):
            end_EVI.append([0.8,0.78,0.51])
        elif (EVI<0.125):
            end_EVI.append([0.74,0.72,0.42])
        elif (EVI<0.15):
            end_EVI.append([0.69,0.76,0.38])
        elif (EVI<0.175):
            end_EVI.append([0.64,0.8,0.35])
        elif (EVI<0.2):
            end_EVI.append([0.57,0.75,0.32])
        elif (EVI<0.25):
            end_EVI.append([0.5,0.7,0.28])
        elif (EVI<0.3):
            end_EVI.append([0.44,0.64,0.25])
        elif (EVI<0.35):
            end_EVI.append([0.38,0.59,0.21])
        elif (EVI<0.4):
            end_EVI.append([0.31,0.54,0.18])
        elif (EVI<0.45):
            end_EVI.append([0.25,0.49,0.14])
        elif (EVI<0.5):
            end_EVI.append([0.19,0.43,0.11])
        elif (EVI<0.55):
            end_EVI.append([0.13,0.38,0.07])
        elif (EVI<0.6):
            end_EVI.append([0.06,0.33,0.04])
        else:
            end_EVI.append([0,0.27,0])
    return evi, end_EVI

def convertToSingle(data):
    new_line = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            new_line.append(data[i][j])
    return new_line


def combine(items):
    return_array = []
    for i in range(len(items[0])):
        new_column = []
        for j in range(len(items[0][0])):
            combined_line = np.array(np.uint8([]))
            for item in items:
                combined_line = np.concatenate((combined_line,item[i][j]))
            new_column.append(combined_line)
        return_array.append(np.array(new_column))
    return np.array(return_array)

def formatElevation(elevations):
    new_elevations = []
    for i in range(len(elevations[0])):
        new_column = []
        for j in range(len(elevations[0][0])):
            new_column.append(np.array([np.uint8(elevations[0][i][j])]))
        new_elevations.append(np.array(new_column))
    return [np.array(new_elevations)]


def plot_image(image, title, return_ax,ax = None, factor=1.0, clip_range = None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    if not return_ax:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    
    if return_ax:
        return ax

    
def formatToImage(data,scaler):
    if not scaler:
        return_image = np.empty([max(data['i'])+1,max(data['j'])+1,len(data.columns)-2])
        for k in range(data.shape[0]):
            l = 0
            for m in list(data.columns)[:len(data.columns)-2]:
                return_image[data['i'][k]][data['j'][k]][l] = np.uint8(data[m][k])
                l += 1
    else:
        return_image = np.empty([max(data['i'])+1,max(data['j'])+1])
        for k in range(data.shape[0]):
            return_image[data['i'][k]][data['j'][k]] = int(data[data.columns[0]][k])

    return return_image

def density(comp,boxes1,boxes2):
    """
    This function creates density data by breaking the data into boxes and counting the number of values within that box
    PARAMETERS
    comp: dataframe
        the data frame for which to get the densities - only first two columns are used (and columnsmust be labeled 0 and 1)
    boxes1: int
        this is how many boxes the first column should get broken into
    boxes2: int
        this is how many boxes the second column should get broken into
        
    RETURNS
    dataframe of the density counts for each of the boxes
    """
    min1 = min(comp[0])
    max1 = max(comp[0])
    min2 = min(comp[1])
    max2 = max(comp[1])
    
    l1 = (max1+0.0000001 - min1)/boxes1
    l2 = (max2+0.0000001 - min2)/boxes2
    
    i_range = []
    j_range = []
    counts = []
    for i in range(boxes1):
        print(i)
        for j in range(boxes2):
            comp_sub = comp[comp[0].between(min1+i*l1,min1+(i+1)*l1)]
            comp_sub = comp_sub[comp_sub[1].between(min2+j*l2,min2+(j+1)*l2)]
            counts.append(comp_sub.shape[0])
            i_range.append((min1+i*l1 + min1+(i+1)*l1)/2)
            j_range.append((min2+j*l2 + min2+(j+1)*l2)/2)
    return pd.DataFrame({'i':i_range,'j':j_range,'Counts':counts})


def addGroups(existing_groups, core_indeces, labels, df, to_use, epsilon):
    """
    This function merges clusters found in an instance of DBSCAN on a data split with pre-existing clusters from prior instances
    of DBSCAN on other splits. It compares the core samples to see whether they are within 'Epsilon' distance of each other.
    If they are further than epsilon away, then the cluster is not merged with a pre-existing cluster.
    PARAMETERS:
    existing_groups: list
        contains a list of the pre-existing clusters found already - each item in the list is a dataframe of the clusters core samples
    core_indeces: numpy array
        contains the indices of the core samples found from the latest instance of DBSCAN
    labels: numpy array
        contains the cluster labels for all observations from the latest instance of DBSCAN
    df: pandas dataframe
        the data used in the latest instance of DBSCAN
    to_use: list
        contains the column names from df used in clustering (usually all of them except 'i' and 'j' locator variables)
    epsilon: float
        the epsilon value used for running the latest instance of DBSCAN
        
    RETURNS:
    existing_groups: list
        each element of the list is a dataframe containing the updated data contained in a cluster - this is the input 
        existing_groups but after merges with current instance of DBSCAN has been done
    """
    #create dataframe containing the labels and indexed by the accompanying indeces in the df
    new_cores_indeces = pd.DataFrame(labels[core_indeces],index=core_indeces)
    new_cores_indeces.columns = ['Label']
    #create subset of df to only have the cores
    new_cores_data = df.loc[new_cores_indeces.index,:]
    #if no groups already in, then nothing to compare to so just add the groups in (except the noise)
    if len(existing_groups) == 0:
        for j in np.unique(new_cores_indeces['Label']):
            if j != -1:
                #print("j = "+str(j))
                #print(new_cores_indeces['Label'])
                #print(new_cores_indeces[new_cores_indeces['Label'] == j])
                #print(new_cores_data.loc[new_cores_indeces[new_cores_indeces['Label']==j].index,:])
                existing_groups.append(new_cores_data.loc[new_cores_indeces[new_cores_indeces['Label']==j].index,:])
    #if there are already groups existing then compare to each one to see what to add it to or whether it's a new group
    else:
        total_merges = []
        for j in np.unique(new_cores_indeces['Label']):
            if j != -1:
                merge_groups = []
                i = 0
                #print('Lenght'+str(len(existing_groups[0])))
                for g in existing_groups:
                    #print('Group '+str(i))
                    #print('g')
                    #print(g[to_use])
                    #for k in range(g[to_use].shape[0]):
                    k = 0
                    #####
                    p1 = g.iloc[k,to_use]
                    new_data = new_cores_data.loc[new_cores_indeces[new_cores_indeces['Label']==j].index,:]
                    for l in range(new_data.shape[0]):
                        p2 = new_data.iloc[l,to_use]
                        #if math.dist(p1,p2) <= epsilon:
                        #print('P1')
                        #print(p1)
                        #print('P2')
                        #print(p2)
                        if np.linalg.norm(p1-p2) <= epsilon:
                            if i not in merge_groups:
                                merge_groups.append(i)
                    #print("first p1 value")
                    i += 1
            #now to merge all groups in the 'merge_groups' list
            #print(merge_groups)
            if len(merge_groups) > 0:
                #print('starting merge')
                new_merge = existing_groups[merge_groups[0]]
                #print(new_merge.shape)
                new_merge = new_merge.append(new_data)
              
                del existing_groups[merge_groups[0]]
                subtraction = 1
                if len(merge_groups) > 1:
                    for m in merge_groups[1:]:
                        #print(str(m)+' Merge')
                        #print(new_merge.shape)
                        #print(existing_groups[m].shape)
                        new_merge.append(existing_groups[m-subtraction])
                        del existing_groups[m-subtraction]
                        subtraction += 1
                #print('new merge size')
                #print(new_merge.shape)
                existing_groups.append(new_merge)
            #if there was nothing to merge, then it is a unique group so just append it
            else:
                existing_groups.append(new_cores_data.loc[new_cores_indeces[new_cores_indeces['Label']==j].index,:])
    return existing_groups
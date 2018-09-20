## This file contains very useful functions for image analysis

# importing useful modules (some of these might not be used below...)
import numpy as np
from numpy.linalg import *
import pandas as pd
import scipy as sp
from scipy.spatial.distance import pdist
from scipy.spatial import distance
from scipy.stats import rankdata
from scipy import stats
import math
import time
import shutil
from IPython.display import display, HTML

# matplotlib stuff
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as col
from matplotlib import ticker
import pylab as py

# scientific and computing stuff
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.signal import argrelextrema
import numpy as np
import numdifftools as nd
from itertools import combinations, permutations, product
from sklearn import mixture
import random
import re
import os
import ast

# necessary for image analysis in python
import sys
import scipy.ndimage as ndimage
import skimage
import skimage.measure
import skimage.io
from skimage.filters import threshold_otsu
import tifffile as tiff

# colorbrewer2 Dark2 qualitative color table
import brewer2mpl
dark2_cmap = brewer2mpl.get_map('Dark2', 'Qualitative', 7)
dark2_colors = dark2_cmap.mpl_colors

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 1000
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'Helvetica'
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

# suppress annoying "deprecation warnings"
import warnings
warnings.filterwarnings("ignore")


##########################################################################################


### setting plotting parameters

# procedure to minimize chartjunk by stripping out unnecessary plot borders and axis ticks    
# the top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
def remove_border(axes=None, top=False, right=False, left=True, bottom=True):

    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    # turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    # now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        
        
# procedure to take away tick marks, but leave plot frame
def yes_border_no_ticks(axes=None, border=True):
    
    ax = axes or plt.gca()
    
    # removing frame and ticks
    remove_border(ax, left=False, right=False, top=False, bottom=False)
    
    if border:
        # putting frame back
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)


# making procedure to set some preferred plot parameters
# 'fig' is the figure name
# 'ax' is an axes handle
# if numbers are in xlabels, make sure they are integers and not strings
# 'ylim' is a list with a minimum value and a maximum value
# if no ylim desired, insert empty list
def set_plot_params(fig=None, ax=None, xlabels=[], xlim=[], ylim=[], 
                    tick_size=17, label_size=25, legend_size=15, 
                    legend_loc='best', all_axis_lines=False):

    if fig == None:
        fig = plt.gcf()
    if ax == None:
        ax = plt.gca()
    
    # adjusting thickness of axis spines
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    
    # adjusting thickness of ticks
    ax.tick_params('both', width=2, which='major')
    
    # this positions the xtick labels so that the last letter is beneath the tick
    #fig.autofmt_xdate()
    
    if len(xlabels) != 0:
        
        ##
        if not isinstance(xlabels[0], str):
            ax.set_xticks(xlabels)   # placing the xticks
        else:
            ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels([str(i) for i in xlabels], rotation=0)
    
  
    ax.tick_params(axis='both', which='major', labelsize=tick_size)     # adjusting tick label size
    ax.xaxis.label.set_fontsize(label_size)                             # adjusting x-fontsize
    ax.yaxis.label.set_fontsize(label_size)                             # adjusting y-fontsize
    ax.title.set_fontsize(label_size)
    
    # if a y-limit list is inputted, setting limits for y-axis
    if len(ylim) != 0:
        ax.set_ylim(ylim)
        
    if len(xlim) != 0:
        ax.set_xlim(xlim)
    
    # adding legend and prettifying figure
    if isinstance(legend_loc, str):
    	plt.legend(loc=legend_loc, prop={'size':legend_size}, 
               	   fancybox=True, framealpha=0.5, numpoints=1, frameon=0)
    else:
    	plt.legend(prop={'size':legend_size}, bbox_to_anchor=legend_loc,
               	   fancybox=True, framealpha=0.5, numpoints=1, frameon=0)    	
  
    if all_axis_lines:
        remove_border(axes=ax, top=1, right=1)
    else:
        remove_border(axes=ax)
    
    
##########################################################################################


# function to process image data
# no longer computes fold-change (except for comets); if desired, do this in plot_image_data()
# 'method' can be 'mt', 'comets', 'feret' (i.e. comet length), or 'area'
# 'time_min' can be an array, a number, or None
def process_image_data(df, time_min, method='mt'):
    
    input_df = df.copy()
    
    # sort by field then frame just in case
    if time_min != None:
        input_df = input_df.sort(['Field', 'Frame']).reset_index(drop=True)
    else:
        input_df = input_df.sort('Field').reset_index(drop=True)
    
    fields_present = input_df.Field.unique() # fields present
    field_num = len(fields_present)          # number of fields
    
    ## Restructure dataframe for each field

    # intialize dataframe
    quant_df = pd.DataFrame()

    # repeat for remaining fields
    for field in fields_present:

        # get mt or comets, and cell area measurements
        if method.lower() == 'mt':
            quant_col = 'MT_IntDen'
        elif method.lower() == 'comets':
            quant_col = 'Comet'
        elif method.lower() == 'feret':
            quant_col = 'Feret'
        else:
            quant_col = 'Cell_Area'
            
        quanti = np.array(input_df[quant_col][input_df.Field == field].tolist())

        # check if all values are nan --> means nothing was detected
        if np.all(np.isnan(quanti), 0):
            quanti = np.zeros((len(quanti)))

        quant_df[quant_col + str(int(field))] = quanti
              
    # calculate means and sems
    quant_mean = quant_df.ix[:, :field_num].apply(lambda x: 
                np.nanmean(np.array(x)), axis=1)
    quant_median = quant_df.ix[:, :field_num].apply(lambda x: 
                np.nanmedian(np.array(x)), axis=1)    
    quant_sem = quant_df.ix[:, :field_num].apply(lambda x: 
                np.nanstd(np.array(x))/np.sqrt(field_num), axis=1)
    quant_min = quant_df.ix[:, :field_num].apply(lambda x: 
                np.nanmin(np.array(x)), axis=1)
    quant_max = quant_df.ix[:, :field_num].apply(lambda x: 
                np.nanmax(np.array(x)), axis=1)
    
    quant_df[quant_col+'_Mean'] = quant_mean
    quant_df[quant_col+'_Median'] = quant_median
    quant_df[quant_col+'_SEM'] = quant_sem
    quant_df[quant_col+'_Min'] = quant_min
    quant_df[quant_col+'_Max'] = quant_max

    # adding time column
    # checking if time array was inputted
    if isinstance(time_min, np.ndarray):
        quant_df['Time_Min'] = time_min
    elif time_min == None:
        pass
    else:
        quant_df['Time_Min'] = np.arange(len(quant_df)) * 1.0 * time_min / (len(quant_df)-1)
    
    return quant_df
    

##########################################################################################
    

### code to quantify SiR-tubulin in cells

# 'image_path': directory path to image; or image in numpy array format
# 'method' can be 'area' for cell area, 'mts'
# 'gmix_comps' can be an integer or an array of integers (to optimize number of components)
# in the case of a 'gmix_comps' array, user can return the opimized number of components
# 'first_guass' is useful if the user wants to get the mean background intensity
def sir_autothresh(image_path, gmix_comps=np.arange(3)+2, method='area', plot=False, 
                   bin2x2=True, sub_image=False):

    # reading in image
    if isinstance(image_path, np.ndarray):
        orig_image = image_path
    else:
        orig_image = skimage.io.imread(image_path, plugin='tifffile')

    if not sub_image:
        bit_num = 8
    else:
        bit_num = 4
    
    # apply gaussian blur if quantifying cell area (i.e. blur microtubules)
    if 'area' in method.lower():
        if bin2x2:
            sig = 2.5
        else:
        	sig = 10.0
        
        gauss_10 = ndimage.gaussian_filter(orig_image, sigma=sig, order=0)
    else:
        gauss_10 = orig_image.copy()
        
    # linearizing image
    gauss_10_1d = gauss_10.ravel()
    gauss_10_1d_copy = gauss_10_1d.copy()
        
    # shuffling pixels
    np.random.shuffle(gauss_10_1d)
    
    if isinstance(gmix_comps, list) | isinstance(gmix_comps, np.ndarray):
        # apply gaussian mixture model
        models = []
        for c in gmix_comps:
            
            models.append(mixture.GMM(int(c)).fit(gauss_10_1d[::2**bit_num, np.newaxis]))
            
        # compute the "Akaike information criterion" (AIC)
        # smaller is better; take best model
        AIC = [m.aic(gauss_10_1d[::2**bit_num, np.newaxis]) for m in models]
        gmm = models[np.argmin(AIC)]            # optimized model
        opt_comps = gmix_comps[np.argmin(AIC)]  # optimized number of components
    
    # number of components determined by user
    else:
        gmm = mixture.GMM(int(gmix_comps)).fit(gauss_10_1d[::2**bit_num, np.newaxis])
        opt_comps = int(gmix_comps)
    
    # getting the means pertaining to the model gaussians
    model_means = sorted(sum(gmm.means_.tolist(), []))
    
    # now take the mean of the second gaussian (after background gaussian)
    if len(model_means) > 1:
        gmix_thresh = int(model_means[1])  # take means
    
    else:
        gmix_thresh = int(model_means[0])  # unless we only have one gaussian
    
    if plot:
        # numpy histogram
        counts, bin_edges = np.histogram(gauss_10_1d_copy, bins=2**bit_num)    
        bin_edge_size = bin_edges[1]-bin_edges[0]
        bin_means = bin_edges[:-1] + bin_edge_size/2.0

        # getting gaussian curves for plotting
        logprob, responsibilities = gmm.score_samples(bin_means.copy().reshape((-1, 1)))
        pdf = np.exp(logprob)
        
        pdf = pdf*np.sum(counts)*1.0/np.sum(pdf)  # sum normalization
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        # setting axis limits
        if len(model_means) > 1:
            xlim=[0, model_means[-1]+model_means[-2]]
        else:
            xlim=[0, 3*model_means[0]]
            
        ylim=[0, np.max([np.max(counts), np.max(pdf_individual)])]
        
        # plotting
        plt.figure()
        plt.hist(gauss_10_1d_copy, bins=2**bit_num, edgecolor='k')
        plt.grid()
        plt.plot(bin_means, pdf_individual, 'k-', linewidth=3.0)
        
        # autothresholding using Otsu method
        otsu_thresh = threshold_otsu(gauss_10)
        plt.plot(np.ones(11)*gmix_thresh, np.arange(11)*ylim[1]/10, 
                 'r--', linewidth=3.0, 
                 label='Gmix Threshold: '+str(gmix_thresh))
        plt.plot(np.ones(11)*otsu_thresh, np.arange(11)*ylim[1]/10, 
                 '--', color=(0.5,0.5,0.5), linewidth=3.0, 
                 label='Otsu Threshold: '+str(otsu_thresh))
        plt.legend(loc='upper right')
        plt.xlabel('Thresholds')
        plt.ylabel('Pixel Counts')
        set_plot_params()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
    
    # only need to output optimum number of gaussian components for 'area' in case every frame is thresholded
    if 'area' in method.lower():
        
        # calculating area
        gauss_10[gauss_10 < gmix_thresh] = 0
        gauss_10[gauss_10 >= gmix_thresh] = 1
        area = np.sum(np.sum(gauss_10))
        
        return area, opt_comps
    
    else:
        return gmix_thresh

    
# 'method' can be 'mt' or 'both' or 'bkg'
# 'well_info' should be a string containing row and column info (e.g. 'C - 01')
# 'field_prec' and 'frame prec' strings must be appear only once in the image filenames
# 'min_area_frame' is the frame used to determine removable fields (i.e. 'first' or 'last')
# 'time_prec' or 'time_proc' can equal None if no time-course is desired --> single frame per field is assumed
# all inputted/outputted time parameters are in minutes
# by default, this function assumes that time is present in millisec in the filename
def sir_auto_quant(image_dir, well_info, method='both', 
        field_prec='fld ', field_proc='- time', frame_prec='time ', frame_proc=' - ', 
        time_prec=' - ', time_proc=' ms)', dead_time=0.0, default_time=60.0, 
        gmix_comps=3, min_area_per=10.0, thresh_frame='first', 
        plot=0, im_file_list=[], bin2x2=True, sub_image=False):
    
    
    ## verification of inputs section

    if len(im_file_list) > 0:
    
        # drug condition of interest
        drug_im_files = [f for f in im_file_list if well_info.lower() in f.lower()]
    
    else:
        # drug condition of interest
        drug_im_files = [f for f in os.listdir(image_dir) if ('.tif' in f.lower()) & 
                                                         (well_info.lower() in f.lower())]

    # verify we actually pulled images
    if len(drug_im_files) == 0:
        print 'No images were found when using the well-info string provided...'
        return
    
    ## get pixel area of an entire image
    
    tot_pix_area = len(skimage.io.imread(image_dir+drug_im_files[0], plugin='tifffile').ravel())
    
    
    ## getting max and min frame number
    
    if (time_prec != None) & (time_proc != None):
        max_frame = 0     # assuming we have at least 1 frame
        min_frame = 1000  # assuming we have fewer than 1000 frames
        for im_file in drug_im_files:

            # getting index positions of frame designators      
            frame_prec_idx = im_file.index(frame_prec)
            frame_proc_idx = im_file[frame_prec_idx:].index(frame_proc)+frame_prec_idx

            # getting frame info
            frame = int(im_file[frame_prec_idx+len(frame_prec):frame_proc_idx])

            if frame > max_frame:
                max_frame = frame

            if frame < min_frame:
                min_frame = frame

        frame_num = max_frame - min_frame + 1
    
    
    ## presetting arrays section
    
    # some of these won't be used until later
    mt_intdens = np.zeros(len(drug_im_files))
    cell_areas = np.zeros(len(drug_im_files))
    field_assign = np.zeros(len(drug_im_files)) # will need to re-order later
    frame_assign = np.zeros(len(drug_im_files)) # will need to re-order later
    times = np.zeros(len(drug_im_files))        # times are in msec by default
    
    
    ## area percent calculation section
    
    # determine area proportion of total image area
    # also determine microtubule thresholds if quantifying polymer
    rep_areas = np.zeros(100)-1     # assuming fewer than 100 fields...
    area_comps = np.zeros(100)-1
    area_percents = np.zeros(100)-1
    already_quant_idx = []
    ignore_fields = []
    for i, im_file in enumerate(drug_im_files):
        
        # getting index positions of field and frame designators
        field_prec_idx = im_file.index(field_prec)
        field_proc_idx = im_file[field_prec_idx:].index(field_proc)+field_prec_idx
        field = int(im_file[field_prec_idx+len(field_prec):field_proc_idx])
        
        if (time_prec != None) & (time_proc != None):
            frame_prec_idx = im_file[field_proc_idx:].index(frame_prec)+field_proc_idx
            frame_proc_idx = im_file[frame_prec_idx:].index(frame_proc)+frame_prec_idx
            frame = int(im_file[frame_prec_idx+len(frame_prec):frame_proc_idx])
        else:
            frame = 0
            min_frame = 0
            max_frame = 0
        
        # only using first frame to determine threshold
        if ((frame == min_frame) & (thresh_frame == 'first')) | ((frame == max_frame) & (thresh_frame == 'last')):
            
            # get cell area threshold and area measurement for each image
            rep_areas[field-1], area_comps[field-1] = sir_autothresh(image_dir+im_file, 
                                                    gmix_comps, 'area', bin2x2=bin2x2, sub_image=sub_image)
            area_percents[field-1] = rep_areas[field-1]*100.0 / tot_pix_area
            
            # notate sub-optimal images
            if area_percents[field-1] < min_area_per*1.0:
                ignore_fields.append(field)
                
                # don't quantify cell area or microtubules
                cell_areas[i] = np.nan
                
            else:
                cell_areas[i] = rep_areas[field-1]
                
            # noting the image as already quantified
            already_quant_idx.append(i)

    # get rid of unnecessary zeros
    # note: each position corresponds to a respective field position
    rep_areas = rep_areas[rep_areas > -1]
    area_comps = area_comps[area_comps > -1]
    area_percents = area_percents[area_percents > -1]
    
    if len(area_percents) <= 9:
        print 'Area percents: %s' % str(area_percents)
      

    ## autothresholding section
    
    for i, im_file in enumerate(drug_im_files):
        
        # getting index positions of field and frame designators
        field_prec_idx = im_file.index(field_prec)
        field_proc_idx = im_file[field_prec_idx:].index(field_proc)+field_prec_idx
        field = int(im_file[field_prec_idx+len(field_prec):field_proc_idx])
        field_assign[i] = field
        
        
        if (time_prec != None) & (time_proc != None):
            frame_prec_idx = im_file[field_proc_idx:].index(frame_prec)+field_proc_idx
            frame_proc_idx = im_file[frame_prec_idx:].index(frame_proc)+frame_prec_idx
            frame = int(im_file[frame_prec_idx+len(frame_prec):frame_proc_idx])
            frame_assign[i] = frame
        
            # for the InCELL Analyzer, time comes after timepoint, which follows frame
            time_prec_idx = im_file[frame_proc_idx:].index(time_prec)+frame_proc_idx
            time_proc_idx = im_file[time_prec_idx:].index(time_proc)+time_prec_idx
            time_msec = int(im_file[time_prec_idx+len(time_prec):time_proc_idx])
            times[i] = time_msec
        
        # quantifying cell area
        if method.lower() == 'both':
            
            if i in already_quant_idx:
                pass
            elif field in ignore_fields:
                cell_areas[i] = np.nan
            else:
                # use optimized number of gmix components determined above
                cell_areas[i], redund_comps = sir_autothresh(image_dir+im_file, area_comps[field-1], 'area', 
                                                             bin2x2=bin2x2, sub_image=sub_image)
 
        if field in ignore_fields:
            mt_intdens[i] = np.nan
        else:
                
            # reading in image to threshold and quantify polymer integrated density
            sirt_image = skimage.io.imread(image_dir+im_file, plugin='tifffile')

            # thresholding MTs
            mt_thresh = sir_autothresh(image_dir+im_file, area_comps[field-1], 'mt', 
                                      bin2x2=bin2x2, sub_image=sub_image)
            
            sirt_image[sirt_image > 0.99*(2**16)] = 0  # removing saturated pixels
            sirt_image[sirt_image < mt_thresh] = 0

            if method.lower() == 'both':
                mt_intdens[i] = np.sum(np.sum(sirt_image))

            # if only quantifying mt, at least normalize by representative area to be safe
            elif method.lower() == 'mt':
                mt_intdens[i] = np.sum(np.sum(sirt_image))*1.0 / rep_areas[field-1]
            elif method.lower() == 'bkg':
                mt_intdens[i] = np.sum(np.sum(sirt_image))*1.0 / (tot_pix_area - rep_areas[field-1])
            

    # recording MT quant data in dataframe
    mt_df = pd.DataFrame(mt_intdens, columns=['MT_IntDen'])
    mt_df['Field'] = field_assign
    
    if (time_prec != None) & (time_proc != None):
        mt_df['Frame'] = frame_assign
        mt_df['Time_Min'] = times *1.0/60000  # convert times from millisec to min
        mt_df = mt_df.sort(['Field', 'Frame']).reset_index(drop=True)

        # now re-define time array and drop from df
        # this is necessary for compatibility with process_image_data()
        times = np.array(mt_df.Time_Min.tolist())
        times = times[:frame_num]  # truncating to make compatible with process_image_data()

        times = times + dead_time  # adding dead-time
        times[0] = 0.0             # fix first timepoint
        
        mt_df = mt_df.drop('Time_Min', axis=1) 
    else:
        times = None

    if method.lower() == 'both':
        area_df = pd.DataFrame(cell_areas, columns=['Cell_Area'])
        area_df['Field'] = field_assign
        
        if (time_prec != None) & (time_proc != None):
            area_df['Frame'] = frame_assign
            area_df = area_df.sort(['Field', 'Frame']).reset_index(drop=True)
        else:
            area_df = area_df.sort('Field').reset_index(drop=True)
      
    # if cell area not quantified
    if method.lower() != 'both':
        mt_df = process_image_data(mt_df, times, 'mt')
        return mt_df

    # if returning both MTs and cell area, normalize MTs by cell area
    # also, process and return cell area dataframe
    else:
        mt_df['Cell_Area'] = area_df.Cell_Area
        mt_df['MT_IntDen'] = mt_df.MT_IntDen / mt_df.Cell_Area
        mt_df = mt_df.drop('Cell_Area', axis=1)
        mt_df = process_image_data(mt_df, times, 'mt')
        
        area_df = process_image_data(area_df, times, 'area')
        return mt_df, area_df

    
# function to split an image into nxm images
def split_image(image_path, row_split=4, col_split=4):
    
    # reading in image
    orig_image = skimage.io.imread(image_path, plugin='tifffile')
    
    # getting dimensions of image
    rows, cols = np.shape(orig_image)
    
    # verify desired splits are compatible with image dimensions
    if rows % row_split != 0:
        print 'Desired row split is incompatible with image dimensions...'
        return
    elif cols % col_split != 0:
        print 'Desired column split is incompatible with image dimensions...'
        return
    
    # getting coordinates for splits
    row_idc = [rows/row_split*i for i in range(row_split)]+[rows]
    col_idc = [cols/col_split*i for i in range(col_split)]+[rows]
    
    row_coord = [[row_idc[i], row_idc[i+1]] for i in range(row_split)]
    col_coord = [[col_idc[i], col_idc[i+1]] for i in range(col_split)]
    
    
    # getting coordinates for sub-images to be produced
    sub_im_coord = []
    for row_c in row_coord:
        for col_c in col_coord:
            sub_im_coord.append([row_c, col_c])

    # performing image split
    sub_image_list = []
    for coord in sub_im_coord:
        
        row_start = coord[0][0]
        row_end = coord[0][1]
        col_start = coord[1][0]
        col_end = coord[1][1]
        
        sub_image = orig_image[row_start:row_end, col_start:col_end]
        sub_image_list.append(sub_image)
        
    return sub_image_list


# adding image filename specification (i.e. a string designator)
# NOTE: do not make 'output_dir' the same as 'image_dir' or files will be overwritten
def make_subimages(image_dir, output_dir=[], row_split=4, col_split=4, 
                   field_prec='(fld ', field_proc='- time', image_spec=''):
    
    # verifying compatible input parameters
    if image_dir[-1] != '/':
        image_dir = image_dir + '/'
    
    if isinstance(output_dir, list):
        output_dir = image_dir+'../sub_images/'
        os.mkdir(output_dir)
    elif output_dir[-1] != '/':
        output_dir = output_dir + '/'
        
    row_split = int(row_split)
    col_split = int(col_split)
        

    # gets all tif files in the input directory
    all_im_files = [f for f in os.listdir(image_dir) if '.tif' in f.lower()]
    
    # utilize specification
    all_im_files = [im for im in all_im_files if image_spec in im]
    
    for im in all_im_files:
        
        # getting image attributes
        im_start = im[:im.index(field_prec)+len(field_prec)]
        field = int(im[im.index(field_prec)+len(field_prec):im.index(field_proc)])
        im_end = im[im.index(field_proc):]
        
        # resetting field names
        new_fields = [row_split*col_split*(field-1)+i+1 for i in range(row_split*col_split)]

        sub_images = split_image(image_dir+im, row_split, col_split)
        
        # writing out sub-image files
        for idx in range(row_split*col_split):
            
            sub_im_name = im_start + str(new_fields[idx]) + im_end
            sub_im = sub_images[idx]
            
            tiff.imsave(output_dir+sub_im_name, sub_im)


# function to merge SirT biorep data
# if user desires all time points, set 'equil_time=[]'
# 'sirt_ctrl': rounded SirT concentration that should be used to normalize and merge plates
def sir_biorep(path_list, equil_time=[], sirt_ctrl=800.0):
    
    df_list = [pd.read_csv(path) for path in path_list]
    
    # determine biggest dataset (may have more time points or conditions, etc.)
    len_list = [len(df) for df in df_list]
    max_idx = np.argsort(len_list)[-1]
    
    if not isinstance(equil_time, list):
        df_list = [df[df.Time_Min == equil_time].reset_index(drop=True) for df in df_list]
    else:
        equil_time=600
    
    # get unique drug conditions that are present in all biorepeats
    # use 'Drug' and 'Drug_Conc_nM' columns to be safe
    unq_cond = df_list[max_idx][['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1).tolist()
    for df in df_list:
        unq_cond_i = df[['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1).tolist()
        unq_cond = set(unq_cond_i).intersection(unq_cond)  # get intersection
        
    df_norm_list = []
    for df in df_list:
        
        # only take those conditions that are present in all the datasets
        df['Key_Col'] = df[['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1)
        df = df[df.Key_Col.isin(unq_cond)].reset_index(drop=True)
        
        # 'Drug_Conc_nM' has been converted into strings since it has been read out as *.csv
        # this is due to the 'DMSO' rows that are noted as '0.01%'
        df['Drug_Conc_nM'] = df.Drug_Conc_nM.apply(lambda x: float(x) if '%' not in x else x)
        
        # sort appropriate columns
        df = df.sort(['Drug', 'Drug_Conc_nM', 'SirT_Conc_nM']).reset_index(drop=True)

        # will normalize by negative control intensity (i.e. 800nM SirT + DMSO)        
        mean_sirt_val = df.MT_IntDen_Mean[(df.Drug == 'DMSO') & (df.Time_Min == equil_time)].values
        
        if len(mean_sirt_val) > 1:  # if more than one concentration of SirT is present in the dataset
            sirt_conc = df.SirT_Conc_nM.unique().tolist()
            ctrl_pos = sirt_conc.index(sirt_ctrl)
            mean_sirt_val = mean_sirt_val[ctrl_pos]

        df['MT_IntDen_Mean_Norm'] = df.MT_IntDen_Mean / mean_sirt_val
        df['MT_IntDen_SEM_Norm'] = df.MT_IntDen_SEM / mean_sirt_val
        
        df_norm_list.append(df)
    
    biorep_df =  df_norm_list[max_idx].copy()
    biorep_df['MT_IntDen_Mean'] = np.nanmean([df.MT_IntDen_Mean for df in df_norm_list], 0)
    biorep_df['MT_IntDen_SEM'] = np.sqrt(np.nansum([df.MT_IntDen_SEM**2 for df in df_norm_list], 0)) / len(path_list)
    
    biorep_df['MT_IntDen_Mean_Norm'] = np.nanmean([df.MT_IntDen_Mean_Norm for df in df_norm_list], 0)
    biorep_df['MT_IntDen_SEM_Norm'] = np.sqrt(np.nansum([df.MT_IntDen_SEM_Norm**2 for df in df_norm_list], 0)) / len(path_list)
    
    # drop unnecessary column
    biorep_df = biorep_df.drop('Rep_Wells', 1).drop('Key_Col', 1)
    
    return biorep_df
    
    
# function that computes elemental multiplation for two lists
# lists should be of equal length
def element_prod(list1, list2):
    
    if len(list1) != len(list2):
        return None
    
    prod_list = np.zeros(len(list1)).tolist()
    for i in range(len(list1)):
        e1 = list1[i]
        e2 = list2[i] 
        prod_list[i] = e1*1.0*e2
        
    return prod_list


# takes report file from D300 dispenser, organizes the well info, and merges replicates
# NOTE: 'report_df' can be a string leading to a manually created report file
# 'sirt_list': list of quantified well dataframes
# 'area_list': list of quantified well dataframes
# 'well_list': list of quantified wells; if inputted, will verify wells are in D300 report
def aggregate_ALL_sirt(report_df, sirt_list, area_list, well_list=[], pctrl_conc=800.0):
    
    # processing inputted well list
    if len(well_list) > 0:
        well_list = [w.split(' - ')[0] + w.split(' - ')[1] for w in well_list]
    
    sort_cols = ['Drug', 'Drug_Conc_nM']
    
    # if a d300 report file is inputted
    if not isinstance(report_df, str):
        
        sort_cols += ['SirT_Conc_nM']


        ### organizing well info

        # first truncation: look for "Tabular detail"
        start_row = report_df.ix[:, 0].tolist().index('Tabular detail')
        report_df = report_df.ix[start_row+1:, :].reset_index(drop=True)

        cols = report_df.columns  # getting column names

        # second truncation: look for where current third column is blank/NaN
        # assuming more than 10 different dispensations
        end_row = report_df.ix[:, 2].tolist().index(np.nan)
        report_df = report_df.ix[:end_row-1, :].reset_index(drop=True)  # for whatever reason, includes "up-to" index

        # turn dataframe into np.array then back to dataframe to reset column names
        report_df = pd.DataFrame(np.array(report_df))

        # fix column names and drop first row
        report_df.columns = report_df.iloc[0]
        report_df = report_df.reindex(report_df.index.drop(0))

        # third truncation: only take select columns
        report_df = report_df.rename(columns={'Dispensed well':'Well', 'Fluid name':'Drug', 
                                              'Dispensed concentration':'Drug_Conc_nM'}).reset_index(drop=True)
        
        # verify we are only considering the wells that are inputted AND present in D300 report
        if len(well_list) > 0:
            output_wells = set(report_df.Well.tolist()).intersection(set(well_list))
            report_df = report_df[report_df.Well.isin(output_wells)].reset_index(drop=True)
        
        
        # get dmso concentration (assuming all dispense volumes are in nL, i.e. the default)
        sample_well = report_df[report_df['Well'] == report_df['Well'].tolist()[0]]  # take any one well
        sample_well['Dispensed volume'] = sample_well['Dispensed volume'].apply(lambda x: float(x))      # convert to float

        tot_well_vol = float(sample_well['Total well volume'].tolist()[0])
        dmso_vol = sample_well.groupby('Well')['Dispensed volume'].apply(np.sum).values[0]
        dmso_perc = np.around(dmso_vol*100.0 / tot_well_vol, 3)

        # only retain desired columns
        report_df = report_df[['Well', 'Drug', 'Drug_Conc_nM']]

        # fix drug names, i.e. take out the stock concentration
        # convert uM concentration to nM, i.e. default here is uM
        report_df['Drug'] = report_df.Drug.apply(lambda x: x.split(' ')[0])
        report_df['Drug_Conc_nM'] = report_df.Drug_Conc_nM.apply(lambda x: float(x)*1000.0)  # convert to nM
        report_df['Drug_Conc_nM'] = report_df.Drug_Conc_nM.apply(lambda x: round_sig(x, 2))  # take 2 sig figs      
    
    
    
        # check to see if SirT is present in the report file
        if 'SirT' in report_df.Drug.unique().tolist():

            sirt_report = report_df[report_df.Drug == 'SirT'].reset_index(drop=True)
            report_df = report_df[report_df.Drug != 'SirT'].reset_index(drop=True)

            # fourth truncation: only retain DMSO designation for control wells
            report_df = report_df.drop_duplicates('Well').reset_index(drop=True)

            # merge SirT concentrations
            sirt_report = sirt_report.rename(columns={'Drug_Conc_nM':'SirT_Conc_nM'})
            report_df = report_df.merge(sirt_report[['Well', 'SirT_Conc_nM']], on='Well', 
                                        how='left', copy=False)
            pctrl_conc = 310.0
            
            
            ##
            report_df = report_df[report_df.SirT_Conc_nM == 310.0].reset_index(drop=True)
            output_wells = set(report_df.Well.tolist()).intersection(set(well_list))
            report_df = report_df[report_df.Well.isin(output_wells)].reset_index(drop=True)
            ##
            

        else:    
            pctrl_conc = 800.0  # using 800nM probe concentration
            report_df['SirT_Conc_nM'] = pctrl_conc

            # fourth truncation: only retain DMSO designation for control wells
            report_df = report_df.drop_duplicates('Well').reset_index(drop=True)

        
        # round SirT conc
        report_df['SirT_Conc_nM'] = report_df.SirT_Conc_nM.apply(lambda x: round_sig(x, 2))

        # set Drug_Conc_nM to % for DMSO condition
        report_df.Drug_Conc_nM[report_df.Drug == 'DMSO'] = 0.0 #str(dmso_perc)+'%'

        # get unique conditions
        report_df['Condition'] = report_df[['Drug', 'Drug_Conc_nM', 'SirT_Conc_nM']].apply(lambda x: 
                                        str(x[0])+'_'+str(x[1])+'_'+str(x[2]), axis=1)
         
        unq_cond = report_df.Condition.unique().tolist()  # set of unique conditions

        rep_idc = []  # list of replicate indices, respective to unq_cond order
        for cond in unq_cond:
            rep_idx = tuple(report_df.index[report_df.Condition == cond].tolist())
            rep_idc.append(rep_idx)
            
        # retaining well information
        cond_well_dict = {key: group['Well'].tolist() for key, group in report_df.groupby('Condition')}

        # add these rows to report_df
        report_df['Rep_Wells'] = report_df.Condition.apply(lambda x: cond_well_dict[x])
   
        # only keep unique conditions at this point
        report_df = report_df.drop_duplicates('Condition').reset_index(drop=True).drop('Well', 1)
 
    else:
        
        # if user is inputting manually created template, it likely includes SirT in it...
        sort_cols += ['SirT_Conc_nM']
        
        # just use inputted wells as is
        output_wells = well_list
        
        report_df, rep_idc, unq_cond = plate_map_report(report_df, start_well=output_wells[0], 
                                                       end_well=output_wells[-1])
        
        ## reset DMSO drug concentration to 0.0; just take note of DMSO concentration from experiment
        report_df.Drug_Conc_nM[report_df.Drug == 'DMSO'] = 0.0
        

    ### merging well replicates
         
    # only use dataframes whose respective wells are present in D300 
    # indices of dataframes to use
    include_df_idc = find_idx([w in output_wells for w in well_list], True)
      
    # tailor df list appropriately    
    sirt_list = [df for idx, df in enumerate(sirt_list) if idx in include_df_idc]
    area_list = [df for idx, df in enumerate(area_list) if idx in include_df_idc]
    
    plate_df = pd.DataFrame()
    for c, idc in enumerate(rep_idc):
        
        cond_df = pd.DataFrame()
        for r, idx in enumerate(idc):
         
            data_df = sirt_list[idx]
            area_df = area_list[idx]
            
            # drop unnecessary simple statistics
            # we will re-compute these from the aggregated data
            data_cols = []
            for col in data_df.columns:
                if 'time' not in col.lower():
                    if col[col.index('IntDen')+6] == '_':
                        data_df = data_df.drop(col, 1)
                    else:
                        data_cols.append(col)
                        
            area_cols = []
            for col in area_df.columns:
                if 'time' not in col.lower():
                    if col[col.index('Area')+4] == '_':
                        area_df = area_df.drop(col, 1)
                    else:
                        area_cols.append(col)
                        
            # record a column of aggregated data for each plate
            # need to convert to string to trick pandas, and then convert back to list
            data_df['All_SirT_Norm_Data'] = data_df[data_cols].apply(lambda x: str(np.array(x).tolist()), axis=1)
            data_df['All_SirT_Norm_Data'] = data_df.All_SirT_Norm_Data.apply(lambda x: ast.literal_eval(x))
            data_df['All_Area_Data'] = area_df[area_cols].apply(lambda x: str(np.array(x).tolist()), axis=1)
            data_df['All_Area_Data'] = data_df.All_Area_Data.apply(lambda x: ast.literal_eval(x))
            
            # convert SirT/Area back to SirT alone for each "micro-field"
            # then sum all SirT intensities and normalize by total cell area present in well
            data_df['All_SirT_Data'] = data_df[['All_SirT_Norm_Data', 'All_Area_Data']].apply(lambda x: 
                            element_prod(x[0], x[1]), axis=1)
            
            data_df['SirT_Well_Val'] = data_df[['All_SirT_Data', 'All_Area_Data']].apply(lambda x: 
                            np.sum(x[0]) / np.sum(x[1]), axis=1)
            
            # record the well values
            cond_df['All_Data_'+str(r+1)] = data_df.SirT_Well_Val
              
        # aggregate all of the aggregate data
        agg_cols = [col for col in cond_df.columns if 'All_Data' in col]
        
        # hack to bring values into an array
        cond_df['All_Data'] = cond_df[agg_cols].apply(lambda x: str(list(x)), axis=1)
        cond_df['All_Data'] = cond_df.All_Data.apply(lambda x: ast.literal_eval(x))
        
        # remove unnecessary columns
        for col in agg_cols:
            cond_df = cond_df.drop(col, 1)
        
        cond_df['Mean'] = cond_df.All_Data.apply(lambda x: np.mean(x))
       
        # if it's a time-course
        timed = False  # preset
        if 'Time_Min' in sirt_list[0].columns:
            timed = True
            cond_df['Time_Min'] = sirt_list[0].Time_Min
                
        cond_df['Condition'] = unq_cond[c]
        
        # record condition
        plate_df = pd.concat([plate_df, cond_df]).reset_index(drop=True)   

    # merge all information together and drop unnecessary column
    plate_df = plate_df.merge(report_df, how='left', on='Condition', copy=False)
    
    # sort plate_df and output
    if timed:
        sort_cols += ['Time_Min']
    
    plate_df = plate_df.sort(sort_cols).reset_index(drop=True)  # sort df
    
    # labeling positive controls
    plate_df['Pos_Ctrl_Well'] = plate_df[['Drug', 'Drug_Conc_nM', 'SirT_Conc_nM']].apply(lambda x: 
                        (True if x[1] == x[2] else False) if x[0] == 'EpoB' else False, axis=1)

    # normalize plate data to control at 10 hrs (arbitrary, but must be consistent across bio-repeats)
    nctrl_mean = plate_df[(plate_df.Drug == 'DMSO') & (plate_df.SirT_Conc_nM == pctrl_conc)]
    nctrl_mean = nctrl_mean.Mean[nctrl_mean.Time_Min == 600.0].values[0]
    
    plate_df['Norm_Data'] = plate_df.All_Data.apply(lambda x: (np.array(x)/nctrl_mean).tolist())
    
    # remove unnecessary columns and output
    return plate_df.drop('Mean', 1).drop('All_Data', 1)
    
    
##########################################################################################
    
    
# function to turn all negative values in a list into zero
def neg2zero(list1):
    
    array1 = np.array(list1)
    array1[array1 < 0] = 0
    return array1.tolist()

# function to take aggregated SirT dataframes, further aggregate, and subtract positive control
# data should be normalized to the control mean
def agg_subtract_ctrl(data_list, pctrl_conc=800.0):
    
    agg_data = data_list[0].copy()
    agg_data['condition_plus_time'] = agg_data[['Condition', 'Time_Min']].apply(lambda x: x[0]+str(x[1]), axis=1)
    
    # need to use time info for merging too
    for df in data_list[1:]:
        df['condition_plus_time'] = df[['Condition', 'Time_Min']].apply(lambda x: x[0]+str(x[1]), axis=1)
        
        agg_data = agg_data.merge(df[['condition_plus_time', 'Norm_Data', 'Rep_Wells']], 
                                  on='condition_plus_time', how='left', copy=False)
    
    # delete unnecessary column
    agg_data = agg_data.drop('condition_plus_time', 1)
        
        
    # get data col names and replicate wells
    data_cols = []
    well_cols = []
    for col in agg_data.columns:
        if 'Norm_Data' in col:
            data_cols.append(col)
        if 'Rep_Wells' in col:
            well_cols.append(col)
            
    # aggregate data columns
    agg_data['Agg_Norm_Data'] = agg_data[data_cols].apply(lambda x: sum(list(x), []), axis=1)
    agg_data['Agg_Rep_Wells'] = agg_data[well_cols].apply(lambda x: np.sum(x), axis=1)
    
    # delete unnecessary columns
    for col in data_cols+well_cols:
        agg_data = agg_data.drop(col, 1)
    
    # get means and sems
    agg_data['Agg_Mean'] = agg_data.Agg_Norm_Data.apply(lambda x: np.mean(x))
    agg_data['Agg_SEM'] = agg_data.Agg_Norm_Data.apply(lambda x: np.std(x)/np.sqrt(len(x)))
    
    # expand control wells according to number of respective drug wells
    pos_ctrl = agg_data[agg_data.Pos_Ctrl_Well == True]
    agg_data = agg_data[agg_data.Pos_Ctrl_Well == False]

    unq_sirt_conc = pos_ctrl.SirT_Conc_nM.unique().tolist()  # unique probe concentrations used

    exp_pos_ctrl = pd.DataFrame()
    for conc in unq_sirt_conc:

        len_conc = len(agg_data[agg_data.SirT_Conc_nM == conc])
        len_conc_ctrl = len(pos_ctrl[pos_ctrl.SirT_Conc_nM == conc])

        # number of times we need to replicate this set of control wells
        rep_int = int(len_conc*1.0/len_conc_ctrl)

        # expanding controls
        exp_pos_ctrl = pd.concat([exp_pos_ctrl]+[pos_ctrl[pos_ctrl.SirT_Conc_nM == conc]]*rep_int).reset_index(drop=True)   

        
    agg_data = agg_data[['Agg_Rep_Wells', 'Drug', 'Drug_Conc_nM', 'SirT_Conc_nM', 'Time_Min', 'Agg_Norm_Data', 
                         'Agg_Mean', 
                         'Agg_SEM']].sort(['Drug', 'Drug_Conc_nM', 'SirT_Conc_nM', 'Time_Min']).reset_index(drop=True)
    exp_pos_ctrl = exp_pos_ctrl[['Agg_Rep_Wells', 'Drug', 'Drug_Conc_nM', 'SirT_Conc_nM', 'Time_Min', 'Agg_Norm_Data', 
                                 'Agg_Mean', 
                                 'Agg_SEM']]  # do not sort or else replicate data will be messed up
        
    # performing the subtraction on mean/sem cols and replicate cols        
    agg_data['Agg_Mean'] = np.array(agg_data['Agg_Mean']) - np.array(exp_pos_ctrl['Agg_Mean'])
    agg_data['Agg_SEM'] = np.sqrt(np.array(agg_data['Agg_SEM'])**2 + np.array(exp_pos_ctrl['Agg_SEM'])**2)

    # make temporary column to subtract positive control from aggregate data
    agg_data['temp_pos_ctrl'] = exp_pos_ctrl['Agg_SEM']
    agg_data['Agg_Norm_Data'] = agg_data[['Agg_Norm_Data', 'temp_pos_ctrl']].apply(lambda x: 
                                (np.array(x[0])-x[1]).tolist(), axis=1)
    agg_data = agg_data.drop('temp_pos_ctrl', 1)

    # re-normalize everything to negative control (i.e. pctrl_conc) at 10 hrs
    nctrl_mean = agg_data[(agg_data.Drug =='DMSO') & (agg_data.SirT_Conc_nM == pctrl_conc)]
    nctrl_mean = nctrl_mean.Agg_Mean[nctrl_mean.Time_Min == 600.0].values.tolist()[0]
       
    agg_data['Agg_Norm_Data'] = agg_data.Agg_Norm_Data.apply(lambda x: np.array(x)/nctrl_mean)
    agg_data['MT_IntDen_SEM_Norm'] = agg_data.Agg_SEM.apply(lambda x: x/nctrl_mean)
    agg_data['MT_IntDen_Mean_Norm'] = agg_data.Agg_Mean.apply(lambda x: x/nctrl_mean)
    
    # set all negative values to zero
    agg_data['Agg_Norm_Data'] = agg_data.Agg_Norm_Data.apply(lambda x: neg2zero(x))
    agg_data['Agg_Mean'] = agg_data.Agg_Mean.apply(lambda x: 0.0 if x < 0.0 else x)
    
    # sort by condition and then drop unnecessary columns
    agg_data = agg_data.sort(['Drug', 'Drug_Conc_nM', 'Time_Min']).reset_index(drop=True)
    
    # add column with drug concentration repeated same number of times as there are aggregated data points
    agg_data['Agg_Drug_Conc_nM'] = agg_data[['Drug_Conc_nM', 'Agg_Norm_Data']].apply(lambda x: 
                                            (np.ones(len(x[1]))*x[0]).tolist(), axis=1)
    
    return agg_data
    
    
##########################################################################################

### code to annotate well information and merge with image plate data values

# function that makes plate maps and outputs replicate well indices
# used in order_well_info()
# NOTE: files are always quantified in well order
def plate_map_report(map_path, print_template=0, start_well='B02', end_well='O23'):
    
    # read in the plate map
    # NOTE: this converts the top cells into column names, which we don't want
    plate_map = pd.read_csv(map_path)
    
    # get number of columns
    col_num = np.shape(plate_map)[1]
    
    # read in plate map again, this time without disturbing top cells
    plate_map = np.array(pd.read_csv(map_path, names=np.arange(col_num)))
    final_plate_map = sum(np.array(plate_map).tolist(), [])

    # create 384-well plate
    rows = list('abcdefghijklmnop'.upper())
    cols = [str(i+1) for i in range(24)]
    cols = ['0'+i if len(i) == 1 else i for i in cols]
    well_list = list(product(rows, cols))
    well_array = np.array([w[0]+w[1] for w in well_list])
    well_array = np.reshape(well_array, (16, 24))
    
    # only take desired wells...
    start_row = start_well[0]
    end_row = end_well[0]
    start_col = start_well[1:]
    end_col = end_well[1:]
    
    # ...respective indices
    start_row_idx = rows.index(start_row)
    end_row_idx = rows.index(end_row)
    start_col_idx = cols.index(start_col)
    end_col_idx = cols.index(end_col)
    
    # truncate well array accordingly
    final_well_array = sum(well_array[start_row_idx:end_row_idx+1, start_col_idx:end_col_idx+1].tolist(), [])
    
    
    # at this point, final_well_array should have same dimensions as final_plate_map
    # unless the template is just a plate full of the same condition
    if len(set(final_plate_map)) != 1:
        assert(len(final_plate_map) == len(final_well_array))
    # or else shorten the length of the plate map
    else:
        
        final_plate_map = final_plate_map[:len(final_well_array)]
    
    if print_template:
        print np.array(plate_map)
        
    # finding replicate wells
    unq_cond = pd.Series(final_plate_map).unique()

    # get list indices of replicates; tuple the replicates
    rep_idc = []
    rep_wells = []
    for c in unq_cond:
        rep_idc.append(tuple(find_idx(final_plate_map, c)))
        
    # get tuples of replicate wells corresponding to rep_idc
    rep_wells = []
    for idc in rep_idc:
        wells = []
        for idx in idc:
            wells.append(final_well_array[idx])
        rep_wells.append(wells)
            
    # initiating report file
    report_df = pd.DataFrame(pd.Series(rep_wells), columns=['Rep_Wells'])
    report_df['Condition'] = unq_cond
    
    # first drug listed is SirT; second is the other drug of interest or DMSO
    report_df['Drug'] = report_df.Condition.apply(lambda x: 
                                                  x.split('_')[1].split(' ')[1])
    report_df['Drug_Conc_nM'] = report_df.Condition.apply(lambda x: 
                                                  x.split('_')[1].split(' ')[0])
    report_df['Drug_Conc_nM'] = report_df.Drug_Conc_nM.apply(lambda x: 
                                                  float(x) if '%' not in x else x)
    report_df['SirT_Conc_nM'] = report_df.Condition.apply(lambda x: 
                                                  float(x.split('_')[0].split(' ')[0]))
    
    # return indices and conditions
    return report_df, rep_idc, unq_cond
    

# to round up to a desired number of significant figures
def round_sig(x, sig=2):
    
    try: # in case user accidentally inputs np.nan
        # in user accidentally inputs a string, just return the string back
        if isinstance(x, str):
            return x
        else:
            if x == 0:
                return 0
            else:
                abs_round = float(np.around(np.abs(x), sig-int(np.floor(np.log10(np.abs(x))))-1))
                if x < 0:
                    return -abs_round
                else:
                    return abs_round
    except:
        return np.nan
        
        
# takes report file from D300 dispenser, organizes the well info, and merges replicates
# NOTE: 'report_df' can be a string leading to a manually created report file
# 'brep_list': list of quantified well dataframes
# 'well_list': list of quantified wells; if inputted, will verify wells are in D300 report
# 'method': can be 'mt', 'area', 'comet', 'mitosis', or 'ctg' for cell-titer-glo
# if using manually created report template, use 'start_well'/'end_well' instead of 'rows_used'/'cols_used'
# 'fc': if fold-change desired
def order_well_info(report_df, brep_list, method='sirt', rep_att='Median', well_list=[],
                    fc=False):
    
    # processing inputted well list
    if len(well_list) > 0:
        well_list = [w.split(' - ')[0] + w.split(' - ')[1] for w in well_list]
    
    method = method.lower()
    sort_cols = ['Drug', 'Drug_Conc_nM']
    
    # if a d300 report file is inputted
    if not isinstance(report_df, str):
        
        if method == 'sirt':
            sort_cols += ['SirT_Conc_nM']


        ### organizing well info

        # first truncation: look for "Tabular detail"
        start_row = report_df.ix[:, 0].tolist().index('Tabular detail')
        report_df = report_df.ix[start_row+1:, :].reset_index(drop=True)

        cols = report_df.columns  # getting column names

        # second truncation: look for where current third column is blank/NaN
        # assuming more than 10 different dispensations
        end_row = report_df.ix[:, 2].tolist().index(np.nan)
        report_df = report_df.ix[:end_row-1, :].reset_index(drop=True)  # for whatever reason, includes "up-to" index

        # turn dataframe into np.array then back to dataframe to reset column names
        report_df = pd.DataFrame(np.array(report_df))

        # fix column names and drop first row
        report_df.columns = report_df.iloc[0]
        report_df = report_df.reindex(report_df.index.drop(0))

        # third truncation: only take select columns
        report_df = report_df.rename(columns={'Dispensed well':'Well', 'Fluid name':'Drug', 
                                              'Dispensed concentration':'Drug_Conc_nM'}).reset_index(drop=True)
        
        # verify we are only considering the wells that are inputted AND present in D300 report
        if len(well_list) > 0:
            output_wells = set(report_df.Well.tolist()).intersection(set(well_list))
            report_df = report_df[report_df.Well.isin(output_wells)].reset_index(drop=True)
        
        
        # get dmso concentration (assuming all dispense volumes are in nL, i.e. the default)
        sample_well = report_df[report_df['Well'] == report_df['Well'].tolist()[0]]  # take any one well
        sample_well['Dispensed volume'] = sample_well['Dispensed volume'].apply(lambda x: float(x))      # convert to float

        tot_well_vol = float(sample_well['Total well volume'].tolist()[0])
        dmso_vol = sample_well.groupby('Well')['Dispensed volume'].apply(np.sum).values[0]
        dmso_perc = np.around(dmso_vol*100.0 / tot_well_vol, 3)

        # only retain desired columns
        report_df = report_df[['Well', 'Drug', 'Drug_Conc_nM']]

        # fix drug names, i.e. take out the stock concentration
        # convert uM concentration to nM, i.e. default here is uM
        report_df['Drug'] = report_df.Drug.apply(lambda x: x.split(' ')[0])
        report_df['Drug_Conc_nM'] = report_df.Drug_Conc_nM.apply(lambda x: float(x)*1000.0)  # convert to nM
        report_df['Drug_Conc_nM'] = report_df.Drug_Conc_nM.apply(lambda x: round_sig(x, 2))  # take 2 sig figs      
        
        if method == 'sirt':

            # check to see if SirT is present in the report file
            if 'SirT' in report_df.Drug.unique().tolist():

                sirt_report = report_df[report_df.Drug == 'SirT'].reset_index(drop=True)
                report_df = report_df[report_df.Drug != 'SirT'].reset_index(drop=True)

                # fourth truncation: only retain DMSO designation for control wells
                report_df = report_df.drop_duplicates('Well').reset_index(drop=True)

                # merge SirT concentrations
                sirt_report = sirt_report.rename(columns={'Drug_Conc_nM':'SirT_Conc_nM'})
                report_df = report_df.merge(sirt_report[['Well', 'SirT_Conc_nM']], on='Well', how='left', copy=False)

            else:    
                pctrl_conc = 800.0  # using 800nM probe concentration
                report_df['SirT_Conc_nM'] = pctrl_conc

                # fourth truncation: only retain DMSO designation for control wells
                report_df = report_df.drop_duplicates('Well').reset_index(drop=True)
                
            # round SirT conc
            report_df['SirT_Conc_nM'] = report_df.SirT_Conc_nM.apply(lambda x: round_sig(x, 2))

        else:
            report_df = report_df.drop_duplicates('Well').reset_index(drop=True)

        # set Drug_Conc_nM to % for DMSO condition
        report_df.Drug_Conc_nM[report_df.Drug == 'DMSO'] = str(dmso_perc)+'%'

        # get unique conditions
        if method == 'sirt':
            report_df['Condition'] = report_df[['Drug', 'Drug_Conc_nM', 'SirT_Conc_nM']].apply(lambda x: 
                                            str(x[0])+'_'+str(x[1])+'_'+str(x[2]), axis=1)
        else:
            report_df['Condition'] = report_df[['Drug', 'Drug_Conc_nM']].apply(lambda x: 
                                            str(x[0])+'_'+str(x[1]), axis=1)
        
        unq_cond = report_df.Condition.unique().tolist()  # set of unique conditions

        rep_idc = []  # list of replicate indices, respective to unq_cond order
        for cond in unq_cond:
            rep_idx = tuple(report_df.index[report_df.Condition == cond].tolist())
            rep_idc.append(rep_idx)
            
        # retaining well information
        cond_well_dict = {key: group['Well'].tolist() for key, group in report_df.groupby('Condition')}

        # add these rows to report_df
        report_df['Rep_Wells'] = report_df.Condition.apply(lambda x: cond_well_dict[x])
   
        # only keep unique conditions at this point
        report_df = report_df.drop_duplicates('Condition').reset_index(drop=True).drop('Well', 1)
 
    else:
        
        # if user is inputting manually created template, it likely includes SirT in it...
        sort_cols += ['SirT_Conc_nM']
        
        # just use inputted wells as is
        output_wells = well_list
        
        report_df, rep_idc, unq_cond = plate_map_report(report_df, start_well=output_wells[0], 
                                                       end_well=output_wells[-1])
        
        
   
    ### merging well replicates

    # setting column names
    if (method == 'mt') | (method == 'sirt'): 
        rep_col = 'MT_IntDen_'+rep_att+'_'
        mean_col = 'MT_IntDen_Mean'
        sem_col = 'MT_IntDen_SEM'
    elif method == 'area':
        rep_col = 'Cell_Area_'+rep_att+'_'
        mean_col = 'Cell_Area_Mean'
        sem_col = 'Cell_Area_SEM'
    elif method == 'mitosis':
        rep_col = 'Mitotic_Index_'
        mean_col = 'Mitotic_Index_Mean'
        sem_col = 'Mitotic_Index_SEM'
    elif method == 'nuclear size':
        rep_col = 'Nuclear_Size_'
        mean_col = 'Nuclear_Size_Mean'
        sem_col = 'Nuclear_Size_SEM'
    elif method == 'nuclear count':
        rep_col = 'Nuclear_Count_'
        mean_col = 'Nuclear_Count_Mean'
        sem_col = 'Nuclear_Count_SEM'
    elif method == 'ctg':
        rep_col = 'Life_Index_'
        mean_col = 'Life_Index_Mean'
        sem_col = 'Life_Index_SEM'
    elif method == 'feret':
        rep_col = 'Feret_'+rep_att+'_'
        mean_col = 'Feret_Mean'
        sem_col = 'Feret_SEM'
    else:
        rep_col = 'Comet_'+rep_att+'_'
        mean_col = 'Comet_Mean'
        sem_col = 'Comet_SEM'
            
         
    # only use dataframes whose respective wells are present in D300 
    # indices of dataframes to use
    include_df_idc = find_idx([w in output_wells for w in well_list], True)
      
    # tailor df list appropriately    
    brep_list = [df for idx, df in enumerate(brep_list) if idx in include_df_idc]
    
    plate_df = pd.DataFrame()
    for c, idc in enumerate(rep_idc):
        
        cond_df = pd.DataFrame()
        for r, idx in enumerate(idc):
         
            med_col = rep_col+str(r+1)
            if isinstance(brep_list[idx], pd.DataFrame):
                cond_df[med_col] = brep_list[idx][rep_col[:-1]]
            else:
                cond_df[med_col] = [brep_list[idx]]         
                
        # calculate means and sems
        cond_mean = cond_df.ix[:, :len(idc)].apply(lambda x: 
                    np.mean(np.array(x)), axis=1)
        cond_sem = cond_df.ix[:, :len(idc)].apply(lambda x: 
                    np.std(np.array(x))/np.sqrt(len(idc)), axis=1)
        cond_df[mean_col] = cond_mean
        cond_df[sem_col] = cond_sem
        
        # if fold-change desired
        if fc:
            cond_df[mean_col] = cond_df[mean_col].values / cond_df[mean_col].values[0]
            cond_df[sem_col] = cond_df[sem_col].values / cond_df[mean_col].values[0]
        
        # if it's a time-course
        timed = False  # preset
        if isinstance(brep_list[0], pd.DataFrame):
            if 'Time_Min' in brep_list[0].columns:
                timed = True
                cond_df['Time_Min'] = brep_list[0].Time_Min
                
        cond_df['Condition'] = unq_cond[c]
        
        # record condition
        plate_df = pd.concat([plate_df, cond_df]).reset_index(drop=True)
          
    # merge all information together and drop unnecessary column
    plate_df = plate_df.merge(report_df, how='left', on='Condition', copy=False)
    
    # sort plate_df and output
    if timed:
        sort_cols += ['Time_Min']
    
    plate_df = plate_df.sort(sort_cols).reset_index(drop=True)
        
    # labeling positive controls
    if method == 'sirt':
        plate_df['Pos_Ctrl_Well'] = plate_df[['Drug', 'Drug_Conc_nM', 'SirT_Conc_nM']].apply(lambda x: 
                            (True if x[1] == x[2] else False) if x[0] == 'EpoB' else False, axis=1)


        
    ### getting z-prime score to assess set-up
    
    # only for SirT experiments
    if method == 'sirt':
        
        # setting control concentration
        ctrl_conc = 800.0
        
        # verify that control concentration is in the report
        # or else use a log or half-log concentration
        if 800.0 not in plate_df.SirT_Conc_nM.unique().tolist():
            ctrl_conc = 1000.0
        
        # checking if there's a time-course
        if len(plate_df.Time_Min.unique()) > 1:
            equil_time = 600.0
        else:
            equil_time = 0.0
        
        z_prime_df = plate_df.ix[plate_df.SirT_Conc_nM.apply(lambda x: 
                            round_sig(x, 1) == ctrl_conc), :].reset_index(drop=True)
        z_prime_df = z_prime_df[z_prime_df.Time_Min == equil_time].reset_index(drop=True)
        
        # positive control statistics
        pos_ctrl_mean = z_prime_df.ix[z_prime_df.Pos_Ctrl_Well == True, mean_col].values[0]
        pos_ctrl_sem = z_prime_df.ix[z_prime_df.Pos_Ctrl_Well == True, sem_col].values[0]
        pos_ctrl_std = pos_ctrl_sem * np.sqrt(z_prime_df[z_prime_df.Pos_Ctrl_Well == 
                                    True].Rep_Wells.apply(lambda x: len(x)).tolist()[0]*1.0)
        
        # negative control statistics
        neg_ctrl_mean = z_prime_df.ix[z_prime_df.Drug == 'DMSO', mean_col].values[0]
        neg_ctrl_sem = z_prime_df.ix[z_prime_df.Drug == 'DMSO', sem_col].values[0]
        neg_ctrl_std = neg_ctrl_sem * np.sqrt(z_prime_df[z_prime_df.Drug == 
                                    'DMSO'].Rep_Wells.apply(lambda x: len(x)).tolist()[0]*1.0)
        
        
        # z-prime
        z_prime = 1.0 - 3.0*(pos_ctrl_std + neg_ctrl_std) / np.abs(pos_ctrl_mean - neg_ctrl_mean)
        print '\nPlate Z-prime score: %f\n' % z_prime
        
        
    
    ### subtracting positive control

    # only for SirT experiments
    if method == 'sirt':
    
        # expand control wells according to number of respective drug wells
        pos_ctrl = plate_df[plate_df.Pos_Ctrl_Well == True]
        plate_df = plate_df[plate_df.Pos_Ctrl_Well == False]
        
        unq_sirt_conc = pos_ctrl.SirT_Conc_nM.unique().tolist()  # unique probe concentrations used
        
        exp_pos_ctrl = pd.DataFrame()
        for conc in unq_sirt_conc:
            
            len_conc = len(plate_df[plate_df.SirT_Conc_nM == conc])
            len_conc_ctrl = len(pos_ctrl[pos_ctrl.SirT_Conc_nM == conc])

            # number of times we need to replicate this set of control wells
            rep_int = int(len_conc*1.0/len_conc_ctrl)

            # expanding controls
            exp_pos_ctrl = pd.concat([exp_pos_ctrl]+[pos_ctrl[pos_ctrl.SirT_Conc_nM == conc]]*rep_int).reset_index(drop=True)   
            
        # getting the replicate columns
        rep_col_list = [c for c in plate_df.columns if rep_col in c]

        # removing field replicate value columns
        plate_df = plate_df[['Rep_Wells']+sort_cols+rep_col_list+[mean_col, 
                             sem_col]].sort(sort_cols).reset_index(drop=True)
        exp_pos_ctrl = exp_pos_ctrl[['Rep_Wells']+sort_cols+[mean_col, 
                             sem_col]]  # do not sort or else replicate data will be messed up
        
        # performing the subtraction on mean/sem cols and replicate cols        
        plate_df[mean_col] = np.array(plate_df[mean_col]) - np.array(exp_pos_ctrl[mean_col])
        plate_df[sem_col] = np.sqrt(np.array(plate_df[sem_col])**2 + np.array(exp_pos_ctrl[sem_col])**2)

        for col in rep_col_list:
            plate_df[col] = plate_df[col] - exp_pos_ctrl[mean_col]
            
    # adding 'Labels' column for plotting purposes
    plate_df['Labels'] = plate_df[['Drug', 'Drug_Conc_nM']].apply(lambda x: str(x[1])+' '+x[0], axis=1)
       
    # sort before output
    plate_df = plate_df.sort(sort_cols).reset_index(drop=True)

    return plate_df
    
    

    
        
        
##########################################################################################


### SiR-tubulin modeling functions

# only input 'mt_o_prior' if you know initial and max polymer amount (i.e. 'mt_high_prior')
# 'sim_num': number of values to test for each variable
def get_kdp_app(p_out, pmt, mt_o_prior=[], mt_high_prior=4500.0, kdp_app_prior=[], sim_num=10):

    ## setting kdd_app bounds
    
    # we will assume that the apparent Kd is within the concentration bounds of our experiment
    # with a big enough range of concentrations, this should be a safe assumption
    start_conc = p_out[0]
    end_conc = p_out[-1]
    
    
    ## variable pre-sets
    
    # if the initial polymer amount is unknown
    if isinstance(mt_o_prior, list):
        mt_high_range = np.linspace(mt_high_prior, 2.0*mt_high_prior, sim_num) 
        mt_o_range = np.linspace(mt_high_prior/2.0, mt_high_prior, sim_num)
        
    # or else use known values
    else:
        mt_high_range = [mt_high_prior] 
        mt_o_range = [mt_o_prior]       
        
    # k constant is proportional to 1/kdd_app (within 10-fold as judged by drugs and SirT); test 100-fold
    # if no prior inputted, test a wide range
    if isinstance(kdp_app_prior, list):
        k_range = 10.0**np.linspace(np.log10(0.0001), np.log10(10), sim_num)
        
    # or else test a narrower range
    else:
        k_range = 10.0**np.linspace(np.log10(0.01/kdp_app_prior), np.log10(100.0/kdp_app_prior), sim_num)
    
    
    ## simulation
    
    # variables to optimize in addition to 'kdd_app': 'mt_high', 'mt_o', 'k'
    
    
    # starting all values at middle of respective range
    start_var_idx = np.floor(sim_num/2.0)
    
    # if mt_o is inputted, don't mess with mt_o or mt_high
    if len(mt_high_range) == 1:
        start_var_combo = [0, 0, start_var_idx]                 # i.e. 3 variables
    else:
        start_var_combo = [start_var_idx]*3         
        
    max_range = int(max(sim_num-start_var_idx, start_var_idx))  # max difference from start index

    SSE_min = np.inf
    for i in range(max_range):
        
        range_i = max_range - i                         # descending (500, 499,..., 2, 1)
                
        # make array with values to subtract or add
        delta_options = np.array([-range_i, 0, range_i])
        delta_array = np.array(list(product(delta_options, repeat=3)))    
        
        
        # if mt_o is inputted, disregard any deltas for mt_o and mt_high
        if len(mt_high_range) == 1:
            delta_array = delta_array[np.all(delta_array[:, :2] == 0, 1)]
        
        if i > 0:
            # setting starting variable combo to the previous optimal combo
            start_var_combo = opt_var_combo
        
        # making repeat-matrix of starting combination
        start_var_array = np.tile(start_var_combo, [len(delta_array), 1])
        
        # add the start array and the delta array to get combinations
        test_var_combos_i = start_var_array+delta_array
        
        # check if any values in test_var_combos exceed max or min threshold
        # discard these combinations
        test_var_combos_i = test_var_combos_i[~np.any((test_var_combos_i >= sim_num) | 
                                            (test_var_combos_i < 0), 1)]
  
        # iterate through combinations
        for combo in test_var_combos_i:
            
            ### setting variable test values
            mt_high_idx = int(combo[0])
            mt_o_idx = int(combo[1])
            k_idx = int(combo[2])
        
            mt_high = mt_high_range[mt_high_idx]  # setting mt_high
            mt_o = mt_o_range[mt_o_idx]           # setting mt_o
            k = k_range[k_idx]                    # setting k

            
            # slow-rise exponential model equation
            mt_tot = mt_o + (mt_high - mt_o)*(1.0 - np.exp(-k*p_out))

            # getting N possible kdd_app for the drug, all evenly spaced
            sim_kdp_app = 10.0**np.linspace(np.log10(start_conc), np.log10(end_conc), 100)

            # performing computation in dataframe format
            drug_sim_df = pd.DataFrame(sim_kdp_app, columns=['Sim_Kdp_app'])

            # setting array of external drug concentrations (i.e. every well has an array within it)
            drug_sim_df['P_out'] = str(p_out.tolist())

            # need to use numpy array that way floats and arrays can be combined in computation steps
            drug_sim_df['P_out'] = drug_sim_df.P_out.apply(lambda x: np.array(ast.literal_eval(x)))

            # setting known information
            drug_sim_df['PMT'] = str(pmt.tolist())  # observed probe signal
            drug_sim_df['PMT'] = drug_sim_df.PMT.apply(lambda x: np.array(ast.literal_eval(x)))

            # performing simulation
            drug_sim_df['Sim_PMT'] = drug_sim_df[['P_out', 'Sim_Kdp_app']].apply(lambda x: 
                                                str(list(mt_tot*x[0]/(x[1] + x[0]))), axis=1)
        
            drug_sim_df['Sim_PMT'] = drug_sim_df.Sim_PMT.apply(lambda x: np.array(ast.literal_eval(x)))

            # calculating error between simulation and actual observed signal
            drug_sim_df['SSE'] = drug_sim_df[['PMT', 'Sim_PMT']].apply(lambda x: np.sum((x[0] - x[1])**2), axis=1)

            
            # finding optimal kdd_val with the inputted test parameters
            drug_sim_df = drug_sim_df.sort('SSE').reset_index(drop=True)
            
            # getting current minimum SSE
            SSE = drug_sim_df.SSE.values[0]

            if SSE < SSE_min:
                SSE_min = SSE

                # resetting start combo
                opt_var_combo = combo
                
                # the desired output variables
                kdp_app = drug_sim_df.Sim_Kdp_app.values[0]
                k_final = k;            
                mt_o_final = mt_o
                mt_high_final = mt_high
                orig_sim_pmt = drug_sim_df.Sim_PMT.values[0]

                
    ## now applying simulated parameters

    # using kdd_app to simulate probe signal for "num_sim" concentrations 
    # this will enable the user to build a smooth, regressed curve later
    sim_p_out = 10.0**np.linspace(np.log10(start_conc), np.log10(end_conc), 100)
    final_mt_tot = mt_o_final + (mt_high_final - mt_o_final)*(1.0 - np.exp(-k_final*sim_p_out))
    
    pmt_sim_df = pd.DataFrame(sim_p_out, columns=['P_out'])
    pmt_sim_df['MT_tot'] = final_mt_tot
    pmt_sim_df['Sim_PMT'] = pmt_sim_df[['MT_tot', 'P_out']].apply(lambda x: 
                                                    x[0]*x[1]/(x[1] + kdp_app), axis=1)   
                                                                  
    sim_pmt = pmt_sim_df.Sim_PMT.values
    
    print 'Value of k constant: %s' % str(k_final)
    
    if len(set(final_mt_tot)) != 1:
        print 'Estimated MT polymer fold-change: %s' % str(final_mt_tot[-1]/final_mt_tot[0])
        
        
    # getting R squared to assess model fit
    SS_res = sum([(pmt[i] - orig_sim_pmt[i])**2 for i in range(len(pmt))])
    SS_tot = sum([(pmt[i] - np.mean(pmt))**2 for i in range(len(pmt))])
    R2 = 1.0 - SS_res*1.0/SS_tot
    print 'R squared of model fit: %f' % R2

    return kdp_app, sim_p_out, sim_pmt, final_mt_tot

    
# 'sim_num': number of values to test for each variable
def get_kdd_app(drug_out, pmt, ctrl_pmt, all_drug=[], all_pmt=[], p_out=800.0, kdp_app=1000.0, kdd_app_prior=[], mt_fc=2.0, 
                sim_num=100, conf_95=False, annotate=False):

    
    if (len(all_drug) == 0) & (len(all_pmt) == 0):
        all_drug = drug_out
        all_pmt = pmt
    

    ## setting kdd_app bounds
    
    # we will assume that the apparent Kd is within the concentration bounds of our experiment
    # with a big enough range of concentrations, this should be a safe assumption
    start_conc = np.min(drug_out)
    end_conc = np.max(drug_out)
    
    # calculating initial MT polymer value
    mt_o = ctrl_pmt*(1.0 + kdp_app/p_out)
    mt_max = mt_o * mt_fc
 

    ## variable pre-sets
         
    # k constant is proportional to 1/kdd_app (within 10-fold as judged by taxane-site drugs); test 1000-fold
    if not isinstance(kdd_app_prior, list):
        k_range = 10.0**np.linspace(np.log10(0.001/kdd_app_prior), np.log10(1000.0/kdd_app_prior), sim_num)
    else:
        k_range = 10.0**np.linspace(np.log10(0.001), np.log10(1000.0), sim_num)
    
    ## simulation
    
    # optimizing k constant
    RSS_min = np.inf
    # iterate through combinations
    for k in k_range:

        # slow-rise exponential model equation
        mt_tot = mt_max*(1.0 - np.exp(-k*drug_out)) + mt_o*np.exp(-k*drug_out)

        # getting N possible kdd_app for the drug, all evenly spaced
        sim_kdd_app = 10.0**np.linspace(np.log10(start_conc), np.log10(end_conc), 100)

        # performing computation in dataframe format
        drug_sim_df = pd.DataFrame(sim_kdd_app, columns=['Sim_Kdd_app'])

        # setting array of external drug concentrations (i.e. every well has an array within it)
        drug_sim_df['D_out'] = str(drug_out.tolist())

        # need to use numpy array that way floats and arrays can be combined in computation steps
        drug_sim_df['D_out'] = drug_sim_df.D_out.apply(lambda x: np.array(ast.literal_eval(x)))

        # setting known information
        drug_sim_df['PMT'] = str(pmt.tolist())  # observed probe signal
        drug_sim_df['PMT'] = drug_sim_df.PMT.apply(lambda x: np.array(ast.literal_eval(x)))

        # performing simulation
        drug_sim_df['Sim_PMT'] = drug_sim_df[['D_out', 'Sim_Kdd_app']].apply(lambda x: 
                                            str(list(mt_tot/(kdp_app/p_out*(x[0]/x[1]+1)+1))), axis=1)
        drug_sim_df['Sim_PMT'] = drug_sim_df.Sim_PMT.apply(lambda x: np.array(ast.literal_eval(x)))

        # calculating error between simulation and actual observed signal
        drug_sim_df['RSS'] = drug_sim_df[['PMT', 'Sim_PMT']].apply(lambda x: np.sum((x[0] - x[1])**2), axis=1)

        # calculating negative log likelihood
        drug_sim_df['Neg_Log_Like'] = drug_sim_df.RSS.apply(lambda x: len(drug_out)/2.0*np.log(x))


        # finding optimal kdd_val with the inputted test parameters
        drug_sim_df = drug_sim_df.sort('RSS').reset_index(drop=True)

        # getting current minimum SSE
        RSS = drug_sim_df.RSS.values[0]

        if RSS < RSS_min:
            RSS_min = RSS

            # the desired output variables
            kdd_app = drug_sim_df.Sim_Kdd_app.values[0]
            k_final = k;            
            orig_sim_pmt = drug_sim_df.Sim_PMT.values[0]
        
        
    # theta = (kdd_app, k, mt_max, kdp_app)
    pmt_func = lambda theta: (mt_o*np.exp(-theta[1]*all_drug) + theta[2]*(1.0-np.exp(-theta[1]*all_drug))) / (theta[3]*all_drug/theta[0]/p_out + theta[3]/p_out + 1.0)

    # defining the negative log likelihood function
    neg_log_like = lambda theta: len(all_drug)/2.0*np.log(np.sum((all_pmt - pmt_func(theta))**2.0))
    
    # getting the matrix of second derivatives
    H = nd.Hessian(neg_log_like)((kdd_app, k_final, mt_max, kdp_app))
    H_inv = np.linalg.inv(H)

    # if user desires 99 percent confidence interval, they can input a string
    if isinstance(conf_95, str):
        cval = 2.58
    if conf_95:
        cval = 1.96
    else:
        cval = 1.0
    
    
    # get 95% confidence for kdd_app, k constant, mt_max, and kdp_app
    kdd_app_err = cval*np.sqrt(np.abs(H_inv[0][0]))
    k_final_err = cval*np.sqrt(np.abs(H_inv[1][1]))
    mt_max_err = cval*np.sqrt(np.abs(H_inv[2][2]))
    kdp_app_err = cval*np.sqrt(np.abs(H_inv[3][3]))
    
    mt_o_err = kdp_app_err/p_out*ctrl_pmt
    mt_fc_err = mt_max/mt_o*np.sqrt((mt_max_err/mt_max)**2.0+(mt_o_err/mt_o)**2.0)
    
            
    # getting R squared to assess model fit
    SS_res = sum([(pmt[i] - orig_sim_pmt[i])**2 for i in range(len(pmt))])
    SS_tot = sum([(pmt[i] - np.mean(pmt))**2 for i in range(len(pmt))])
    R2 = 1.0 - SS_res*1.0/SS_tot
    
    kdd_app_tup = (kdd_app, kdd_app_err)
    k_final_tup = (k_final, k_final_err)
    mt_fc_tup = (mt_fc, mt_fc_err)
    kdp_app_tup = (kdp_app, kdp_app_err)
    
    return kdd_app_tup, k_final_tup, mt_fc_tup, kdp_app_tup, R2
    
        
# 'kdp_app_bound': lower and upper bounds for kdp_app
def model_drug_binding(plate_df, equil_time=600.0, sirt_conc=800.0, 
                       kdp_app_bounds=[100.0, 100000.0], kdd_app_prior=1.0, mt_fc=[], 
                       sim_num=100, conf_95=False, xlim=[-3.1, 3.2], sirt_ylim=[-0.05, 1.25]):

    ### data organization
    
    # don't count DMSO as a drug
    drugs = plate_df.Drug.unique().tolist()
    dmso_idx = drugs.index('DMSO')
    drugs = drugs[:dmso_idx]+drugs[dmso_idx+1:]
    
    # data columns
    drug_conc_col = 'Drug_Conc_nM'; mean_col = 'MT_IntDen_Mean_Norm'; sem_col = 'MT_IntDen_SEM_Norm'
    agg_conc_col = 'Agg_Drug_Conc_nM'; agg_data_col = 'Agg_Norm_Data'
    
    
    # negative control
    sirt_neg_mean = plate_df[(plate_df.Drug == 'DMSO') & 
                            (plate_df.Time_Min == equil_time)][mean_col].values[0]
    sirt_neg_sem = plate_df[(plate_df.SirT_Conc_nM == sirt_conc) & 
                            (plate_df.Time_Min == equil_time)][sem_col].values[0]
    sirt_neg_agg = plate_df[(plate_df.Drug == 'DMSO') & 
                            (plate_df.Time_Min == equil_time)][agg_data_col].tolist()[0]
    sirt_drug_conc = np.zeros(len(sirt_neg_agg)).tolist()
    
    # getting equilibrium data
    if 'Time_Min' in plate_df.columns:
        plate_df = plate_df[plate_df.Time_Min == 
                            equil_time].sort(drug_conc_col).reset_index(drop=True)
      
    
    ### setting pre-sets for simulation
    
    # MT fold-change
    if not isinstance(mt_fc, list):
        mt_fc_range = [mt_fc]
    else:
        mt_fc_range = np.linspace(1.0, 2.0, sim_num)
    
    # define range for kdp_app
    kdp_range = 10.0**np.linspace(np.log10(kdp_app_bounds[0]), np.log10(kdp_app_bounds[1]), sim_num)
     
    # just in case
    kdp_range = np.unique(kdp_range)
        

    ### setting up biased walk simulation
        
    # will iterate through pairwise combinations of indices
    var_num = 2
    
    # setting starting indices all to integer average
    start_idx = np.floor(np.mean([0, sim_num]))
    start_combo = (np.ones(var_num)*start_idx).astype(int).tolist()
    max_range = int(max(sim_num-start_idx, start_idx))

    prod_R2_max = -np.inf
    opt_combo = []
    for i in range(max_range):
        
        range_i = max_range-i # descending (50, 49,...,2, 1)
        
        # make array with values to subtract or add
        delta_options = np.array([-range_i, 0, range_i])
        delta_array = np.array(list(product(delta_options, repeat=var_num)))
        
        # setting starting combo to the previous optimal combo
        # check to make sure opt_combo has been newly defined, or else use original start_combo
        if (i > 0) & (len(opt_combo) > 0):
            start_combo = opt_combo

        # making repeat-matrix of starting combination
        start_array = np.tile(start_combo, [len(delta_array), 1])
        
        # add the start array and the delta array to get combinations
        test_combos_i = start_array+delta_array
        
        # check if any values in test_combos exceed max or min threshold
        # discard these combinations
        test_combos_i = test_combos_i[~np.any((test_combos_i >= sim_num) | 
                                            (test_combos_i < 0), 1)]
        
        if len(mt_fc_range) == 1:
            test_combos_i = test_combos_i[test_combos_i[:, 0] == 0]
            
        if len(kdp_range) == 1:
            test_combos_i = test_combos_i[test_combos_i[:, 1] == 0]
        
        # iterate through this subset of index combinations
        for idx_combo in test_combos_i:
        
            mt_fc = mt_fc_range[idx_combo[0]]
            kdp_app = kdp_range[idx_combo[1]]

            # run sub-simulation for panel of drugs simultaneously
            # i.e. kdp_app and max mt_fc should be same for all drugs
            kdd_app_tup_list = []
            k_final_tup_list = []
            mt_fc_tup_list = []
            kdp_app_tup_list = []
            R2_list = []

            prod_R2 = 1.0  # initializing objective function value
            xlabels = []   # setting xlabels for later
            for idx, drug in enumerate(drugs):

                # getting data subset
                equil_drug = plate_df[plate_df.Drug == drug].sort(drug_conc_col).reset_index(drop=True)

                signal_means = equil_drug[mean_col].values       # competition means
                drug_conc = equil_drug.Drug_Conc_nM.values       # drug concentrations
                
                agg_data = np.array(sum(equil_drug[agg_data_col].tolist(), []))       # agg competition means
                agg_drug_conc = np.array(sum(equil_drug[agg_conc_col].tolist(), []))  # agg drug concentrations
                
                # don't include NaN values
                drug_conc = drug_conc.astype(float)[np.isfinite(signal_means)]
                signal_means = signal_means[np.isfinite(signal_means)]
                
                agg_drug_conc = agg_drug_conc.astype(float)[np.isfinite(agg_data)]
                agg_data = agg_data[np.isfinite(agg_data)]

                # sort just in case
                signal_means = signal_means[np.argsort(drug_conc)]
                drug_conc = np.sort(drug_conc)
                signal_means[signal_means < 0] = 0  # turn negative values into zero
                
                # setting xlabels for later
                if len(drug_conc) > len(xlabels):   # set x-axis labels
                    xlabels = drug_conc
                
                agg_data = agg_data[np.argsort(agg_drug_conc)]
                agg_drug_conc = np.sort(agg_drug_conc)
                
                # tack on DMSO values
                agg_data = np.array(sirt_neg_agg+agg_data.tolist())
                agg_drug_conc = np.array(sirt_drug_conc+agg_drug_conc.tolist())
                agg_data[agg_data < 0] = 0          # unnecessary since no values below zero, but whatever...

                # run simulation
                kdd_app_tup, k_final_tup, mt_fc_tup, kdp_app_tup, R2 = get_kdd_app(drug_conc, 
                                        signal_means, sirt_neg_mean, all_drug=agg_drug_conc, all_pmt=agg_data,
                                        kdp_app=kdp_app, kdd_app_prior=kdd_app_prior, 
                                        mt_fc=mt_fc, sim_num=sim_num, conf_95=conf_95, annotate=False)

                # record values in case we want to keep them
                kdd_app_tup_list.append(kdd_app_tup)
                k_final_tup_list.append(k_final_tup)
                mt_fc_tup_list.append(mt_fc_tup)
                kdp_app_tup_list.append(kdp_app_tup)
                R2_list.append(R2)
                
                # maximize with respect to R2 and kdd_app error
                prod_R2 *= R2
                
            # seeing if we have minimized error further
            if prod_R2 > prod_R2_max:
                
                prod_R2_max = prod_R2
                opt_combo = idx_combo

                # current finalized values and respective errors
                final_kdd_app_list = kdd_app_tup_list
                final_k_list = k_final_tup_list
                final_mt_fc_list = mt_fc_tup_list
                final_kdp_app_list = kdp_app_tup_list
                final_R2_list = R2_list
                
        # if only testing one combination overall, no need to loop
        if (len(mt_fc_range) == 1) & (len(kdp_range) == 1):
            break
                

    # further propagating error for kdp_app and mt_fc
    kdp_err_array = np.array(final_kdp_app_list)[:, 1]
    kdp_app_err = np.sqrt(np.sum((kdp_err_array)**2.0))/len(kdp_err_array)
    kdp_app = final_kdp_app_list[0][0]
    
    print 'Kdp_app = %s+/-%snM.' % (str(np.around(kdp_app, 3)), str(np.around(kdp_app_err, 3)))
    
    
    fc_err_array = np.array(final_mt_fc_list)[:, 1]
    mt_fc_err = np.sqrt(np.sum((fc_err_array)**2.0))/len(fc_err_array)
    mt_fc = final_mt_fc_list[0][0]
    
    print 'MT Polymer FC = %s+/-%s.\n' % (str(np.around(mt_fc, 3)), str(np.around(mt_fc_err, 3)))
    
    # define original MT and max MT for later
    mt_o = sirt_neg_mean*(1.0 + kdp_app/sirt_conc)
    mt_max = mt_o*mt_fc
    
    mt_o_err = kdp_app_err/sirt_conc*sirt_neg_mean
    mt_max_err = mt_o*mt_fc*np.sqrt((mt_o_err/mt_o)**2.0+(mt_fc_err/mt_fc)**2.0)
    
    # setting concentrations for simulation
    sim_d_conc = 10.0**np.linspace(np.log10(np.min(xlabels)), np.log10(np.max(xlabels)), 100)
    

    ### plotting section
    
    fig, axes = plt.subplots(figsize=(10, 20), nrows=3, ncols=1)
    
    # 082617: changed red to magenta
    colors = [[0, 0, 0], 
              [1.0, 0.5, 0], 
              [0, 0.6, 0], 
              [0, 0, 0.9]]
    
    
    # re-iterate through drugs
    for idx, drug in enumerate(drugs):
        
        print 'Drug: %s' % drug
        
        # getting data subset
        equil_drug = plate_df[plate_df.Drug == drug].sort(drug_conc_col).reset_index(drop=True)

        signal_means = equil_drug[mean_col].values       # competition means
        signal_sems = equil_drug[sem_col].values
        drug_conc = equil_drug.Drug_Conc_nM.values       # drug concentrations

        # don't include NaN values
        drug_conc = drug_conc.astype(float)[np.isfinite(signal_means)]
        signal_sems = signal_sems[np.isfinite(signal_means)]
        signal_means = signal_means[np.isfinite(signal_means)]
        
        # sort just in case
        signal_means = signal_means[np.argsort(drug_conc)]
        signal_sems = signal_sems[np.argsort(drug_conc)]
        drug_conc = np.sort(drug_conc)
        signal_means[signal_means < 0] = 0  # turn negative values into zero
        
        # get drug-dependent minimized constants
        kdd_app = final_kdd_app_list[idx][0]
        kdd_app_err = final_kdd_app_list[idx][1]
        
        print 'Kdd_app = %s+/-%snM.' % (str(np.around(kdd_app, 3)), str(np.around(kdd_app_err, 3)))
        print 'R squared of model fit: %s' % str(np.around(final_R2_list[idx], 4))
        
        # get site-occupancy
        site_occ = sim_d_conc / (kdd_app + sim_d_conc)
        site_occ_err = sim_d_conc*kdd_app_err/(kdd_app+sim_d_conc)**2.0
        
        # k constant and error
        k = final_k_list[idx][0]
        k_err = final_k_list[idx][1]
        
        # get simulated pmt values from model
        sim_mt_tot =  mt_o*np.exp(-k*sim_d_conc) + mt_max*(1.0 - np.exp(-k*sim_d_conc))
        sim_pmt = sim_mt_tot / (kdp_app/sirt_conc*sim_d_conc/kdd_app + kdp_app/sirt_conc + 1.0)
        
        # propagate error for sim_mt_tot
        sim_mt_err_term1 = mt_o*np.exp(-k*sim_d_conc)*np.sqrt((mt_o_err/mt_o)**2.0+(sim_d_conc*k_err)**2.0)
        sim_mt_err_term2 = mt_max*(1.0-np.exp(-k*sim_d_conc))*np.sqrt((mt_max_err/mt_max)**2.0+(np.exp(-k*sim_d_conc)*sim_d_conc*k_err/(1.0-np.exp(-k*sim_d_conc)))**2.0)
        sim_mt_err = np.sqrt(sim_mt_err_term1**2.0+sim_mt_err_term2**2.0)
        
        # mt fold-change and error
        sim_mt_fc = sim_mt_tot/mt_o
        sim_mt_fc_err = sim_mt_tot/mt_o*np.sqrt((sim_mt_err/sim_mt_tot)**2.0+(mt_o_err/mt_o)**2.0)
        
        # also get drug inhibitory constant
        sim_pmt_o = mt_o / (kdp_app/sirt_conc + 1.0)
        IC50_range = kdd_app/kdp_app*sirt_conc*(2.0*sim_mt_tot/sim_pmt_o - kdp_app/sirt_conc - 1.0)
        sim_pmt_fc = sim_pmt/sim_pmt_o
        sim_pmt_fc = np.around(sim_pmt_fc*2.0, 0)/2.0
        IC50 = np.mean(IC50_range[sim_pmt_fc == 0.5])
        
        cp_kdd_app = IC50 / (1.0 + sirt_conc/kdp_app)
        print 'Cheng-Prusoff Kdd_app: %s' % str(round_sig(cp_kdd_app, sig=4))
        
        

        # plot pmt data, simulated pmt data, and IC50
        axes[0].errorbar(np.log10(drug_conc), signal_means, yerr=list(signal_sems), 
                     color=colors[idx], label=drug, fmt='o', markeredgecolor='none')
        axes[0].plot(np.log10(sim_d_conc), sim_pmt, '-', color=colors[idx], label='', linewidth=2.0)
        axes[0].plot(np.log10(np.ones(2)*IC50), np.linspace(-0.05, 20000, 2), '-', color=colors[idx], 
                 linewidth=0.5, label='', markeredgecolor='none')
        
        # plot MT fold-change
        axes[1].plot(site_occ, sim_mt_fc, '-', color=colors[idx], label=drug, 
                     markeredgecolor='none', linewidth=2.0)
        axes[1].plot(site_occ, sim_mt_fc-sim_mt_fc_err, '--', color=colors[idx], label='', 
                     markeredgecolor='none', linewidth=1.5)
        axes[1].plot(site_occ, sim_mt_fc+sim_mt_fc_err, '--', color=colors[idx], label='', 
                     markeredgecolor='none', linewidth=1.5)
        
        # plot site occupancy
        axes[2].plot(np.log10(sim_d_conc), site_occ, '-', color=colors[idx], label=drug, 
                     markeredgecolor='none', linewidth=2.0)
        axes[2].plot(np.log10(sim_d_conc), site_occ-site_occ_err, '--', color=colors[idx], label='', 
                     markeredgecolor='none', linewidth=1.5)
        axes[2].plot(np.log10(sim_d_conc), site_occ+site_occ_err, '--', color=colors[idx], label='', 
                     markeredgecolor='none', linewidth=1.5)
        
        
        print
        
      
    ### finishing touches to plots
    
    ## SirT Competition
    
    # tidy up xlabels
    xlabels_str = [str(c)[:-2] if str(c)[-2:] == '.0' else str(c) for c in xlabels]
    
    # DMSO negative control
    axes[0].fill_between(np.log10(drug_conc), sirt_neg_mean-sirt_neg_sem, sirt_neg_mean+sirt_neg_sem, 
                     color='k', alpha=0.2, label='DMSO')
    axes[0].set_ylim(sirt_ylim)
    axes[0].set_xlim(xlim)
    axes[0].set_xticks(np.log10(xlabels)[::2])
    axes[0].set_xticklabels(xlabels_str[::2])
    axes[0].set_xlabel('Competing Unlabeled Drug (nM)')
    axes[0].set_ylabel('SirTub Signal (FC)', fontsize=30) 
    axes[0].set_title('MT Drug vs. %inM SirT Competition\n' % int(sirt_conc))
    
    
    ## MT polymer fold-change
    
    # DMSO negative control
    axes[1].fill_between(np.log10(drug_conc), 1.0-mt_o_err/mt_o*2.0, 1.0+mt_o_err/mt_o*2.0,
                     color='k', alpha=0.2, label='DMSO')
    axes[1].set_xlim([-0.05, 1.05])
    axes[1].set_ylim([0.5, 2.75])
    axes[1].set_xlabel('Site Occupancy')
    axes[1].set_ylabel('MT Polymer FC')
    axes[1].set_title('Drug-Induced MT Polymerization\n')


    ## Site occupancy
    
    axes[2].set_ylim([-0.05, 1.05])
    axes[2].set_xlim(xlim)
    axes[2].set_xticks(np.log10(xlabels)[::2])
    axes[2].set_xticklabels(xlabels_str[::2])
    axes[2].set_xlabel('Competing Unlabeled Drug (nM)')
    axes[2].set_ylabel('Site Occupancy')
    axes[2].set_title('Drug-Site Occupancy\n')
    
    
    ## all plots

    set_plot_params(fig=fig, ax=axes[0], poster=True)
    set_plot_params(fig=fig, ax=axes[1], poster=True)
    set_plot_params(fig=fig, ax=axes[2], poster=True)
    
    axes[0].legend(loc='best', prop={'size':22}, 
                    fancybox=True, framealpha=0.5, numpoints=1)
    axes[1].legend(loc='lower right', prop={'size':22}, 
                    fancybox=True, framealpha=0.5, numpoints=1)
    axes[2].legend(loc='best', prop={'size':22}, 
                    fancybox=True, framealpha=0.5, numpoints=1)
    
    fig.tight_layout()
    plt.show()
        
        
##########################################################################################

        
### code for quantifying EB3 comets
        
# function to apply threshold for EB3 'comets', cell 'area', or 'both'
# 'c_a_thresh' is either a single integer or a list of integers
# if running 'both', place the comet threshold first
def apply_thresh(image_path, c_a_thresh, method='comets'):
    
    if not isinstance(c_a_thresh, list):
        c_a_thresh = [c_a_thresh]*2
    
    c_thresh = c_a_thresh[0]  # comet threshold
    a_thresh = c_a_thresh[1]  # cell area threshold
    
    # reading in image
    orig_image = skimage.io.imread(image_path, plugin='tifffile')
    gauss_10 = ndimage.gaussian_filter(orig_image, sigma=10, order=0)
    
    if (method.lower() == 'comets') | (method.lower() == 'both'):
        comet_image = orig_image - gauss_10
        comet_image[comet_image > 2**15] = 0
        comet_image[comet_image < c_thresh] = 0
        comet_image[comet_image >= c_thresh] = 1
        
    if (method.lower() == 'area') | (method.lower() == 'both'):
        gauss_10[gauss_10 < a_thresh] = 0
        gauss_10[gauss_10 >= a_thresh] = 1
        
    if method.lower() == 'comets':
        return comet_image
    elif method.lower() == 'area':
        return gauss_10
    else:
        return comet_image, gauss_10


# function to detect EB3 comets
# NOTE: all background should be subtracted and image should be thresholded
def detect_comets(comet_image, min_area=3, max_area=50):
    
    counter = 0  # setting initial comet number to zero
    
    lengths = []
    label_image = skimage.measure.label(comet_image)
    for region in skimage.measure.regionprops(label_image, intensity_image=comet_image):
        
        # skip areas smaller than 3 pixels (likely noise)
        if (region.area < min_area) | (region.area > max_area):
            continue
            
        # assuming all background is set to zero
        if region.mean_intensity < 1:
            continue
            
        # getting length of particle
        c_length = region.major_axis_length * 1.0
        lengths.append(c_length)  # record
        
        counter += 1  # updating comet number
        
    med_length = np.median(lengths)
        
    return counter*1.0, med_length


# useful for dealing with time-course comet data
def nan2zero(x):
    
    if not isinstance(x, str):
        if np.isnan(x):
            return 0.0
        else:
            return x
    else:
        return x
  

# function to quantify eb3 comet data from InCELL Analyzer
def auto_comet_quant(image_dir, well_info, c_thresh=151, field_prec='fld ', field_proc='- time', 
                     frame_prec='time ', frame_proc=' - ', time_prec=' - ', time_proc=' ms)', 
                     dead_time=15.0, min_area_per=10.0, im_file_list=[]):
    
    
    ## verification of inputs section

    if len(im_file_list) > 0:
    
        # drug condition of interest
        drug_im_files = [f for f in im_file_list if well_info.lower() in f.lower()]
    
    else:
        # drug condition of interest
        drug_im_files = [f for f in os.listdir(image_dir) if ('.tif' in f.lower()) & 
                                                         (well_info.lower() in f.lower())]

    # verify we actually pulled images
    if len(drug_im_files) == 0:
        print 'No images were found when using the well-info string provided...'
        return
        
    
    ## get pixel area of an entire image
    
    tot_pix_area = len(skimage.io.imread(image_dir+drug_im_files[0], plugin='tifffile').ravel())


    ## getting max and min frame number
    
    max_frame = 0     # assuming we have at least 1 frame
    min_frame = 1000  # assuming we have fewer than 1000 frames
    for im_file in drug_im_files:

        # getting index positions of frame designators      
        frame_prec_idx = im_file.index(frame_prec)
        frame_proc_idx = im_file[frame_prec_idx:].index(frame_proc)+frame_prec_idx

        # getting frame info
        frame = int(im_file[frame_prec_idx+len(frame_prec):frame_proc_idx])

        if frame > max_frame:
            max_frame = frame

        if frame < min_frame:
            min_frame = frame

    frame_num = max_frame - min_frame + 1
    
    
    ## presetting arrays section
    
    # some of these won't be used until later
    comet_count = np.zeros(len(drug_im_files))
    comet_length = np.zeros(len(drug_im_files))
    cell_areas = np.zeros(len(drug_im_files))
    field_assign = np.zeros(len(drug_im_files)) # will need to re-order later
    frame_assign = np.zeros(len(drug_im_files)) # will need to re-order later
    times = np.zeros(len(drug_im_files))        # times are in msec by default
    
    
    ## thresholding section
    
    # determine area proportion of total image area
    a_thresh_array = np.zeros(100)-1     # assuming fewer than 100 fields...
    area_percents = np.zeros(100)-1
    already_quant_idx = []
    ignore_fields = []
    for i, im_file in enumerate(drug_im_files):
        
        # getting index positions of field and frame designators
        field_prec_idx = im_file.index(field_prec)
        field_proc_idx = im_file[field_prec_idx:].index(field_proc)+field_prec_idx
        field = int(im_file[field_prec_idx+len(field_prec):field_proc_idx])
        
        if (time_prec != None) & (time_proc != None):
            frame_prec_idx = im_file[field_proc_idx:].index(frame_prec)+field_proc_idx
            frame_proc_idx = im_file[frame_prec_idx:].index(frame_proc)+frame_prec_idx
            frame = int(im_file[frame_prec_idx+len(frame_prec):frame_proc_idx])
        else:
            frame = 0
            min_frame = 0
            max_frame = 0
        
        # only using first frame to determine threshold
        if frame == min_frame:
            
            # reading in image
            orig_image = skimage.io.imread(image_dir+im_file, plugin='tifffile')
            
            gauss_10 = ndimage.gaussian_filter(orig_image, sigma=10.0, order=0)  # blur
           
            a_thresh_array[field-1] = int(threshold_otsu(gauss_10))  # take threshold
            
            # binarize using the otsu threshold
            gauss_10[gauss_10 < a_thresh_array[field-1]] = 0
            gauss_10[gauss_10 >= a_thresh_array[field-1]] = 1
            
            # calculating pixel area percent of cells
            cell_area = np.sum(np.sum(gauss_10))
            area_percents[field-1] = cell_area * 100.0 / tot_pix_area
            
            # notate sub-optimal images
            if area_percents[field-1] < min_area_per*1.0:
                ignore_fields.append(field)
                
                # don't quantify cell area or comets
                cell_areas[i] = np.nan
                comet_count[i] = np.nan
                
            else:
                # record cell areas
                cell_areas[i] = cell_area
                    
            # noting the image as already quantified
            already_quant_idx.append(i)

    # get rid of unnecessary zeros
    # note: each position corresponds to a respective field position
    a_thresh_array = a_thresh_array[a_thresh_array > -1]
    area_percents = area_percents[area_percents > -1]
    
    
    ## image processing section
    
    for i, im_file in enumerate(drug_im_files):
        
        # getting index positions of field and frame designators
        field_prec_idx = im_file.index(field_prec)
        field_proc_idx = im_file[field_prec_idx:].index(field_proc)+field_prec_idx
        field = int(im_file[field_prec_idx+len(field_prec):field_proc_idx])
        field_assign[i] = field
        
        
        if (time_prec != None) & (time_proc != None):
            frame_prec_idx = im_file[field_proc_idx:].index(frame_prec)+field_proc_idx
            frame_proc_idx = im_file[frame_prec_idx:].index(frame_proc)+frame_prec_idx
            frame = int(im_file[frame_prec_idx+len(frame_prec):frame_proc_idx])
            frame_assign[i] = frame
        
            # for the InCELL Analyzer, time comes after timepoint, which follows frame
            time_prec_idx = im_file[frame_proc_idx:].index(time_prec)+frame_proc_idx
            time_proc_idx = im_file[time_prec_idx:].index(time_proc)+time_prec_idx
            time_msec = int(im_file[time_prec_idx+len(time_prec):time_proc_idx])
            times[i] = time_msec
        
            if field in ignore_fields:
                cell_areas[i] = np.nan
                comet_count[i] = np.nan
            else:
                # reading in image
                orig_image = skimage.io.imread(image_dir+im_file, plugin='tifffile')
                
                if i not in already_quant_idx:
                    # blurring to get cell area, then binarize
                    gauss_10 = ndimage.gaussian_filter(orig_image, sigma=10.0, order=0)
                    gauss_10[gauss_10 < a_thresh_array[field-1]] = 0
                    gauss_10[gauss_10 >= a_thresh_array[field-1]] = 1

                    # calculating pixel area of cells
                    cell_areas[i] = np.sum(np.sum(gauss_10))
                
                # quantify comets
                gauss_5 = ndimage.gaussian_filter(orig_image, sigma=5.0, order=0)    # blur
    
                c_image = orig_image - gauss_5                                       # subtract background
                c_image[c_image > 0.99*(2**16)] = 0                                  # removing saturated pixels
            
                c_image[c_image < c_thresh] = 0
                c_image[c_image >= c_thresh] = 1                                     # binarize
                
                comet_count[i], comet_length[i] = detect_comets(c_image, 
                                        min_area=3, max_area=100)                    # count comets
            
                           
    # getting number of total fields, and acceptable fields
    field_num = int(np.max(field_assign))
    acc_field_num = field_num - len(ignore_fields)
                        
    # recording comet quant data in dataframe
    comet_df = pd.DataFrame(field_assign, columns=['Field'])
    comet_df['Field'] = field_assign
    
    comet_df['Frame'] = frame_assign
    comet_df['Time_Min'] = times *1.0/60000  # convert times from millisec to min
    
    # now re-define time array and drop from df
    # this is necessary for compatibility with process_image_data()
    times = np.array(comet_df.Time_Min.tolist())
    times = times[:frame_num]                          # truncating to make compatible with process_image_data()

    times = times + dead_time                          # adding dead-time
    times[0] = 0.0                                     # fix first timepoint
    comet_df = comet_df.drop('Time_Min', axis=1)       # drop unnecessary column
    
    comet_df = comet_df.sort(['Field', 'Frame'])
    comet_argsort = np.array(comet_df.index.tolist())  # getting sorting order
    comet_df = comet_df.reset_index(drop=True)  
    
    # sort comets and area arrays
    comet_count = comet_count[comet_argsort]
    comet_length = comet_length[comet_argsort]
    cell_areas = cell_areas[comet_argsort]*1.0
    
    # now assemble comet length data and cell area data
    # will continue processing it later
    feret_df = comet_df.copy()
    area_df = comet_df.copy()

    # now process comet and feret dfs
    comet_df['Comet'] = comet_count
    feret_df['Feret'] = comet_length
    area_df['Cell_Area'] = cell_areas
    opt_comet_df = process_image_data(comet_df, times, 'comets').ix[:, :acc_field_num]
    opt_feret_df = process_image_data(feret_df, times, 'feret').ix[:, :acc_field_num]
    opt_area_df = process_image_data(area_df, times, 'area').ix[:, :acc_field_num]
    
    
    
    ## final data processing
    
    # get average number of comets among fields and normalize by average cell area among fields 
    # get median also, just in case
    opt_comet_df['Comet_Mean'] = opt_comet_df.ix[:, :acc_field_num].apply(lambda x: 
                                                                          np.mean(np.array(x)), axis=1)
    opt_comet_df['Comet_Median'] = opt_comet_df.ix[:, :acc_field_num].apply(lambda x: 
                                                                          np.median(np.array(x)), axis=1)
    opt_comet_df['Time_Min'] = times
    
    # do the same for the df of comet lengths
    opt_feret_df['Feret_Mean'] = opt_feret_df.ix[:, :acc_field_num].apply(lambda x: 
                                                                          np.mean(np.array(x)), axis=1)
    opt_feret_df['Feret_Median'] = opt_feret_df.ix[:, :acc_field_num].apply(lambda x: 
                                                                          np.median(np.array(x)), axis=1)
    opt_feret_df['Time_Min'] = times
    
    # and for the df of cell areas 
    opt_area_df['Cell_Area_Mean'] = opt_area_df.ix[:, :acc_field_num].apply(lambda x: 
                                                                          np.mean(np.array(x)), axis=1)
    opt_area_df['Cell_Area_Median'] = opt_area_df.ix[:, :acc_field_num].apply(lambda x: 
                                                                          np.median(np.array(x)), axis=1)
    opt_area_df['Time_Min'] = times
    
    # now normalize comets by cell area
    opt_comet_df['Comet_Mean'] = opt_comet_df.Comet_Mean / opt_area_df.Cell_Area_Mean
    opt_comet_df['Comet_Median'] = opt_comet_df.Comet_Median / opt_area_df.Cell_Area_Median
    
    # now convert comets and comet lengths to fold-change
    opt_comet_df['Comet_Mean'] = opt_comet_df.Comet_Mean / opt_comet_df.Comet_Mean.values[0]
    opt_comet_df['Comet_Median'] = opt_comet_df.Comet_Median / opt_comet_df.Comet_Median.values[0]
    
    opt_feret_df['Feret_Mean'] = opt_feret_df.Feret_Mean / opt_feret_df.Feret_Mean.values[0]
    opt_feret_df['Feret_Median'] = opt_feret_df.Feret_Median / opt_feret_df.Feret_Median.values[0]
    
    
    return opt_comet_df, opt_feret_df, opt_area_df
        

    
# function to merge biorep comet data
# 'method': can be 'comet', 'feret', or 'area'
def comet_biorep(path_list):
    
    # datasets all have fold-change data normalized to Time 0 control (0.1% DMSO + 10uM Ver)
    df_list = [pd.read_csv(path) for path in path_list]
    
    
    # get unique drug conditions that are present in all biorepeats
    # use 'Drug' and 'Drug_Conc_nM' columns to be safe
    unq_cond = df_list[0][['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1).tolist()
    for df in df_list:
        unq_cond_i = df[['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1).tolist()
        unq_cond = set(unq_cond_i).intersection(unq_cond)  # get intersection
        
    new_df_list = []
    for df in df_list:
        
        # only take those conditions that are present in all the datasets
        df['Key_Col'] = df[['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1)
        df = df[df.Key_Col.isin(unq_cond)].reset_index(drop=True)
        
        # 'Drug_Conc_nM' has been converted into strings since it has been read out as *.csv
        # this is due to the 'DMSO' rows that are noted as '0.01%'
        df['Drug_Conc_nM'] = df.Drug_Conc_nM.apply(lambda x: float(x) if '%' not in x else x)
        
        # sort appropriate columns and append df
        df = df.sort(['Drug', 'Drug_Conc_nM']).reset_index(drop=True)
        new_df_list.append(df)
    
    biorep_df =  new_df_list[0].copy()
       
    # comet numbers
    biorep_df['Comet_Mean_Norm'] = np.mean([df.Comet_Mean_Norm for df in new_df_list], 0)
    biorep_df['Comet_SEM_Norm'] = np.sqrt(np.sum([df.Comet_SEM_Norm**2 for df in new_df_list], 0)) / len(path_list)
    
    # comet lengths (i.e. 'feret')
    biorep_df['Feret_Mean_Norm'] = np.mean([df.Feret_Mean_Norm for df in new_df_list], 0)
    biorep_df['Feret_SEM_Norm'] = np.sqrt(np.sum([df.Feret_SEM_Norm**2 for df in new_df_list], 0)) / len(path_list)
    
    # cell areas
    biorep_df['Cell_Area_Mean_Norm'] = np.mean([df.Cell_Area_Mean_Norm for df in new_df_list], 0)
    biorep_df['Cell_Area_SEM_Norm'] = np.sqrt(np.sum([df.Cell_Area_SEM_Norm**2 for df in new_df_list], 0)) / len(path_list) 
    
    # drop unnecessary columns
    biorep_df = biorep_df.drop('Key_Col', 1)
    if 'Rep_Wells' in biorep_df.columns:
        biorep_df = biorep_df.drop('Rep_Wells', 1)
    
    return biorep_df

    
# function to annotate and aggregate all EB3 comet/feret/area data
def agg_eb3_data(comet_data_list, feret_data_list, area_data_list, 
                 report_df, output_path, rep_att='Mean', well_list=[]):
    
    ## comet data
    
    # annotate data
    comet_plate = order_well_info(report_df, comet_data_list, method='comets', 
                                        rep_att=rep_att, well_list=well_list)
    
    # comets; normalize out photobleaching effects
    comet_plate_ctrl = comet_plate[comet_plate.Drug == 'DMSO'].reset_index(drop=True)

    cond_num = len(comet_plate) / len(comet_plate_ctrl)

    comet_plate_ctrl = pd.concat([comet_plate_ctrl]*cond_num).reset_index(drop=True)

    comet_plate['Ctrl_Comet_Mean'] = comet_plate_ctrl.Comet_Mean
    comet_plate['Ctrl_Comet_SEM'] = comet_plate_ctrl.Comet_SEM

    comet_plate['Comet_Mean_Norm'] = comet_plate[['Comet_Mean', 'Ctrl_Comet_Mean']].apply(lambda x: 
                                                    x[0]/x[1], axis=1)
    comet_plate['Comet_SEM_Norm'] = comet_plate[['Comet_Mean', 'Ctrl_Comet_Mean', 
                                                         'Comet_SEM', 'Ctrl_Comet_SEM']].apply(lambda x: 
                            x[0]/x[1] * np.sqrt((x[2]/x[0])**2.0+(x[3]/x[1])**2.0), axis=1)

    try:
        os.mkdir(output_path+'Merged_Data/')
    except:
        pass
    
    
    ## comet lengths
    
    # annotate data
    feret_plate = order_well_info(report_df, feret_data_list, method='feret', 
                                        rep_att=rep_att, well_list=well_list)

    # comet lengths; normalize out photobleaching effects
    feret_plate_ctrl = feret_plate[feret_plate.Drug == 'DMSO'].reset_index(drop=True)

    cond_num = len(feret_plate) / len(feret_plate_ctrl)

    feret_plate_ctrl = pd.concat([feret_plate_ctrl]*cond_num).reset_index(drop=True)

    feret_plate['Ctrl_Feret_Mean'] = feret_plate_ctrl.Feret_Mean
    feret_plate['Ctrl_Feret_SEM'] = feret_plate_ctrl.Feret_SEM

    feret_plate['Feret_Mean_Norm'] = feret_plate[['Feret_Mean', 'Ctrl_Feret_Mean']].apply(lambda x: 
                                                    x[0]/x[1], axis=1)
    feret_plate['Feret_SEM_Norm'] = feret_plate[['Feret_Mean', 'Ctrl_Feret_Mean', 
                                                         'Feret_SEM', 'Ctrl_Feret_SEM']].apply(lambda x: 
                            x[0]/x[1] * np.sqrt((x[2]/x[0])**2.0+(x[3]/x[1])**2.0), axis=1)


    ## cell areas

    # a little trickier to work with...
    area_plate = order_well_info(report_df, area_data_list, method='area', 
                                           rep_att=rep_att, well_list=well_list)

    # take fold-change of area with respect to time
    area_fc_plate = area_plate.copy()
    area_fc_plate = area_fc_plate.drop('Cell_Area_Mean', 1).drop('Cell_Area_SEM', 1)

    # for every well, take fold-change for each condition (i.e. each label)
    for col in area_fc_plate.columns:
        if rep_att+'_' not in col:
            continue

        for label in area_fc_plate.Labels.tolist():

            area_fc_plate[col][area_fc_plate.Labels == 
                                label] = area_fc_plate[col][area_fc_plate.Labels == 
                                label] / area_fc_plate[col][(area_fc_plate.Labels == 
                                label) & (area_fc_plate.Time_Min == 0)].values[0]

    # now get means and sems for area fold-change
    area_fc_cols = [col for col in area_fc_plate.columns if rep_att+'_' in col]

    # note use of np.nanmean
    area_fc_plate['Cell_Area_Mean'] = area_fc_plate[area_fc_cols].apply(lambda x: 
                                                np.nanmean(x), axis=1)

    # sem requires number of relevant elements
    area_fc_plate['Cell_Area_SEM'] = area_fc_plate[area_fc_cols].apply(lambda x: 
                                                np.nanstd(x)/np.sqrt((~np.isnan(x)).sum()), axis=1)

    # cell areas; normalize out photobleaching effects
    area_fc_plate_ctrl = area_fc_plate[area_fc_plate.Drug == 'DMSO'].reset_index(drop=True)

    cond_num = len(area_fc_plate) / len(area_fc_plate_ctrl)

    area_fc_plate_ctrl = pd.concat([area_fc_plate_ctrl]*cond_num).reset_index(drop=True)

    area_fc_plate['Ctrl_Cell_Area_Mean'] = area_fc_plate_ctrl.Cell_Area_Mean
    area_fc_plate['Ctrl_Cell_Area_SEM'] = area_fc_plate_ctrl.Cell_Area_SEM

    area_fc_plate['Cell_Area_Mean_Norm'] = area_fc_plate[['Cell_Area_Mean', 'Ctrl_Cell_Area_Mean']].apply(lambda x: 
                                                    x[0]/x[1], axis=1)
    area_fc_plate['Cell_Area_SEM_Norm'] = area_fc_plate[['Cell_Area_Mean', 'Ctrl_Cell_Area_Mean', 
                                                         'Cell_Area_SEM', 'Ctrl_Cell_Area_SEM']].apply(lambda x: 
                            x[0]/x[1] * np.sqrt((x[2]/x[0])**2.0+(x[3]/x[1])**2.0), axis=1)
    
    
    ## data aggregation
    
    # make new df which aggregates all comet, feret, and cell area fold-change data 
    # only means and sems
    eb3_data = comet_plate.copy()[['Drug', 'Drug_Conc_nM', 'Labels', 
                                                       'Time_Min', 'Comet_Mean_Norm', 'Comet_SEM_Norm']]

    eb3_data['Feret_Mean_Norm'] = feret_plate['Feret_Mean_Norm']
    eb3_data['Feret_SEM_Norm'] = feret_plate['Feret_SEM_Norm']
    eb3_data['Cell_Area_Mean_Norm'] = area_fc_plate['Cell_Area_Mean_Norm']
    eb3_data['Cell_Area_SEM_Norm'] = area_fc_plate['Cell_Area_SEM_Norm']

    # no nan values should be present; if there are, convert to 0s
    eb3_data = eb3_data.applymap(nan2zero)
    return eb3_data
    

##########################################################################################


# takes in list of dataframes (must have same number of frames)
# also takes in list of labels (in same order as dataframes)
# recommended: list control first
# 'readout' can be 'comet', 'mt', or 'area'
def plot_image_data(im_data_dfs, labels=[], fig=[], plot_fc=False, color='multicolor', 
                xlim=[], ylim=[], fill=False, readout='mt', 
                all_in_one_df=True, y_label=True):
    
    if all_in_one_df:  # only one dataframe is inputted for ifm data
        unq_cond = list(im_data_dfs.Labels.unique())
        
        indiv_dfs = []
        for cond in unq_cond:
            indiv_dfs.append(im_data_dfs[im_data_dfs.Labels == 
                                     cond].reset_index(drop=True))    
        im_data_dfs = indiv_dfs
       
    if not isinstance(im_data_dfs, list):
        im_data_dfs = [im_data_dfs]
    if len(labels) > 0:
        if not isinstance(labels, list):
            labels = [labels]
    
    if color.lower() == 'multicolor':
        # black, green, blue, red, magenta, cyan, orange, yellow, grey
        color_list = [[0, 0, 0],
                      [0, 0.8, 0],
                      [0, 0, 0.9],
                      [0.95, 0, 0],
                      [0.75, 0, 0.9],
                      [0, 0.9, 0.9], 
                      [1.0, 0.5, 0],
                      [0.95, 0.95, 0],
                      [0.6, 0.6, 0.6], 
                      [0, 0.4, 0], 
                      [0, 0, 0.4], 
                      [0.4, 0, 0], 
                      [0.4, 0, 0.4], 
                      [0, 0.4, 0.4]]  
    else:
        # greens
        color_list = [[0, 0, 0],
                      [0, 0.3, 0],
                      [0, 0.4, 0], 
                      [0, 0.5, 0],
                      [0, 0.6, 0],
                      [0, 0.7, 0],
                      [0, 0.8, 0],
                      [0, 0.9, 0], 
                      [0, 0.95, 0]] 
        
        if len(im_data_dfs) == 1:
            color_list = [[0, 0.6, 0]]
    
    if isinstance(fig, list):
        fig = plt.figure()
        
    for idx, df in enumerate(im_data_dfs):
        
        x_time = np.array(df['Time_Min'].tolist())
        
        # plot comet numbers
        if readout.lower() == 'comet':
            y = np.array(df['Comets_Mean'].tolist())
            y_sem = np.array(df['Comets_SEM'].tolist())

        elif readout.lower() == 'mt':
            y = np.array(df['MT_IntDen_Mean_Norm'].tolist())
            y_sem = np.array(df['MT_IntDen_SEM_Norm'].tolist())
        
        elif readout.lower() == 'area':
            y = np.array(df['Cell_Area_Mean'].tolist())
            y_sem = np.array(df['Cell_Area_SEM'].tolist())
            
        # if fold-change desired
        if plot_fc:
            y = y/y[0]
            y_sem = y_sem/y[0]
            fc_label = ' (FC)'
        else:
            fc_label = ''
            
        if len(labels) > 0:
            label = labels[idx]
        else:
            label = None
        
        # if only 1 field, then SEM=0 for every frame --> no errorbars
        if not np.all(y_sem == 0):
            if not fill:
                plt.errorbar(x_time, y, yerr=y_sem, 
                             color=color_list[idx], 
                             linewidth=1.0)
            else:
                plt.fill_between(x_time, y+y_sem, y-y_sem,
                             color=color_list[idx], alpha=0.5)
            
        plt.plot(x_time, y, color=color_list[idx], 
                 linewidth=3.0, label=label)
    
    if len(labels) > 0:
        plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
        
    if len(xlim) > 0:
        plt.xlim(xlim)
        
    if len(ylim) > 0:
        plt.ylim(ylim)
        
        
    set_plot_params()
    
    if fill:
        remove_border(bottom=True, left=True)#, top=True, right=True)
        plt.grid()
    else:
        remove_border(bottom=True, left=True)
        
    plt.xlabel('Time (min)', fontsize=25)
    
    if not y_label:
        pass
    
    elif readout.lower() == 'comet':
        plt.ylabel('Comets / Cell Area'+fc_label, fontsize=25) #40
            
    elif readout.lower() == 'mt':
        plt.ylabel('Normlized Intensity' +fc_label, fontsize=25)
    
    elif readout.lower() == 'area':
        plt.ylabel('Cell Area'+fc_label, fontsize=25)
        
        
    plt.gca().yaxis.tick_left()
    plt.show()
    

##########################################################################################


### code to score mitosis

# function to segment mitosis in fiji
# NOTE: function assumes InCELL images have been binned 2x2
# NOTE: definitely input threshold (based on negative control image); allows more consistent quantification
def fiji_mitosis(image_dir, thresh, mitosis_ch='wix 1).tif', bin2x2=True):
    
    # setting gaussian blur parameters and area cut-offs
    if bin2x2:
        sig1 = 12.5
        sig2 = 2.5
    else:
        sig1 = 25
        sig2 = 5
    
    
    ## make temporary directory to dump images for segmentation in FIJI
    
    fiji_input_path = image_dir+'../FIJI_input/'
    try:
        os.mkdir(fiji_input_path)
        
        # get all relevant image files and copy to fiji directory
        mitosis_files = [f for f in os.listdir(image_dir) if ('.tif' in f.lower()) & (mitosis_ch in f)]
        
        # do background subtractions in python to save time
        for f in mitosis_files:
            
            orig_image = skimage.io.imread(image_dir+f, plugin='tifffile')   # read in image

            # blurring image and subtracting interphase cells
            gauss_12d5 = ndimage.gaussian_filter(orig_image, sigma=sig1, order=0)

            subtract_1 = orig_image - gauss_12d5
            subtract_1[subtract_1 < 0] = 0
            subtract_1[subtract_1 > 0.9*(2**16)] = 0        # negative numbers turn to very big numbers weirdly...

            # now smooth the result of the image subtraction to make thresholding more reliable
            gauss_subtract = ndimage.gaussian_filter(subtract_1, sigma=sig2, order=0) 

            gauss_subtract[gauss_subtract < thresh] = 0     # binarizing
            gauss_subtract[gauss_subtract >= thresh] = 255
            
            tiff.imsave(fiji_input_path+f, gauss_subtract)  # save result in temporary FIJI directory

    except:
        pass
    
    
    ## run FIJI watershed segmentation
  
# fiji macro:  
# input_path = <input path>;
# output_path = <output path>;

# file_list = getFileList(input_path);  // there should only be image files here!

# for (i=0; i<file_list.length; i++) {
# 		setBatchMode(true);
		
# 		filename = file_list[i];
# 		image_name = substring(filename, 0, indexOf(filename, "."));

# 		open(input_path+filename);
# 		run("Set Scale...", "distance=1 known=1 global");
# 		setAutoThreshold("Default dark");
# 		setOption("BlackBackground", true);
# 		run("Make Binary");
# 		run("Watershed");
# 		saveAs("Tiff", output_path+image_name+"_mitosis_watershed.tif");
# 		close();
# }

# run("Quit");
# run("Quit");
    
    
    mitotic_script = open('mitosis_watershed_script.ijm','r')
    mitotic_script = mitotic_script.read()

    # make sure to export segmented images into original image folder
    mitotic_script = mitotic_script.replace('<input path>', 
                                            '"'+fiji_input_path+'"').replace('<output path>', '"'+image_dir+'"')
    
    # writing the new comet script and replacing template
    mitotic_script_temp = open('mitosis_watershed_script_temp.ijm', 'w')

    mitotic_script_temp.write(mitotic_script)
    mitotic_script_temp.close()

    # the actual comet segmentation
    os.system('/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx -macro /Users/javier/Desktop/Mitchison_Lab/Coding_Stuff/mitosis_watershed_script_temp.ijm')

    # deleting the temporary ijm files
    os.remove('mitosis_watershed_script_temp.ijm')
    
    # now delete unnecessary FIJI input directory
    shutil.rmtree(fiji_input_path, ignore_errors=True)
    

# written also for Eribulin-SiR stain

# function that opens a specified image and counts the number of EB3 cells
# assumes user has set an appropriate z-position such that only mitotic cells will be counted
# 'mit_thresh': empirically determined mitotic intensity threshold for plate
# if not binning 2x2, then set min_area to 200
def count_mitosis(image_path, min_area=50, bin2x2=True, circ_filter=0.75):
    

    # read in image
    water_image = skimage.io.imread(image_path, plugin='tifffile')
    
    mit_count = 0  # counting mitotic cells
    label_image = skimage.measure.label(water_image)
    for region in skimage.measure.regionprops(label_image, intensity_image=water_image):

        # skip areas smaller than 50 pixels (likely noise)
        if region.area < min_area:
            continue

        # getting length and width, and circularity of particle
        major_ax = region.major_axis_length * 1.0
        minor_ax = region.minor_axis_length * 1.0
        circularity = minor_ax / major_ax

        # should between 0.75-1.0 for something circular, or else not likely a mitotic cell
        if circularity < circ_filter:
            continue

        mit_count += 1.0  # record area

    return mit_count


# function to score "mitotic index"
# not conventional mitotic index: normalized by cell area instead of cell count
# this function assumes mitotic and normal cell images are all in the same input folder
# assumes one timepoint, one well at a time
# 'well_info': designator for desired well; will search for this string in the image filenames
# 'mitosis_ch': designator for mitotic images; ""
# 'cell_ch': designator for images with all cells; ""
# if only mitotic count (i.e. not index) desired, let cell_ch=[]
def score_mitotic(image_dir, well_info, mitosis_ch='mitosis_watershed', area_ch='DAPI).tif', bin2x2=True):
    
    if image_dir[-1] != '/':
        image_dir += '/'
    
    # getting all relevant image filenames; split into mitotic and normal images
    all_images = [f for f in os.listdir(image_dir) if ('.tif' in f.lower()) and (well_info in f)]
    
    mit_images = [im for im in all_images if mitosis_ch in im]  # mitotic images
    
    if not isinstance(area_ch, list):
        reg_images = [im for im in all_images if area_ch in im]      # regular cell images
    
    # counting mitotic cells in all fields and summing together
    well_mit_count = 0
    for im in mit_images:
        well_mit_count += count_mitosis(image_dir+im, bin2x2=bin2x2)

    if not isinstance(area_ch, list):
        # need to normalize particle count by total cell area in a well
        well_cell_area = score_area(image_dir, well_info, image_ch=area_ch, bin2x2=bin2x2)
        
        # calculate mitotic idx
        mit_idx = well_mit_count * 1.0/ well_cell_area
        return mit_idx
    else:
        return well_mit_count * 1.0
        
        
##########################################################################################
        
        
### code to score micronucleation
        
# function to segment micronuclei in fiji
# NOTE: function assumes InCELL images have been binned 2x2
# NOTE: definitely input threshold (based on negative control image); allows more consistent quantification
def fiji_micronuc(image_dir, thresh, micronuc_ch='DAPI).tif'):
    
    
    ## make temporary directory to dump images for segmentation in FIJI
    
    fiji_input_path = image_dir+'../FIJI_input/'
    try:
        os.mkdir(fiji_input_path)
        
        # get all relevant image files and copy to fiji directory
        micronuc_files = [f for f in os.listdir(image_dir) if ('.tif' in f.lower()) & (micronuc_ch in f)]
        for f in micronuc_files:
            shutil.copy2(image_dir+f, fiji_input_path+f)
    except:
        pass
    
    
    ## run FIJI watershed segmentation
    
# fiji macro:
# input_path = <input path>;
# output_path = <output path>;
# thresh = <thresh>;
# sigma_value = <sigma_value>;

# file_list = getFileList(input_path);  // there should only be image files here!

# for (i=0; i<file_list.length; i++) {
# 		setBatchMode(true);
		
# 		filename = file_list[i];
# 		image_name = substring(filename, 0, indexOf(filename, "."));

# 		open(input_path+filename);
# 		run("Set Scale...", "distance=1 known=1 global");
# 		run("Gaussian Blur...", "sigma=sigma_value");
# 		setAutoThreshold("Default dark");
# 		setThreshold(thresh, 64880);
# 		setOption("BlackBackground", true);
# 		run("Make Binary");
# 		run("Watershed");
# 		saveAs("Tiff", output_path+image_name+"_watershed.tif");
# 		close();
# }

# run("Quit");
# run("Quit");
    
    
    # SMALL blur to smoothen micronuclei for more consistent particle detection
    # sig = 1.0 causes ~3% oversegmentation in control but accurate segmentation in drug treatment
    sig = 1.0
    
    micro_script = open('micronuc_watershed_script.ijm','r')
    micro_script = micro_script.read()

    # make sure to export segmented images into original image folder
    micro_script = micro_script.replace('<input path>', '"'+fiji_input_path+'"').replace('<output path>', 
                '"'+image_dir+'"').replace('<thresh>', str(thresh)).replace('<sigma_value>', str(sig))
    
    # writing the new comet script and replacing template
    micro_script_temp = open('micronuc_watershed_script_temp.ijm', 'w')

    micro_script_temp.write(micro_script)
    micro_script_temp.close()

    # the actual comet segmentation
    os.system('/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx -macro /Users/javier/Desktop/Mitchison_Lab/Coding_Stuff/micronuc_watershed_script_temp.ijm')

    # deleting the temporary ijm files
    os.remove('micronuc_watershed_script_temp.ijm')
    
    # now delete unnecessary FIJI input directory
    shutil.rmtree(fiji_input_path, ignore_errors=True)
    


# function that detects particles of all sizes in an image
# outputs areas of particles and, if desired, mean intensities
def detect_particles(image_path, min_area=10, max_area=10000, circ_filter=0.0):
    
    # read in image
    water_image = skimage.io.imread(image_path, plugin='tifffile')
    
    areas = []   # initializing
    intens = []
        
    label_image = skimage.measure.label(water_image)
    for region in skimage.measure.regionprops(label_image, intensity_image=water_image):
        
        # skip areas smaller than 3 pixels (likely noise)
        if region.area < min_area:
            continue
            
        # assuming all background is set to zero
        if region.mean_intensity < 1:
            continue
            
        # getting length and width, and circularity of particle
        if circ_filter > 0.0:
            major_ax = region.major_axis_length * 1.0
            minor_ax = region.minor_axis_length * 1.0
            circularity = minor_ax / major_ax

            if circularity <= circ_filter:
                continue
            
        # getting area of particle
        areas.append(region.area * 1.0)  # record
        
    return np.array(areas)


# will give total area across all fields in a well
# 'frag_ch': 'fragment' channel designator that appears in all image filenames; use if multiple channels present
# 'area_ch': channel used to quantify cell areas; leave as area_ch=[] if no normalization desired
def score_particles(image_dir, well_info, frag_ch='watershed', area_ch='DAPI).tif', 
                    min_area=10, max_area=10000, bin2x2=True):
    
    if image_dir[-1] != '/':
        image_dir += '/'
    
    # getting all relevant image filenames; split into channel images
    all_images = [f for f in os.listdir(image_dir) if ('.tif' in f.lower()) and (well_info in f)]
    ch_images = [f for f in all_images if frag_ch.lower() in f.lower()]
    
    # counting particles in all fields and summing together
    particle_areas = []
    for im in ch_images:
        
        # get particle areas
        areas = detect_particles(image_dir+im, min_area, max_area)
        particle_areas.extend(areas)
        
    if not isinstance(area_ch, list):   
        # need to normalize particle count by total cell area in a well
        well_cell_area = score_area(image_dir, well_info, image_ch=area_ch, bin2x2=bin2x2)

        # normalized particle count
        norm_part_count = len(particle_areas)*1.0 / well_cell_area

        # return median area and number of particles
        return np.median(particle_areas), norm_part_count
    
    else:
        return np.median(particle_areas), len(particle_areas)*1.0

    
    
##########################################################################################


### code to score cell area

# 'image_path': path to image, or alternatively, the image array
def count_area(image_path, bin2x2=True, sig=12.5):
    
    # setting gaussian blur parameters and area cut-offs        
    if not bin2x2:
        min_area = 200
    else:
        min_area = 10
        
    # read in image
    orig_image = skimage.io.imread(image_path, plugin='tifffile')
        
    # blurring image and subtracting interphase cells
    gauss = ndimage.gaussian_filter(orig_image, sigma=sig, order=0)
    
    thresh = threshold_otsu(gauss)  # thresholding
    gauss[gauss < thresh] = 0       # binarizing
    gauss[gauss >= thresh] = 1
    
    cell_areas = []  # will filter by size in the end, just in case; also use this variable as a counter
    label_image = skimage.measure.label(gauss)
    for region in skimage.measure.regionprops(label_image, intensity_image=gauss):

        # skip areas smaller than 50 pixels (likely noise)
        # don't set max cut-off
        if region.area < min_area:
            continue

        cell_areas.append(region.area)  # record area

    # sum up and output
    return np.sum(cell_areas)
    
    
# will give total area across all fields in a well
# specify image channel if more than one channel present
def score_area(image_dir, well_info, image_ch=[], bin2x2=True, sig=12.5):
    
    if image_dir[-1] != '/':
        image_dir += '/'
    
    # getting all relevant image filenames; split into channel and normal images
    ch_images = [f for f in os.listdir(image_dir) if ('.tif' in f.lower()) and (well_info in f)]
    
    # if channel not inputted
    if not isinstance(image_ch, list):
        ch_images = [f for f in ch_images if image_ch.lower() in f.lower()]
    
    # counting total cell areas in all fields and summing together
    tot_cell_area = 0
    for im in ch_images:
        tot_cell_area += count_area(image_dir+im, bin2x2=bin2x2, sig=sig)
    
    return tot_cell_area


##########################################################################################


# 'method' can be 'mitosis', 'life', 'nuclear size', 'nuclear count', 'doubling'
# can use for mitosis data or cell-titer-glo/cell area data
def plate_biorep(path_list, method='mitosis', data_col=[]):
    
    if isinstance(data_col, list):
        if method.lower() == 'mitosis':
            data_col = 'Mitotic_Index_'
        elif method.lower() == 'nuclear size':
            data_col = 'Nuclear_Size_'
        elif method.lower() == 'nuclear count':
            data_col = 'Nuclear_Count_'
        elif method.lower() == 'doubling':
            data_col = 'Double_Freq_'
        elif method.lower() == 'area':
            data_col = 'Cell_Area_'
        elif method.lower() == 'area ratio':
            data_col = 'Cell_Area_Ratio_'
        else:
            data_col = 'Life_Index_'
    
    
    df_list = [pd.read_csv(path) for path in path_list]
    
    # get unique drug conditions that are present in all biorepeats
    # use 'Drug' and 'Drug_Conc_nM' columns to be safe
    # verify that drug concentration column has floats
    # make sure drug concentrations are the same type
    for idx, df in enumerate(df_list):
        df['Drug_Conc_nM'] = df.Drug_Conc_nM.apply(lambda x: float(x) if '%' not in x else x)
        df_list[idx] = df
        
    # get unique conditions
    unq_cond = df_list[0][['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1).tolist()
    for df in df_list:
        unq_cond_i = df[['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1).tolist()     
        unq_cond = set(unq_cond_i).intersection(unq_cond)  # get intersection
        
    
    df_norm_list = []
    for df in df_list:
        
        # only take those conditions that are present in all the datasets
        df['Key_Col'] = df[['Drug', 'Drug_Conc_nM']].apply(lambda x: x[0]+str(x[1]), axis=1)
        df = df[df.Key_Col.isin(unq_cond)].reset_index(drop=True)
        
        # sort appropriate columns
        df = df.sort(['Drug', 'Drug_Conc_nM']).reset_index(drop=True)

        if method == 'life':
        # will normalize by negative control intensity (i.e. DMSO)        
            mean_val = df[data_col+'Mean'][df.Drug == 'DMSO'].values
            df[data_col+'Mean_Norm'] = df[data_col+'Mean'] / mean_val
            df[data_col+'SEM_Norm'] = df[data_col+'SEM'] / mean_val

        df_norm_list.append(df)
    
    biorep_df =  df_norm_list[0].copy()
    biorep_df[data_col+'Mean'] = np.mean([df[data_col+'Mean'] for df in df_norm_list], 0)
    biorep_df[data_col+'SEM'] = np.sqrt(np.sum([df[data_col+'SEM']**2 for df in df_norm_list], 0)) / len(path_list)
    
    if method.lower() == 'life':
        biorep_df[data_col+'Mean_Norm'] = np.mean([df[data_col+'Mean_Norm'] for df in df_norm_list], 0)
        biorep_df[data_col+'SEM_Norm'] = np.sqrt(np.sum([df[data_col+'SEM_Norm']**2 for df in df_norm_list], 0)) / len(path_list)
    
    # drop unnecessary column
    biorep_df = biorep_df.drop('Rep_Wells', 1).drop('Key_Col', 1)
    
    return biorep_df


##########################################################################################


# making simpler procedure to plot plate data
# if time-course not desired, input data subset for the desired time-point
def plot_plate_data(data_df, y_col, y_err, x_col='Drug_Conc_nM', 
                    label_col='Drug', ylabel=[], xlim=[], ylim=[], 
                    grid_on=False, fill=False, fc=False, scale=False, dmso_norm=True, legend_loc='upper center', 
                    show_xticks=True):
    
    
    # get unique conditions for labeling purposes
    labels = list(data_df[label_col].unique())
        
    color_list = [[0, 0, 0], 
                  [1.0, 0.5, 0], 
                  [0, 0.6, 0], 
                  [0, 0, 0.9], 
                  [0.6, 0.6, 0.6], 
                  [0, 0.9, 0.9], 
                  [0.75, 0, 0.9], 
                  [0.9, 0.9, 0], 
                  [0, 0.4, 0], 
                  [0, 0, 0.4], 
                  [0.4, 0, 0], 
                  [0.4, 0, 0.4], 
                  [0, 0.4, 0.4], 
                  [0.5, 0.25, 0]]
    
    # set the figure
    fig = plt.figure()
    ax1 = fig.add_axes((0.1,0.3,0.8,0.8))
    
    # get control data
    orig_dmso_val = -1000  # initialize
    if dmso_norm:
        try:
            orig_dmso_val = data_df[data_df.Drug == 'DMSO'][y_col].tolist()[0]
            orig_dmso_sem = data_df[data_df.Drug == 'DMSO'][y_err].tolist()[0]
            
            data_df[y_col] = data_df[y_col] / orig_dmso_val
            data_df[y_err] = data_df[y_err] / orig_dmso_val

            dmso_val = 1.0
            dmso_sem = orig_dmso_sem / orig_dmso_val
            
            fc = False
                        
        except:
            pass
        
    if scale:
        
        # if user wants to scale data
        # get data to plot
        y_array = np.array(data_df[y_col].tolist())
        y_sem_array = np.array(data_df[y_err].tolist())

        # re-scale data to maximum and minimum
        # NOTE: order of code matters here!!!
        y_sem_min = y_sem_array[np.argsort(y_array)[0]]
        y_min = y_array.min() - y_sem_min
        y_array_minus = y_array-y_min

        y_sem_max = y_sem_array[np.argsort(y_array)[-1]]
        y_max = y_array_minus.max() + y_sem_max
        y_array = y_array_minus / y_max
        y_sem_array = y_sem_array / y_max

        # no values should be 0
        y_array[y_array < 0] = 0

        # record scaled values
        data_df[y_col] = y_array
        data_df[y_err] = y_sem_array
        
        dmso_val = data_df[data_df.Drug == 'DMSO'][y_col].tolist()[0]
        dmso_sem = data_df[data_df.Drug == 'DMSO'][y_err].tolist()[0]
        
    xticks = []  #intializing
    # iterate through unique conditions and plot
    for idx, label in enumerate(labels):
        
        # plot control data later
        if 'dmso' in str(label).lower():
            continue
        
        drug_df = data_df[data_df[label_col] == label].reset_index(drop=True)
        
        # data to plot
        x_data = np.array(drug_df[x_col].tolist()).astype(float)
        y_data = np.array(drug_df[y_col].tolist())
        y_sem = np.array(drug_df[y_err].tolist())

        # sort data just in case
        y_data = y_data[np.argsort(x_data)]
        y_sem = y_sem[np.argsort(x_data)]
        
        if 'site_occ' in x_col.lower():
            x_sem = np.array(drug_df[x_col+'_SEM']).astype(float)
            x_sem = x_sem[np.argsort(x_data)]
        x_data = np.sort(x_data)
        
        if fc:
            y_sem = y_sem / y_data[0]
            y_data = y_data / y_data[0]
        
        # y data should be log-transformed if dose information
        if 'conc' in x_col.lower():
            x_data = np.log10(x_data)
            
            # in casesome drugs have more conditions than others
            if len(x_data) > len(xticks):
                xticks = x_data

        # plot data
        if not fill:
            if label[-2:] == '.0':
                label = label[:-2]
            
            if 'site_occ' not in x_col.lower():
                ax1.errorbar(x_data, y_data, yerr=y_sem, fmt='-o', color=color_list[0], 
                         markeredgecolor='none', elinewidth=1.5, label=label)
            else:
                ax1.errorbar(x_data, y_data, yerr=y_sem, xerr=x_sem, fmt='-o', color=color_list[0], 
                         markeredgecolor='none', elinewidth=1.5, label=label)
            
        else:
            ax1.plot(x_data, y_data, '-', color=color_list[0], markeredgecolor='none', label='')
            ax1.fill_between(x_data, y_data-y_sem, y_data+y_sem, color=color_list[0], 
                             alpha=0.2, label=label)
        color_list.pop(0)
        
    # plot control data
    try:
        if not dmso_norm:
            dmso_val = data_df[data_df.Drug == 'DMSO'][y_col].tolist()[0]
            dmso_sem = data_df[data_df.Drug == 'DMSO'][y_err].tolist()[0]

        # plotting control value across all doses
        if 'site_occ' in x_col.lower():
            x_data = [0, 1]
        ax1.fill_between(x_data, dmso_val-dmso_sem, dmso_val+dmso_sem, 
                         color=(0, 0, 0), alpha=0.2, label='DMSO')
    except:
        pass
    
    # plot with log transform, but show non-log-transformed concentrations
    if 'conc' in x_col.lower():
        ax1.set_xticks(xticks[::2])
        
        xtick_labels = 10.0**(xticks[::2])
        xtick_labels = [str(i)[:-2] if str(i)[-2:] == '.0' in str(i) else str(i) for i in xtick_labels]
        
        ax1.set_xticklabels(xtick_labels)
        
    if 'site_occ' in x_col.lower():
        
        xtick_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xticks = xtick_labels
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xtick_labels)
        
    # set legend and default parameters
    set_plot_params(fig=fig, ax=ax1, poster=1)
    remove_border(bottom=1, left=1)
    
    if legend_loc.lower() != 'none':
        ax1.legend(loc=legend_loc, fancybox=1, framealpha=0.5, fontsize=22, numpoints=1)
    else:
        ax1.legend().set_visible(False)
    
    if len(xlim) > 0:    # modify x-axis limits
        ax1.set_xlim(xlim)
    if len(ylim) > 0:    # modify y-axis limits
        ax1.set_ylim(ylim)

    # include grid lines, or not
    if grid_on:
        plt.grid()

    # set axis labels
    if isinstance(ylabel, list):
        ax1.set_ylabel('Normalized Intensity') #25
    else:
        ax1.set_ylabel(ylabel)

    if 'time_min' in x_col.lower():
        ax1.set_xlabel('Time (min)')
    elif 'time_hr' in x_col.lower():
        ax1.set_xlabel('Time (hr)')
    elif 'conc' in x_col.lower():
        ax1.set_xlabel('Dose (nM)')
    elif 'site_occ' in x_col.lower():
        ax1.set_xlabel('Site Occupancy')
        
        
    if not show_xticks:
        ax1.set_xticks([])
        ax1.set_xlabel('')
    
    ax1.xaxis.label.set_fontsize(30)
    ax1.yaxis.label.set_fontsize(30)        
    ax1.yaxis.tick_left()
    
    plt.show() 
    

##########################################################################################


### code for plotting and visualizing site occupancy results

# function that plots multiple columns from single dataframe
# this only plots with site_occ as the x-axis
# if not antilog, will use linear
def site_plot(agg_data, ycols, ysemcols, labels, antilog=True):
    
    
    ### input data
    
    if not isinstance(ycols, list):
        ycols = [ycols]
        
    if not isinstance(ysemcols, list):
        ysemcols = [ysemcols]
    
    # verify all concentrations are floats
    agg_data['Drug_Conc_nM'] = agg_data.Drug_Conc_nM.apply(lambda x: 
                                            float(x) if '%' not in str(x) else x)
    
    # get all drugs not including DMSO
    unq_drugs = agg_data.Drug.unique().tolist()
    unq_drugs = [d for d in unq_drugs if 'DMSO' not in d]
        
        
    ### plotting section
     
    # setting colors: magenta, cyan, orange, gray
    colors = [[0.75, 0, 0.9], 
              [0.0, 0.75, 0.75],
              [1.0, 0.5, 0], 
              [0.6, 0.6, 0.6]]
    
    for drug in unq_drugs:

        fig = plt.figure()
        ax1 = fig.add_axes((0.1,0.3,0.8,0.8))
       
        drug_conc = np.array(agg_data.Drug_Conc_nM[agg_data.Drug == drug].tolist())
    
        # getting data for drug (antilog to space out data)
        if antilog:
            site_occ = 10.0**np.array(agg_data.Site_Occ[agg_data.Drug == drug].tolist())
            site_sem = np.log(10.0)*site_occ*np.array(agg_data.Site_Occ_SEM[agg_data.Drug == drug].tolist())
            
        else:
            site_occ = np.array(agg_data.Site_Occ[agg_data.Drug == drug].tolist())
            site_sem = np.array(agg_data.Site_Occ_SEM[agg_data.Drug == drug].tolist())
      
        # must sort values for plotting to make sense
        sort_idx = np.argsort(site_occ)
        site_occ = site_occ[sort_idx]
        site_sem = site_sem[sort_idx]
        drug_conc = drug_conc[sort_idx]
        
        # iterate through phenotypic data types
        for j, col in enumerate(ycols):
            
            # get data to plot
            y_array = np.array(agg_data[col][agg_data.Drug == drug].tolist())
            y_sem_array = np.array(agg_data[ysemcols[j]][agg_data.Drug == drug].tolist())
            
            # re-scale data to maximum and minimum
            # NOTE: order of code matters here!!!
            y_sem_min = y_sem_array[np.argsort(y_array)[0]]
            y_min = y_array.min() - y_sem_min
            y_array_minus = y_array-y_min
            
            y_sem_max = y_sem_array[np.argsort(y_array)[-1]]
            y_max = y_array_minus.max() + y_sem_max
            y_array = y_array_minus / y_max
            y_sem_array = y_sem_array / y_max
            
            # must sort values for plotting to make sense
            y_array = y_array[sort_idx]
            y_sem_array = y_sem_array[sort_idx]
            
            # no values should be 0
            y_array[y_array < 0] = 0
            
            
            # fill in vertical bars for x error
            if j == 0:
                for idx in range(len(site_occ)):
                    ax1.axvspan(site_occ[idx]-site_sem[idx], site_occ[idx]+site_sem[idx], 
                               alpha=0.1, color=[0.6, 0.6, 0.6])

            # plot Y values with y error filled in
            plt.fill_between(site_occ, y_array-y_sem_array, y_array+y_sem_array, 
                            color=colors[j], alpha=0.5, label=labels[j])
        
        # set y-axis limits
        ax1.set_ylim([-0.025, 1.1])
        
        # set tick labels
        xtick_labels1 = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        if antilog:
            ax1.set_xlim([0.9, 10.2])
            xticks1 = 10.0**np.array(xtick_labels1)
        else:
            ax1.set_xlim([-0.01, 1.01])
            xticks1 = xtick_labels1
            
        ax1.set_ylim([-0.01, 1.31])
            
        ax1.set_xticks(xticks1)
        ax1.set_xticklabels(xtick_labels1)
        
        
        ## hard-coded
#         ax1.plot([0.675, 0.675], [-0.025, 1.31], '--', color='k', label='Therapeutic Dose')
        ##
    
        # label axes
        print drug
        ax1.set_xlabel('Site Occupancy')
        ax1.set_ylabel('Rescaled FC')
        
        set_plot_params(ax=ax1, poster=True)
        yes_border_no_ticks(axes=ax1)
        ax1.legend(loc='upper left', prop={'size':22}, fancybox=True, framealpha=0.5, numpoints=1)

        
        # plot second axis for drug concentration
        ax2 = fig.add_axes((0.1,0.1,0.8,0.0))
        ax2.yaxis.set_visible(False) # hide the yaxis
        
        if antilog:
            ax2.set_xlim([0.9, 10.2])
        else:
            ax2.set_xlim([-0.01, 1.01])
        
        xticks2 = site_occ
        round_conc = np.array([round_sig(c, 1) for c in drug_conc])
        xtick_labels2 = np.array([str(int(c)) if c >= 1 else str(c) for c in round_conc])
        xtick_labels2[[1, 2, 3, 4, 5, 9, 10, 11]] = ''
        
        ax2.set_xticks(xticks2)
        ax2.set_xticklabels(xtick_labels2)
        ax2.set_xlabel('Dose (nM)')
        ax2.set_xlim()
        set_plot_params(ax=ax2, poster=True)
        plt.show()


# 't_crit_vals': critical values for t test, in decreasing order
# e.g. [3.686, 2.921, 2.583, 1.746] for pval < [0.001, 0.005, 0.01, 0.05] and 16 degrees of freedom
def plot_max(agg_data, ycols, ysemcols, labels, t_crit_vals=[], ylim=[], scale=True, label_sig=True):
    
    if not isinstance(t_crit_vals, list):
        t_crit_vals = [t_crit_vals]
    
    
    ### input data
    
    if not isinstance(ycols, list):
        ycols = [ycols]
        ysemcols = [ysemcols]
        labels = [labels]
    
    # verify all concentrations are floats
    agg_data['Drug_Conc_nM'] = agg_data.Drug_Conc_nM.apply(lambda x: 
                                            float(x) if '%' not in str(x) else x)
    
    # colors
    colors = [[0, 0, 0], 
              [1.0, 0.5, 0], 
              [0, 0.6, 0], 
              [0, 0, 0.9]]
    
    # get unique drugs not including DMSO
    unq_drugs =  agg_data.Drug.unique().tolist()
    unq_drugs = [d for d in unq_drugs if 'DMSO' not in d]
    
    
    if scale:
        for i, y_col in enumerate(ycols):
            
            y_err = ysemcols[i]  # corresponding SEM column
            
            # if user wants to scale data
            # get data to plot
            y_array = np.array(agg_data[y_col].tolist())
            y_sem_array = np.array(agg_data[y_err].tolist())

            # re-scale data to maximum and minimum
            # NOTE: order of code matters here!!!
            y_sem_min = y_sem_array[np.argsort(y_array)[0]]
            y_min = y_array.min() - y_sem_min
            y_array_minus = y_array-y_min

            y_sem_max = y_sem_array[np.argsort(y_array)[-1]]
            y_max = y_array_minus.max() + y_sem_max
            y_array = y_array_minus / y_max
            y_sem_array = y_sem_array / y_max

            # no values should be 0
            y_array[y_array < 0] = 0

            # record scaled values
            agg_data[y_col] = y_array
            agg_data[y_err] = y_sem_array
        
    
    
    # iterate through phenotypes
    for i, col in enumerate(ycols):
  
        fig = plt.figure(figsize = (5,5))
        ax1 = plt.gca()
        
        # iterate through drugs
        max_vals = []
        sem_vals = []
        abs_max = 0
        for drug in unq_drugs:
            
            # get maximum value for each drug
            y_array = agg_data[col][agg_data.Drug == drug].values
            y_sem_array = agg_data[ysemcols[i]][agg_data.Drug == drug].values
            
            y_max = y_array.max()
            y_sem = y_sem_array[np.argsort(y_array)[-1]]
            
            if y_max+y_sem > abs_max:
                abs_max = y_max+y_sem
            
            max_vals.append(y_max)
            sem_vals.append(y_sem)
        
        ax1.bar(np.arange(4), max_vals, color=colors, edgecolor='k', width=0.75, linewidth=2.0)
        plt.errorbar(np.arange(4)+0.375, max_vals, yerr=sem_vals, color='k', 
                     fmt=' k', capsize=10.0, markeredgewidth=1.0, linewidth=1.0)
        
        plt.xlim([-0.5, 4.25])
        
        xticks = np.arange(4)+0.375
        plt.xticks(xticks)
        plt.gca().set_xticklabels(unq_drugs, rotation=0.0)
        
        yticks = ax1.get_yticks()
        non_integer = np.sum(['.5' in str(t) for t in yticks])
        
        plt.ylabel(labels[i])
        set_plot_params()
        
        if len(ylim) > 0:
            plt.ylim(ylim)
        
        plt.show()


# all arguments are lists
# assumes aggregated data has exact same drugs and concentrations
def drug_sensitivity(agg_data_pair, sens_cols, sem_cols, axis_labels):
    
    ### input data
    
    if not isinstance(sens_cols, list):
        sens_cols = [sens_cols]
        
    if not isinstance(sem_cols, list):
        sem_cols = [sem_cols]
    
    # get all drugs not including DMSO
    unq_drugs = agg_data_pair[0].Drug.unique().tolist()
    unq_drugs = [d for d in unq_drugs if 'DMSO' not in d]
    
    agg_data_pair[0] = agg_data_pair[0].sort(['Drug', 'Drug_Conc_nM']).reset_index(drop=True)
    agg_data_pair[1] = agg_data_pair[1].sort(['Drug', 'Drug_Conc_nM']).reset_index(drop=True)
     
        
    ### plotting section
     
    # setting colors: magenta, cyan, orange, gray
    colors = [[0, 0, 0], 
              [1.0, 0.5, 0], 
              [0, 0.6, 0], 
              [0, 0, 0.9]]
    
    # re-scale phenotypic data
    for agg_data in agg_data_pair:
        for i, col in enumerate(sens_cols):
            y_array = np.array(agg_data[col].tolist())
            y_sem_array = np.array(agg_data[sem_cols[i]].tolist())

            # re-scale data to maximum and minimum
            # NOTE: order of code matters here!!!
            y_sem_min = y_sem_array[np.argsort(y_array)[0]]
            y_min = y_array.min() - y_sem_min
            y_array_minus = y_array-y_min

            y_sem_max = y_sem_array[np.argsort(y_array)[-1]]
            y_max = y_array_minus.max() + y_sem_max
            y_array = y_array_minus / y_max
            y_sem_array = y_sem_array / y_max

            y_array[y_array < 0] = 0             # no values should be 0
            agg_data[col] = y_array              # recording rescaled data
            agg_data[sem_cols[i]] = y_sem_array
            
    agg_data_pair1 = agg_data_pair[0]
    agg_data_pair2 = agg_data_pair[1]
    
    
    
         
    for i, col in enumerate(sens_cols):
        
        
        fig, axes_tup = plt.subplots(figsize=(7, 7), nrows=2, ncols=2)
        
        # turning the nested axes tuple into one list
        axes = sum([list(a) for a in list(axes_tup)], [])
        
        for j, drug in enumerate(unq_drugs):
            
       
            # plotting the two datasets
            site_occ1 = np.array(agg_data_pair1.Site_Occ[agg_data_pair1.Drug == drug].tolist())
            site_occ1_error = np.array(agg_data_pair1.Site_Occ_SEM[agg_data_pair1.Drug == drug].tolist())
            
            # must sort values for plotting to make sense
            sort_idx1 = np.argsort(site_occ1)
            site_occ1 = site_occ1[sort_idx1]
            site_occ1_error = site_occ1_error[sort_idx1]
            y1_data = np.array(agg_data_pair1[col][agg_data_pair1.Drug == 
                                                   drug].tolist())[sort_idx1]
            y1_error = np.array(agg_data_pair1[sem_cols[i]][agg_data_pair1.Drug == 
                                                   drug].tolist())#[sort_idx1]
            
            # re-scale data to maximum and minimum
            # NOTE: order of code matters here!!!
            y1_error_min = y1_error[np.argsort(y1_data)[0]]
            y1_data_min = y1_data.min() - y1_error_min
            y1_data_minus = y1_data-y1_data_min

            y1_error_max = y1_error[np.argsort(y1_data)[-1]]
            y1_data_max = y1_data_minus.max() + y1_error_max
            y1_data = y1_data_minus / y1_data_max
            y1_error = y1_error / y1_data_max
            y1_data[y1_data < 0] = 0             # no values should be 0
            
            site_occ2 = np.array(agg_data_pair2.Site_Occ[agg_data_pair2.Drug == drug].tolist())
            site_occ2_error = np.array(agg_data_pair2.Site_Occ_SEM[agg_data_pair2.Drug == drug].tolist())
            
            # must sort values for plotting to make sense
            sort_idx2 = np.argsort(site_occ2)
            site_occ2 = site_occ2[sort_idx2]
            site_occ2_error = site_occ2_error[sort_idx2]
            
            
            y2_data = np.array(agg_data_pair2[col][agg_data_pair2.Drug == 
                                                   drug].tolist())[sort_idx2]
            y2_error = np.array(agg_data_pair2[sem_cols[i]][agg_data_pair2.Drug == 
                                                   drug].tolist())[sort_idx2]
            
            # re-scale data to maximum and minimum
            # NOTE: order of code matters here!!!
            y2_error_min = y2_error[np.argsort(y2_data)[0]]
            y2_data_min = y2_data.min() - y2_error_min
            y2_data_minus = y2_data-y2_data_min

            y2_error_max = y2_error[np.argsort(y2_data)[-1]]
            y2_data_max = y2_data_minus.max() + y2_error_max
            y2_data = y2_data_minus / y2_data_max
            y2_error = y2_error / y2_data_max
            y2_data[y2_data < 0] = 0             # no values should be 0


            axes[j].errorbar(site_occ1, y1_data, yerr=y1_error, xerr=site_occ1_error, color=colors[j], fmt='-')
            axes[j].errorbar(site_occ2, y2_data, yerr=y2_error, xerr=site_occ2_error, color=colors[j], fmt='--')
            
            axes[j].set_ylim([-0.025, 1.1])
            axes[j].set_xlim([-0.025, 1.025])

            # take off unnecessary ticks to avoid clutter
            if j in [0, 1]:
                axes[j].set_xticks([])
            if j in [1, 3]:
                axes[j].set_yticks([])
                
            set_plot_params(fig=fig, ax=axes[j], poster=True)
            yes_border_no_ticks(axes[j])

            axes[j].tick_params(axis='both', which='major', labelsize=15)
            
        
        # making bix axis to center x- and y-axis labels
        pos0 = axes[0].get_position()
        pos1 = axes[1].get_position()
        pos2 = axes[2].get_position()
        pos3 = axes[3].get_position()
        
        big_pos = [pos2.x0, pos2.y0, pos3.x0+pos3.width-pos2.x0, pos0.y0+pos0.height-pos3.y0]

        big_ax = fig.add_axes(big_pos)
        big_ax.patch.set_visible(False)
        big_ax.set_xlabel('\n\nSite Occupancy')
        big_ax.set_ylabel(axis_labels[i]+'\n\n')
        big_ax.xaxis.label.set_fontsize(20)
        big_ax.yaxis.label.set_fontsize(20)
        remove_border(axes=big_ax, left=0, bottom=0)
        
        big_ax.set_xticks([])
        big_ax.set_yticks([])
            
        plt.show()


# a function that takes a value, finds the closest value in an independent variable array, 
# and gives a corresponding value from a dependent variable array
# useful for when we know a relationship between two variables but do not have 
# all the other variables necessary to use an equation to calculate corresponding values
# 'ind_array': independent variable array
# 'dep_array': dependent variable array
# 'dep_sem_array': dependent variable standard error array
def fn_dict(value, ind_array, dep_array, dep_sem_array=[]):
    
    # if an NaN value has been inputted, then output NaNs as well
    if np.isnan(value):
        if len(dep_sem_array) > 0:
            return np.nan, np.nan
        else:
            return np.nan
    
    if isinstance(ind_array, list) | isinstance(dep_array, list):
        ind_array = np.array(ind_array)
        dep_array = np.array(dep_array)
        dep_sem_array = np.array(dep_sem_array)
    
    # getting the differences between independent variable values and the inputted value
    diff_array = np.abs(ind_array - value)
    
    # find where the minimum difference is
    best_match = np.argmin(diff_array)
    
    # output the corresponding dependent variable values
    if len(dep_sem_array) > 0:
        return float(dep_array[best_match]), float(dep_sem_array[best_match])
    else:
        return float(dep_array[best_match])
    

# function to simulate a variable in the equation for SirT competition experiments
# input a range for whichever variable to simulate (one at a time), i.e. ctrl_pmt, p_out, kdp_app, or mt_fc
# 'equil_drug': dataframe with data for one drug at a user-defined equilibrium time-point
# 'vivo_data': dataframe with in vivo SirT (FC) values for which we need site occ measurements
# 'model_p_out': if site occupancy desired, input lower and upper bound for probe
def sim_site_occ(vivo_data, vivo_cols, kdd_app_tup, k_factor_tup, kdp_app_tup, mt_fc_tup,
                 pmt_o_800=1.0, model_p_out=[10.0, 1000.0]):

    # getting simulated parameter values
    kdd_app = kdd_app_tup[0]
    kdd_app_err = kdd_app_tup[1]

    k = k_factor_tup[0]
    k_err = k_factor_tup[1]

    kdp_app = kdp_app_tup[0]
    kdp_app_err = kdp_app_tup[1]

    mt_fc = mt_fc_tup[0]
    mt_fc_err = mt_fc_tup[1]
    
    # calculate initial MT at 800nM and also max MT
    mt_o_800 = pmt_o_800*(1.0 + kdp_app/800.0)
    mt_o_800_err = kdp_app_err/800.0*pmt_o_800

    mt_max = mt_o_800*mt_fc
    mt_max_err = mt_o_800*mt_fc*np.sqrt((mt_o_800_err/mt_o_800)**2.0+(mt_fc_err/mt_fc)**2.0)


    ### error propagation for site occupancy

    # theoretical drug concentrations
    sim_d_conc = 10.0**np.linspace(np.log10(0.001), 
                                   np.log10(10000), 1000)

    # assuming probe concentration
    # treat as uniformly distributed variable from 10nM - 1uM
    p_out = 0.5*np.sum(model_p_out)
    p_out_err = (model_p_out[1]-model_p_out[0])/np.sqrt(12.0)

    # assuming initial MT concentration hasn't changed at all without drug
    mt_o = mt_max/2.0
    mt_o_err = mt_max_err/2.0

    # computing simulated MT 2-fold-change
    sim_mt_tot =  mt_o*np.exp(-k*sim_d_conc) + mt_max*(1.0 - np.exp(-k*sim_d_conc))     
    sim_mt_fc = sim_mt_tot / mt_o

    sim_mt_err_term1 = mt_o*np.exp(-k*sim_d_conc)*np.sqrt((mt_o_err/mt_o)**2.0+(sim_d_conc*k_err)**2.0)
    sim_mt_err_term2 = mt_max*(1.0-np.exp(-k*sim_d_conc))*np.sqrt((mt_max_err/mt_max)**2.0+(np.exp(-k*sim_d_conc)*sim_d_conc*k_err/(1.0-np.exp(-k*sim_d_conc)))**2.0)
    sim_mt_err = np.sqrt(sim_mt_err_term1**2.0+sim_mt_err_term2**2.0)
    sim_mt_fc_err = sim_mt_err / mt_o
    
    # compute simulated PMT fold-change 
    # realize that we are assuming that we know the values of PMT fold-change
    # i.e. PMT fold-change is our independent variable now
    pmt_denom = kdp_app/p_out*sim_d_conc/kdd_app + kdp_app/p_out + 1.0        
    pmt_denom_o = kdp_app/p_out + 1.0
    pmt_denom_fc = pmt_denom / pmt_denom_o   
    sim_pmt_fc = sim_mt_fc / pmt_denom_fc                                   
    sim_pmt_fc_err = 0.0

    # new equation for site occupancy
    site_occ = (sim_mt_fc - sim_pmt_fc)/(sim_mt_fc - sim_pmt_fc/(kdp_app/p_out+1.0))

    # error of kdp_app/p_out+1.0
    E_error = kdp_app/p_out*np.sqrt((kdp_app_err/kdp_app)**2.0+(p_out_err/p_out)**2.0)

    # error of bottom right term
    D_error = sim_pmt_fc/(kdp_app/p_out+1.0)*np.sqrt((sim_pmt_fc_err/sim_pmt_fc)**2.0+(E_error/(kdp_app/p_out+1.0))**2.0)

    # error of the top
    AB_error = np.sqrt(sim_mt_fc_err**2.0+sim_pmt_fc_err**2.0)

    # error of the bottom
    CD_error = np.sqrt(sim_mt_fc_err**2.0+D_error**2.0)

    # error of site occupancy
    site_occ_err = site_occ*np.sqrt((AB_error/(sim_mt_fc - sim_pmt_fc))**2.0+(CD_error/(sim_mt_fc - sim_pmt_fc/(kdp_app/p_out+1.0)))**2.0)

    
    
    ### determine in vivo site occupancy point estimates
    
    site_occ_cols = []
    site_occ_sem_cols = []
    for col in vivo_cols:
        
        site_occ_cols.append(col+'_Site_Occ')
        site_occ_sem_cols.append(col+'_Site_Occ_SEM')
        
        vivo_data[col+'_Site_Occ'] = vivo_data[col].apply(lambda x: 
                        fn_dict(x, sim_pmt_fc, site_occ, site_occ_err)[0])
        vivo_data[col+'_Site_Occ_SEM'] = vivo_data[col].apply(lambda x: 
                        fn_dict(x, sim_pmt_fc, site_occ, site_occ_err)[1])
    
    
    fig1 = plt.figure()
    ax1 = plt.gca()
    
    medians = make_dot_plot(vivo_data, site_occ_cols, site_occ_sem_cols, 
                  xticks=['Vehicle', '6 mg/kg', '30 mg/kg'], 
                  xlabel='', ylabel='Site Occupancy', xtick_angle=0, 
                  fig=fig1, ax=ax1)
    
    # getting the site occupancy error at the medians
    median_errs = []
    for med in medians:
        median_errs.append(fn_dict(med, site_occ, site_occ_err))
    
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_ylim([-0.075, 1.075])
    plt.show()
    
    fig2 = plt.figure()
    ax2 = plt.gca()
    ax2.fill_between(sim_pmt_fc, site_occ-site_occ_err, site_occ+site_occ_err, 
                color=(0, 0, 0.9), alpha=0.25, label='')
    
    # plot lines for 30 mg/kg site occupancy
    upper_bound = medians[-1] + median_errs[-1]
    lower_bound = medians[-1] - median_errs[-1]
    
    med_pmt = fn_dict(medians[-1], site_occ, sim_pmt_fc)
    
    ax2.plot([med_pmt, med_pmt], [-0.1, upper_bound], '--', color=(0, 0, 0.9), label='')
    ax2.plot([0, med_pmt], [upper_bound, upper_bound], '--', color=(0, 0, 0.9), label='')
    ax2.plot([0, med_pmt], [lower_bound, lower_bound], '--', color=(0, 0, 0.9), label='')
    
    # setting plot parameters
    ax2.set_xlabel('SirTub Signal (FC)')
    ax2.set_ylabel('Site Occupancy')
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_xlim([0, 1.0])
    set_plot_params(ax=ax2, poster=True)
    plt.show()
    
    
##########################################################################################
  
  

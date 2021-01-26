import os
import sys
import time
import pyvips
import argparse
from glob import glob
import openslide
from openslide import open_slide

#Getting arguments
parser = argparse.ArgumentParser()
parser.add_argument('--masks', type=str)
parser.add_argument('--slides', type=str)
args = parser.parse_args()

#Casting arguments
masks = glob(args.masks)
slides = glob(args.slides)

#Start timer
start = time.time()

counter = 0
#Loop over masks
for mask in masks:
    #Extract slide name from mask path in order to match both
    extracted_slide_filename = os.path.basename(os.path.dirname(mask))
    #Loop over slides
    for slide in slides:
        #.svs slidename
        actual_slide_filename = os.path.basename(slide)
        #Find a match between current mask and a slide
        if extracted_slide_filename == actual_slide_filename:
            print(os.path.basename(mask))
            #Get width of original slide to upscale mask to it
            slider = pyvips.Image.new_from_file(slide)
            #Get width of original slide to upscale mask to it
            slide_width = slider.width
            #Get height of original slide to upscale mask to it
            slide_height = slider.height
            #Construct output directory/file
            output = os.path.join(os.path.dirname(mask), os.path.splitext(os.path.basename(mask))[0]+"PD.tif")
            #Save your upscaled mask
            #Thumbnail but upscaling dimensions are not accurate
            image = pyvips.Image.thumbnail(mask, slide_width, height=slide_height).write_to_file(output+"[compression=jpeg,bigtiff,tile,pyramid]")
        else:
            continue

print("%d masks were upscaled" % counter)
end = time.time()
print(f'Execution time {end - start:.2f}s')
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue, Manager
import time
import os
import sys
import argparse
from PIL import Image
import numpy as np
import pandas as pd

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, mask, mask_min_fraction_white, source, tile_size, overlap, bounds,quality, tile_dict):
        Process.__init__(self, name='TileWorker')

        self._queue = queue
        self._mask = mask
        self._mask_min_fraction_white = mask_min_fraction_white
        self._overlap = overlap
        self._bounds = bounds
        self._quality = quality
        self._tile_size = tile_size
        self._source = source
        self._tile_dict = tile_dict
        

    def run(self):
        dz_mask = DeepZoomGenerator(self._mask, self._tile_size, self._overlap, limit_bounds=self._bounds)
        dz_source = DeepZoomGenerator(self._source, self._tile_size, self._overlap, limit_bounds=self._bounds)
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            source_level, mask_level, address, outfile, outfile_bw = data
            if True:
                try:
                    #Get slide tile
                    source_tile = dz_source.get_tile(source_level, address)
                    #Get mask tile
                    mask_tile = dz_mask.get_tile(mask_level, address)
                    gray_mask_tile = mask_tile.convert('L')

                    num_white_pix = np.sum(np.array(gray_mask_tile) == 255)
                    frac_white_pix = num_white_pix / ( gray_mask_tile.width * gray_mask_tile.height )
                    if frac_white_pix >= self._mask_min_fraction_white:
                        source_tile.save(outfile)
                    self._tile_dict[os.path.basename(outfile)] = {'frac_white_pix': frac_white_pix}
                        
                    self._queue.task_done()

                except Exception as e:
                    print(e)
                    self._queue.task_done()


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz_source, dz_mask, source, mask, output, formatting, queue, magnification):
        self._dz_source = dz_source
        self._dz_mask = dz_mask
        self._source = source
        self._mask = mask
        self._output = output
        self._formatting = formatting
        self._queue = queue
        self._processed = 0
        self._count = 0
        self._magnification = magnification

    def run(self):
        self._write_tiles()
       

    def _write_tiles(self):
        #Get DeepZoom assigned levels for slide and mask
        source_level = self._dz_source.level_count-1
        mask_level = self._dz_mask.level_count-1
        self._count = self._count+1

        #A list of downsample factors for each level of the slide.
        downsamples = self._source.level_downsamples
        
        try:
            objective = float(self._source.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        except:
            print("No Obj information found")
        
        #calculate magnifications
        magnif_available = tuple(objective / x for x in downsamples)
        
        ########################################
        #Loop over levels, looking for one that you need and discarding the others
        for level in range(self._dz_source.level_count-1,-1,-1):

            #Get magnifications
            ThisMag = magnif_available[0]/pow(2,self._dz_source.level_count-(level+1))
            
            #Magnification checker
            if self._magnification > 0:
                #If magnifications dont match, just skip
                if ThisMag != self._magnification:
                    continue
            
            tiledir = os.path.join("%s_files" % self._output, str(ThisMag))
            #Make directory
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)

            #Get tile columns and rows for slide and mask, predominantly for debugging
            cols, rows = self._dz_source.level_tiles[level]
            cols1, rows1 = self._dz_mask.level_tiles[level]
            print("Cols Slide:")
            print(cols)
            print("Cols Mask:")
            print(cols1)
            print("Rows Slide:")
            print(rows)
            print("Rows Mask:")
            print(rows1)
            #Loop over rows and columns, populating the queue to tile with TileWorker
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                    col, row, self._formatting))
                    tilename_msk = os.path.join(tiledir, '%d_%d_mask.%s' % (
                                    col, row, self._formatting))

                    if not os.path.exists(tilename):
                        self._queue.put((source_level, mask_level, (col, row),
                                    tilename, tilename_msk))
                    self._tile_done()
        

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz_source.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, mask, mask_min_fraction_white, slide, output, formatting, overlap,
                limit_bounds, quality, workers, tile_size, magnification):
        
        #Open slide
        self._source = open_slide(slide)
        #Open mask
        self._mask = open_slide(mask)
        #Output, will be altered in 
        self._output = output
        
        self._formatting = formatting
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._tile_size = tile_size
        self._magnification = magnification
        #Minimum required fraction of the tile that is white (in mask)
        self._mask_min_fraction_white = mask_min_fraction_white

        #Create an empty dictionary to be accessible across all processes
        manager = Manager()
        tile_dict = manager.dict()
        self._tile_dict = tile_dict
        
        #Print dimensions, in case its needed for debugging
        print(self._source.dimensions)
        print(self._mask.dimensions)

        #Actual tiler
        for _i in range(workers):
            TileWorker(self._queue, self._mask, self._mask_min_fraction_white, self._source, self._tile_size, overlap,
                       limit_bounds, quality, tile_dict).start()

    def run(self):
        #Mask DeepZoomGenerator
        dz_mask = DeepZoomGenerator(self._mask, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)
        #Slide DeepZoomGenerator
        dz_source = DeepZoomGenerator(self._source, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)
        #Start tiling
        source_tiler = DeepZoomImageTiler(dz_source, dz_mask, self._source, self._mask, self._output, self._formatting, self._queue, self._magnification)
        source_tiler.run()
        metadata_tbl = pd.DataFrame.from_dict(dict(self._tile_dict), orient = 'index').rename_axis('tile').reset_index(drop = False)
        
        self._shutdown()

        return metadata_tbl


    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def tiler():
    #Start timer
    start = time.time()

    #Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--magnification', type=float, default=20.0)
    parser.add_argument('--format', type=str, default='jpeg')
    parser.add_argument('--quality', type=str, default=100)
    parser.add_argument('--bounds', type=bool, default=True)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--slides', type=str)
    parser.add_argument('--masks', type=str)
    parser.add_argument('--mask-min-fraction-white', type=float, default=1.0)
    parser.add_argument('--tile-metadata-output-csv', type=str, default=None)
    args = parser.parse_args()

    #Casting arguments
    threads = args.threads 
    tile_size = args.size
    overlap = args.overlap
    magnification = args.magnification
    formatting = args.format
    quality = args.quality
    bounds = args.bounds
    outdir = args.outdir
    slides = glob(args.slides)
    masks = glob(args.masks)
    mask_min_fraction_white = args.mask_min_fraction_white
    tile_metadata_output_csv = args.tile_metadata_output_csv

    Image_mask_dict = {}
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
                #Populate dictionary
                Image_mask_dict[slide] = mask
                counter += 1
            else:
                continue


    #Loop over slide/mask dictionary
    metadata_dict = {}
    for key, value in Image_mask_dict.items():
        slide = key
        mask = value
        #Construct and output directory, will be altered slightly in DeepZoomImageTiler
        output = os.path.join(outdir, os.path.splitext(os.path.basename(slide))[0])
        print(os.path.basename(mask))

        try:
            img_metadata_tbl = DeepZoomStaticTiler(mask, mask_min_fraction_white, slide, output, formatting, overlap, bounds, quality, threads, tile_size, magnification).run()
            if tile_metadata_output_csv is not None:
                metadata_dict[slide] = img_metadata_tbl
        except Exception as e:
            print("Failed to process file %s, error: %s" % (slide, sys.exc_info()[0]))
            print(e)
    if tile_metadata_output_csv is not None:
        for k, v in metadata_dict.items():
            v['image'] = k
        all_metadata_tbl = pd.concat(metadata_dict, ignore_index=True)
        all_metadata_tbl.to_csv(tile_metadata_output_csv, sep=",", index=False)
    print("End")
    end = time.time()
    print(f'Execution time {end - start:.2f}s')

tiler()

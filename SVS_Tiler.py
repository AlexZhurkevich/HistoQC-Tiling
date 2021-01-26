import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys
import argparse
from PIL import Image


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, mask, source, tile_size, overlap, bounds,quality):
        Process.__init__(self, name='TileWorker')

        self._queue = queue
        self._mask = mask
        self._overlap = overlap
        self._bounds = bounds
        self._quality = quality
        self._tile_size = tile_size
        self._source = source
        

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
                    #Get extrema to see how white the mask tile is
                    mask_extrema = mask_tile.convert('L').getextrema()
                    #If you mask tile is white save your slide tile
                    if mask_extrema == (255, 255):
                        source_tile.save(outfile)
                    
                    self._queue.task_done()

                except Exception as e:
                    print(e)
                    self._queue.task_done()


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz_source, dz_mask, output, formatting, queue):
        self._dz_source = dz_source
        self._dz_mask = dz_mask
        self._output = output
        self._formatting = formatting
        self._queue = queue
        self._processed = 0
        self._count = 0
        

    def run(self):
        self._write_tiles()
       

    def _write_tiles(self):
        #Get DeepZoom assigned levels for slide and mask
        source_level = self._dz_source.level_count-1
        mask_level = self._dz_mask.level_count-1
        self._count = self._count+1
        
        ########################################
        tiledir = os.path.join("%s_files" % self._output)
        #Make directory
        if not os.path.exists(tiledir):
            os.makedirs(tiledir)
        #Get tile columns and rows for slide and mask, predominantly for debugging
        cols, rows = self._dz_source.level_tiles[source_level]
        cols1, rows1 = self._dz_mask.level_tiles[mask_level]
        print("Cols Slide:")
        print(cols)
        print("Cols Mask:")
        print(cols1)
        print("Rows Slide:")
        print(rows)
        print("Rows Mask:")
        print(rows1)

        #Loop over rows and columns, populating the queue to tile with TileWorker
        for row in range(rows1):
            for col in range(cols1):
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

    def __init__(self, mask, slide, output, formatting, overlap,
                limit_bounds, quality, workers, tile_size):
        
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
        
        #Print dimensions, in case its needed for debugging
        print(self._source.dimensions)
        print(self._mask.dimensions)

        #Actual tiler
        for _i in range(workers):
            TileWorker(self._queue, self._mask, self._source, self._tile_size, overlap,
                limit_bounds, quality).start()

    def run(self):
        #Mask DeepZoomGenerator
        dz_mask = DeepZoomGenerator(self._mask, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)
        #Slide DeepZoomGenerator
        dz_source = DeepZoomGenerator(self._source, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)
        #Start tiling
        source_tiler = DeepZoomImageTiler(dz_source, dz_mask, self._output, self._formatting, self._queue)
        source_tiler.run()
        self._shutdown()

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
    parser.add_argument('--format', type=str, default='jpeg')
    parser.add_argument('--quality', type=str, default=100)
    parser.add_argument('--bounds', type=bool, default=True)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--slides', type=str)
    parser.add_argument('--masks', type=str)
    args = parser.parse_args()

    #Casting arguments
    threads = args.threads 
    tile_size = args.size
    overlap = args.overlap
    formatting = args.format
    quality = args.quality
    bounds = args.bounds
    outdir = args.outdir
    slides = glob(args.slides)
    masks = glob(args.masks)

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
    for key, value in Image_mask_dict.items():
        slide = key
        mask = value
        #Construct and output directory, will be altered slightly in DeepZoomImageTiler
        output = os.path.join(outdir, os.path.splitext(os.path.basename(slide))[0])
        print(os.path.basename(mask))

        try:
        	DeepZoomStaticTiler(mask, slide, output, formatting, overlap, bounds, quality, threads, tile_size).run()
        except Exception as e:
        	print("Failed to process file %s, error: %s" % (slide, sys.exc_info()[0]))
        	print(e)    
    print("End")
    end = time.time()
    print(f'Execution time {end - start:.2f}s')

tiler()
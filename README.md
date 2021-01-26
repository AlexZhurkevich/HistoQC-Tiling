# HistoQC-Tiling
Tiling support for HistoQC

This set of programs and instructions will perform quality control analysis based on [HistoQC](https://github.com/choosehappy/HistoQC), upscale resulted HistoQC masks to match original slide resolution, tile your slide based on a given upscaled mask with useful regions identified by HistoQC.

1. HistoQC installation:
You can refer to [original](https://github.com/choosehappy/HistoQC) installation instruction but I higly recommend using my [HQC_Dockerfile](https://github.com/AlexZhurkevich/HistoQC-Tiling/blob/main/HQC_Dockerfile). Simply copy it over to you machine in a separate folder and build with:<br/>
`docker build --no-cache -t histoqc/histoqc --build-arg user=YOUR_USERNAME -f HQC_Dockerfile .`<br/>
Do not forget to specify your host **username**, otherwise you are risking running docker as root and if you do not have a sudo on your machine, all resulted files will be unaccessible to you.

2. Running HistoQC:
To run HistoQC in my container you can: `docker run --rm -it -v /hdd:/mnt histoqc/histoqc:latest python3 qc_pipeline --nthreads 12 --config config.ini --outdir /mnt/HISTOQC_OUTDIR /mnt/SVS_FILE_PATH/*/*.svs`.  
I highly recommend checking out the `qc_pipeline.py` instructions on [HistoQC](https://github.com/choosehappy/HistoQC) under **Basic Usage** section. 
It will give you an idea what kind of arguments you can pass to HistoQC, you can experiment.   
In addition to HistoQC arguments, you should mount a correct volume to your container, it is done with `-v /hdd:/mnt` on sample command.
What do I mean by correct volume? It is the volume or to put it simply a directory that has your folders with `.svs` files. In my case, I am mounting my `/hdd` host machine directory (has .svs files) to `/mnt` container directory, hence `--outdir` starts with `/mnt`.

3. Upscaler + Tiler installation:
Copy over [UT_Dockerfile](https://github.com/AlexZhurkevich/HistoQC-Tiling/blob/main/UT_Dockerfile) to you machine in a separate folder and build with:  
`docker build --no-cache -t upscaler/tiler --build-arg user=YOUR_USERNAME -f UT_Dockerfile .`  
Do not forget your **username**.

4. Running Mask Upscaler:
To run Upscaler in my container you can: `docker run --rm -it -v /hdd:/mnt upscaler/tiler:latest python3 Upscaler.py --masks /mnt/HISTOQC_OUTDIR/*/*svs_mask_use.png --slides /mnt/SVS_FILE_PATH/*/*.svs`.  
`--masks` being output folder of HistoQC step, you want to go through all of the resulting folders looking for files with `svs_mask_use.png` extension, these are finalized masks that will be upscaled by HistoQC later. I do not recommend you to relocate `svs_mask_use.png` files to a separate folder because it will break a piece of code that automatically matches masks with original slides, so please just link your HistoQC output folder.  
`--slides` argument expects a glob pattern for all of your `.svs` files, its essentially the same as last positional argument you've passed to `qc_pipeline`.

5. Running Tiler:
To run Tiler in my container you can: `docker run --rm -it -v /hdd:/mnt upscaler/tiler:latest python3 SVS_Tiler.py --threads 12 --size 256 --format 'jpeg' --outdir /mnt/TILER_OUTDIR --slides /mnt/SVS_FILE_PATH/*/*.svs --masks /mnt/HISTOQC_OUTDIR/*/*svs_mask_use.tif`.  
`--threads` how many CPU threads you have, check with htop or top if you are not sure. Example: 12.  
`--size` tile size. Example: 256 (256x256).  
`--format` tile file format, I recommend jpeg, faster to write and takes less space. Example: 'jpeg' (256x256).   
`--outdir` output directory where your tiles will be stored.  
`--slides` argument expects a glob pattern for all of your `.svs` files, its essentially the same as last positional argument you've passed to `qc_pipeline` and `Upscaler`.  
`--masks` argument expects a glob pattern for all of your `svs_mask_use.tif`, these `.tif` files are upscaled masks that we got after running `Upscaler`, so you will pass almost the same thing you've passed to Upscaler under `--masks` argument, the only difference is for Upscaler the final part of the extension was `.png`, but in case of `Tiler` it will be `.tif`, hence `/mnt/HISTOQC_OUTDIR/*/*svs_mask_use.tif`. 

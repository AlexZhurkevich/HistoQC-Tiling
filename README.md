# HistoQC-Tiling
Tiling support for HistoQC

This set of programs and instructions will perform quality control analysis based on [HistoQC](https://github.com/choosehappy/HistoQC), upscale resulted HistoQC masks to match original slide resolution, tile your slide based on a given upscaled mask with useful regions identified by HistoQC.

# HistoQC
## 1. **Installation**:
You can refer to [original](https://github.com/choosehappy/HistoQC) installation instruction but I higly recommend using my [HQC_Dockerfile](https://github.com/AlexZhurkevich/HistoQC-Tiling/blob/main/HQC_Dockerfile). Simply copy it over to you machine in a separate folder and build with:<br/>
`docker build --no-cache -t histoqc/histoqc --build-arg user=YOUR_USERNAME -f HQC_Dockerfile .`<br/>
Do not forget to specify **YOUR_USERNAME** as your host **username**, otherwise you are risking running docker as root and if you do not have a sudo on your machine, all resulted files will be unaccessible to you.

## 2. **Running**:
To run HistoQC in my container you can:  
`docker run --rm -it -v /hdd:/mnt histoqc/histoqc:latest python3 qc_pipeline --nthreads 12 --config config.ini --outdir /mnt/HISTOQC_OUTDIR /mnt/SVS_FILE_PATH/*/*.svs`.  
I highly recommend checking out the `qc_pipeline.py` instructions on HistoQC under [Basic Usage](https://github.com/choosehappy/HistoQC#basic-usage) section. 
It will give you an idea what kind of arguments you can pass to HistoQC, you can experiment.  

In addition to HistoQC arguments, you should mount a correct volume to your container, it is done with `-v /hdd:/mnt` on sample command.
What do I mean by correct volume? It is the volume or to put it simply a directory that has your folders with `.svs` files. In my case, I am mounting my `/hdd` host machine directory (has .svs files) to `/mnt` container directory, hence `--outdir` starts with `/mnt`.

# Upscaler + Tiler

![](/home/kevich88/Videos/TCGA-78-7536-01Z-00-DX1.6b066140-8b82-40a5-bd48-430eeb4463fe.svs/Upscaler.gif)

## 1. **Installation**:
Copy over [UT_Dockerfile](https://github.com/AlexZhurkevich/HistoQC-Tiling/blob/main/UT_Dockerfile) to you machine in a separate folder and build with:  
`docker build --no-cache -t upscaler/tiler --build-arg user=YOUR_USERNAME -f UT_Dockerfile .`  
Do not forget your **username**.

## 2. **Running Mask Upscaler**:
To run **Upscaler.py** in my container you can: `docker run --rm -it -v /hdd:/mnt upscaler/tiler:latest python3 Upscaler.py --masks /mnt/HISTOQC_OUTDIR/*/*svs_mask_use.png --slides /mnt/SVS_FILE_PATH/*/*.svs`.  
- `--masks` being output folder of HistoQC step, you want to go through all of the resulting folders looking for files with `svs_mask_use.png` extension, these are finalized masks that will be upscaled by HistoQC later. I do not recommend you to relocate `svs_mask_use.png` files to a separate folder because it will break a piece of code that automatically matches masks with original slides, so please just link your HistoQC output folder.  
- `--slides` argument expects a glob pattern for all of your `.svs` files, its essentially the same as last positional argument you've passed to `qc_pipeline`.

## 3. **Running Slide Tiler**:
To run **SVS_Tiler.py** in my container you can: `docker run --rm -it -v /hdd:/mnt upscaler/tiler:latest python3 SVS_Tiler.py --threads 12 --size 256 --format 'jpeg' --outdir /mnt/TILER_OUTDIR --slides /mnt/SVS_FILE_PATH/*/*.svs --masks /mnt/HISTOQC_OUTDIR/*/*svs_mask_use.tif`.  

Arguments:
  - `--threads` how many CPU threads you have, check with htop or top if you are not sure. Example: 12.  
  - `--magnification` magnification at which the tiles should be taken. Example: 20.0.  
  - `--size` tile size. Example: 256 (256x256).  
  - `--overlap` you can tile your slide with overlap of `N` pixels. **Remember!!!**: the formula for overlap: `tile size + 2 * overlap`, so if you want tiles of size 256x256, you need to pass 128 as `--size` argument and 64 as `--overlap` argument. If you want more info [OpenSlide docu](https://openslide.org/api/python/). Example: 64.  
  - `--format` tile file format, I recommend jpeg, faster to write and takes less space. Example: 'jpeg'.   
  - `--outdir` output directory where your tiles will be stored.  
  - `--slides` argument expects a glob pattern for all of your `.svs` files, its essentially the same as last positional argument you've passed to `qc_pipeline` and `Upscaler`.  
  - `--masks` argument expects a glob pattern for all of your `svs_mask_use.tif`, these `.tif` files are upscaled masks that we got after running `Upscaler`, so you will pass almost the same thing you've passed to Upscaler under `--masks` argument, the only difference is for Upscaler the final part of the extension was `.png`, but in case of `Tiler` it will be `.tif`, hence `/mnt/HISTOQC_OUTDIR/*/*svs_mask_use.tif`. 

# Sorting
## 1. **Installation**:
Copy over [TF_Dockerfile](https://github.com/AlexZhurkevich/HistoQC-Tiling/blob/main/TF_Dockerfile) to you machine in a separate folder and build with:  
`docker build --no-cache -t tf/tf:latest -f TF_Dockerfile .`  
No **username** needed, we will pass it at runtime. We will also use this docker image for the rest of pipeline, hold on to it. 

## 2. **Running Sorting**:
Sorting program was taken from [here](https://github.com/ncoudray/DeepPATH/tree/master/DeepPATH_code). To run **Sort_Tiles.py** in my container, you need to manually copy over **Sort_Tiles.py** to a folder where you would like to store the output of this program. Sample command:  
`docker run -t -i -u $(id -u ${USER}):$(id -g ${USER}) -w /mnt/YOUR_SORT_FOLDER -v /hdd:/mnt tf/tf:latest python3 Sort_Tiles.py --SourceFolder=/mnt/TILER_OUTDIR --JsonFile=/mnt/YOUR_SORT_FOLDER/metadata_file --Magnification=20 --MagDiffAllowed=0 --SortingOption=6 --PercentTest=10 --PercentValid=10 --nSplit 0`
### Docker arguments:
  - `-u $(id -u ${USER}):$(id -g ${USER})` sets a user that uses a container, this particular command will set your host username as username running inside of the container, this will eliminate privilige issues. I do not recommend changing it unless you know what are you doing.  
  - `-v` mount volume of host machine to a container, you should mount a volume that has `outdir` of **SVS_Tiler.py** as well as directory where you store **Sort_Tiles.py**. This way you will have an access to files from previous step, as well as sorting program. Example /hdd:/mnt.
  - `-w` setting workplace, this is a container directory that contains **Sort_Tiles.py** file you've mounted with `-v`. Example /mnt/YOUR_SORT_FOLDER. **Remember!!!**:
 ince you've mounted host volume to a container, access to directory inside a container will be relative to a container. In other words, if you have **Sort_Tiles.py** in `/hdd/YOUR_SORT_FOLDER/Sort_Tiles.py` on host machine, when you mount with `-v /hdd:/mnt`, container filepath is this `/mnt/YOUR_SORT_FOLDER/Sort_Tiles.py` and `/mnt/YOUR_SORT_FOLDER/` is argument you should pass, otherwise container wont see **Sort_Tiles.py** file.

### Sort_Tiles.py arguments:
I recommend checking out [argument instructions](https://github.com/ncoudray/DeepPATH/tree/master/DeepPATH_code#02a-sort-the-tiles-into-trainvalidtest-sets-according-to-the-classes-defined). I recommend keeping your `--JsonFile=` file in the same folder as **Sort_Tiles.py**, for `--Magnification=` pass the same thing you've passed for magnification at [Tiler](https://github.com/AlexZhurkevich/HistoQC-Tiling#3-running-slide-tiler) step. Pass the output directory of **SVS_Tiler.py** to `--SourceFolder=`. 

# TFRecords
## 1. **Installation**:
We will be using the same docker image that we've built during [sorting step](https://github.com/AlexZhurkevich/HistoQC-Tiling#1-installation-2).

## 2. **Running TFRecords Creation**:
To run **TFRecord_Creator.py** in my container you can:  
`docker run -t -i -u $(id -u ${USER}):$(id -g ${USER}) -v /hdd:/mnt tf/tf:latest python3 TFRecord_Creator.py --sort_dir '/mnt/YOUR_SORT_FOLDER' --outdir '/mnt/YOUR_TFRecords' --threads 12 --num_files 1020 --size 256 --format 'jpeg' --oversampling 'Yes'`.

Arguments:
  - `--threads` how many CPU threads you have, check with htop or top if you are not sure. Example: 12.  
  - `--sort_dir` a directory where your sorted dataset resides, a place where you ran **Sort_Tiles.py** at previous step. Example: '/mnt/YOUR_SORT_FOLDER'.
  - `--outdir` a directory where you would like to see TFRecords. Example: '/mnt/YOUR_TFRecords'.
  - `--num_files` numer of train TFRecords you want, **Remember!!!**: the number given should be divisible by number of threads. Example: 1020.
  - `--size` tile size. Example: 256 (256x256).  
  - `--format` tile file format. Example: 'jpeg'.
  - `--oversampling` whether you want to oversample your minority class, it will repeat images from minority class until it comes close to number of images from majority class. It is done only for training. **Remember!!!**: for now only works with 2 classes. Example: 'Yes' or 'No'.

# Training
## 1. **Installation**:
We will be using the same docker image that we've built during [sorting step](https://github.com/AlexZhurkevich/HistoQC-Tiling#1-installation-2). In order to train on GPUs with **docker**, you need to install [NVIDIA CONTAINER TOOLKIT](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), it will give you an ability to use `--gpus` argument. 

## 2. **Running Training**:
To run **Training.py** in my container you can:  
`docker run --gpus all -t -i -u $(id -u ${USER}):$(id -g ${USER}) -v /hdd:/mnt tf/tf:latest python Training.py --train_num 2249551 --valid_num 161976 --epochs 500 --size 256 --train_dir '/mnt/YOUR_TFRecords/train*.tfrecord' --valid_dir '/mnt/YOUR_TFRecords/valid*.tfrecord' --ckpt_name '/mnt/YOUR_TRAIN_OUTDIR/Best_Model' --csv_log_name '/mnt/YOUR_TRAIN_OUTDIR/Training.log' --MP 'Yes' --tensorboard_logs '/mnt/YOUR_TRAIN_OUTDIR/TB_logs' --GPU_num 0,1 --batch_size 28`

Arguments:
  - `--batch_size` your typical batch size, scaled linearly with multiple GPU. Example: 64. 
  - `--GPU_num` which GPUs your want to use, one digit for one particular GPU, multiple for multiple GPUs (comma separated). Examples: '0' (first availbale GPU) or '0,1' (first two GPUs).
  - `--train_num` number of training images, was given at the end of **Running TFRecords Creation** execution. Example: 2249551.
  - `--valid_num` number of validation images, was given at the end of **Running TFRecords Creation** execution. Example: 161976.
  - `--epochs` for how many epochs you would like to run, if no improvement will be early stopped. Example: 200.
  - `--size` tile size. Example: 256 (256x256).  
  - `--train_dir` this argument expects train files' `glob` pattern. Example: '/mnt/YOUR_TFRecords/train*.tfrecord'
  - `--valid_dir` this argument expects validation files' `glob` pattern. Example: '/mnt/YOUR_TFRecords/valid*.tfrecord'
  - `--ckpt_name` this is the name/folder of your best performing model, you can also pass filepath with it. Example: '/mnt/YOUR_TRAIN_OUTDIR/Best_Model'
  - `--csv_log_name` this is a log name that will store your training progress information in a csv file, you can also pass filepath with it. Example: '/mnt/YOUR_TRAIN_OUTDIR/Training.log'
  - `--tensorboard_logs` this is a folder which will have all information needed for [TensorBoard](https://www.tensorflow.org/tensorboard/get_started). If you are not familiar with this tool, I highly suggest checking it out. 
  - `--MP` this argument is for [Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). If you are not familiar with the concept I highly suggest checking it out, it can speed up your training by 3.3x, you can also fit 2x batch size. Example: 'Yes' or 'No'.
 
# Testing
## 1. **Installation**:
We will be using the same docker image that we've built during [sorting step](https://github.com/AlexZhurkevich/HistoQC-Tiling#1-installation-2). In order to test on GPUs with **docker**, you need to install [NVIDIA CONTAINER TOOLKIT](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), it will give you an ability to use `--gpus` argument. 

## 2. **Running Testing**:
To run **Testing.py** in my container you can:  
`docker run --gpus all -t -i -u $(id -u ${USER}):$(id -g ${USER}) -v /hdd:/mnt tf/tf:latest python3 Testing.py --mode 'evaluate' --GPU_num '2' --batch_size 288 --model_ckpt '/mnt/YOUR_TRAIN_OUTDIR/Best_Model' --test_dir '/mnt/YOUR_TFRecords/test*.tfrecord' --test_num 161976 --output_csv '/mnt/YOUR_TEST_OUTPUT/Eval_Output.csv'` or you can `docker run --gpus all -t -i -u $(id -u ${USER}):$(id -g ${USER}) -v /hdd:/mnt tf/tf:latest python3 Testing.py --mode 'predict' --GPU_num '2' --batch_size 288 --model_ckpt '/mnt/YOUR_TRAIN_OUTDIR/Best_Model' --test_dir '/mnt/YOUR_TFRecords/test*.tfrecord' --test_num 161976 --roc_im_name 'Best_ROC.jpeg' --threshold 0.5`

Before we will start talking about arguments, I would like to mention that there are 2 modes you can use while testing the model, it is the reason I've included two commands as an example. First mode is `evaluate`, this will evaluate the quality of the model based on predefined metrics, you can add your own metric if you want, the output will be a dictionary with a metric name and a score, it can also be saved as csv via `--output_csv` argument. Second mode is `predict`, this one will give you a list of true labels as well as a list of predicted probabilities (per tile basis), you can use these lists as inputs into various functions from [scikit-learn](https://scikit-learn.org/stable/) or any other data science library, it will empower you to do any kind of analysis, sky's the limit. Currently, predict will draw [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html), you can save it with `--roc_im_name` argument, it also suppors thresholding of predicted probabilities via `--threshold` argument. 

Arguments:
  - `--mode` described abouve modes. Example: 'evaluate' or 'predict'.
  - `--GPU_num` which GPUs your want to use, limited to one GPU. Example: '0'.
  - `--batch_size` your typical batch size, scaled linearly with multiple GPU. Example: 288.
  - `--model_ckpt` saved best performing model. Example: '/mnt/YOUR_TRAIN_OUTDIR/Best_Model'.
  - `--test_dir` this argument expects test files' `glob` pattern. Example: '/mnt/YOUR_TFRecords/test*.tfrecord'.
  - `--test_num` number of testing images, was given at the end of **Running TFRecords Creation** execution. Example: 161976.
  - `--output_csv` your metrics' scores written into a csv file, only in `evaluate` mode. Example: '/mnt/YOUR_TEST_OUTPUT/Eval_Output.csv'.
  - `--roc_im_name` name of the ROC cruve file, only in `predict` mode. Example: '/mnt/YOUR_TEST_OUTPUT/Best_ROC.jpeg'.
  - `--threshold` threshold for predicted values, only in `predict` mode, should be between 0 and 1. Example: 0.5.




# Python instructions:
## HistoQC
If you do not want to use docker, you can install HistoQC manually, instructions [here](https://github.com/choosehappy/HistoQC). 

## Upscaler + Tiler prerequisites
I can confirm code being runnable with these libraries and corresponding versions: `openslide-tools v3.4.1, python3-openslide v1.1.1-4, libvips v8.9.1-2, python v3.8.5`, you can install these via `apt-get install`. Specific to python3 are: `pyvips v2.1.14, Pillow v7.0.0, openslide-python v1.1.1`, you can install these via `pip3 install`. If you want all the details you can look at [UT_Dockerfile](https://github.com/AlexZhurkevich/HistoQC-Tiling/blob/main/UT_Dockerfile) and check what I am installing in docker image. I also recommend using `ubuntu 20.04+`, because older versions might have some incompatibilities with newer `libvips` versions, you can still use `libvips` on older distros but you might need to build it from [source](https://libvips.github.io/libvips/install.html).  

## Sorting, TFRecords, Training, Testing prerequisites
I can confirm code being runnable with these libraries and corresponding versions: `tensorflow-gpu v2.4.1, opencv-python v4.5.1.48, matplotlib v3.3.4, Pillow v8.1.2, scikit-learn v0.24.1, scikit-image v0.17.2`, you can install these via `pip3 install`. If want to run on GPU make sure that you've installed correct [tensorflow prerequisites](https://www.tensorflow.org/install/source#gpu).

## Running python code
In order to run the programs there wont be any changes, same arguments, same programs, just drop `docker run --rm -it -v /hdd:/mnt upscaler/tiler:latest` and `docker run -t -i -u $(id -u ${USER}):$(id -g ${USER}) -w /mnt/YOUR_SORT_FOLDER -v /hdd:/mnt tf/tf:latest` parts, run as usual with `python3`.


**Additional information**:
1. Speed:
Intel(R) Core(TM) i7-5820K CPU overclocked to 4.09Ghz, 12 threads.  
Dataset: 1608 TCGA LUAD .svs slides.    
HistoQC runtime: 3.5 Hours.  
Upscaler runtime: 15 Hours.  
Tiler runtime: 19 Hours.  
Scales lineary depending on how many threads you have, at some point might encounter I/O bottlenecks if using hard drive.  

2. Whats next?
You can either work with tiles however you want or your can continue with Nicolas Coudray's [DeepPath](https://github.com/ncoudray/DeepPATH/tree/master/DeepPATH_code) your steps will start at **0.2a Sort the tiles into train/valid/test sets according to the classes defined** section. 




Thank you. Consider sharing on social media. 

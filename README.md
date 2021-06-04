# CrowdCounting on VisDrone2020

In this repo we use [MobileCount](https://github.com/SelinaFelton/MobileCount) models plus 2 two variants of it

## Trying the net on Colab

Try the following notebook:
https://colab.research.google.com/drive/1NsSAY_LGpDdUFglhv2h7mK_jUya4gLRV?usp=sharing

## Running the code

### Requirements

Python 3.8

Install requirement.txt

To execute use:
  
    python main.py args
    
The first arg is the modality, can be: train, test, run.

For train and test all the parameters given in config.py and the the chosen dataset py file, will be used.

For run mode, you must also specify:

<ul>
<li>--path: path to the video or image file, or the folder containing the image</li>
<li>--callbacks: list of callback function to be executed after the forward of each element</li>
</ul>

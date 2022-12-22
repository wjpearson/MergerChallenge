# Merger Challenge
## William J. Pearson

Code for William J. Pearson's entry to the merger challenge.

## Method

### Architecture

The [swin_tiny_patch4_window7_224_fe](https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224_fe/1) Swin Transformer of [Sayak Paul](https://github.com/sayakpaul/swin-transformers-tf) that was pre-trained on the ImageNet-1k dataset was used. A fully-connected layer with 256 neurons is added along with an output later of 1 neuron (binary) or 4 neurons (quad).


### Training

1. TNG100 training Images are cropped to 128x128 pixels with `split_crop_train.py`
2. Images are split ~80%-20% for training-validation, keeping images from the same merger history in each split
3. In the training scripts ( `TNG100-swin-transfer-###-cut.py` ) images are:
    1. Cropped again to 112x112 pixels
    2. Scaled between 0 and 1
    3. Stacked with itself to form a 3 channel image
    4. Scaled to 224 with nearest neighbour
4. During training, training images (but not validation images) are:
    + Randomly rotated by 90<sup>o</sup> AND
    + Randomly flipped left/right AND
    + Randomly flipped up/down
5. The binary model with the lowest vaidation loss and the quad model with the lowest validation loss and highest validation accuracy are saved
    
### Testing

1. TNG100 test, HSC and Horizon-AGN images are cropped to 128x128 with `split_crop_test_TNG100.py` or `split_crop_train.py` (for HSC and Horizon-AGN)
2. In the test scripts ( `TNG100-swin-transfer-###-cut-predict.py` ) images are:
    1. Cropped again to 112x112 pixels
    2. Scaled between 0 and 1
    3. Stacked with itself to form a 3 channel image
    4. Scaled to 224 with nearest neighbour
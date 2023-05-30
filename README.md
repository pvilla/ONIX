# ONIX: an X-ray deep-learning tool for 3D reconstructions from sparse views

Yuhe Zhang\*, Zisheng Yao, Tobias Ritschel, Pablo Villanueva-Perez

ONIX is a deep-learning 3D X-ray reconstruction approach allowing reconstructing 3D from sparse 2D projections.  
For more detailed information about ONIX training and performance, please refer to our paper: [https://doi.org/10.1002/appl.202300016](https://doi.org/10.1002/appl.202300016).

<p align="center">
<img src="images/ONIX-Illustration.png" width="650"/>
</p>

https://user-images.githubusercontent.com/67344944/164986819-91df30bf-637d-41bb-a5a2-c6670f27faac.mp4


## Getting Started
### Prerequisites

- Linux (not tested for MacOS or Windows)
- Python3
- NVIDIA GPU (not tested for CPU)

### Installation

Clone this repo:

```
git clone https://github.com/pvilla/ONIX.git
cd ONIX
```
To install the required python packages:

```
conda env create -f environment.yml
conda activate onix
```

### Dataset 

We provide a customized npy Dataloader in [DatasetCustom.py](https://github.com/pvilla/ONIX/blob/master/models/DatasetCustom.py).
For the training of attenuation + phase, the dataset should have a dimension of [dataset_size, num_projections, 2, H, W]. For the one-channel training, the dataset should be [dataset_size, num_projections, H, W]. 
Please check [DatasetCustom.py](https://github.com/pvilla/ONIX/blob/master/models/DatasetCustom.py) and [DatasetTest.py](https://github.com/pvilla/ONIX/blob/master/models/DatasetTest.py) for details regarding the training and test data preparation.

If `hdf5` files are to be used, please install [h5py](https://anaconda.org/anaconda/h5py) package which is not included in the dependencies.

Parallel X-ray configuration is used in our approach. 
Please be sure to modify the `generate_poses` function in the dataset preparation files to match the real experimental setup. 
One may need to change the geometrical configuration part in [trainer.py](https://github.com/pvilla/ONIX/blob/master/models/trainer.py) to use other beam configurations.

Example training dataset is not publicly available at this time but may be obtained upon request.


### Training
To run the training:

`python3 train.py`

For more training options, please check out:

`python3 train.py --help`

We also provide a training script to make it easier (suggested):

`. train.sh <run_name> <n_views> <num_val_views> <load_path>`


### Results
The training results and the trained models will be saved in: `run_path/run_name`.
The training parameters and losses will be saved to a txt file: `.run_path/run_name/log.txt`.
If you launch training from `train.sh`, a run.log file will be automatically generated in the onix directory, which was redirected from the standard output.
Voxel 3D models (\*.npy) with the resolution HxWxW are also saved in the working directory after each validation step. 
 

### Evaluation
Run `eval.py` to check the performance of the trained model and save 3D models. 
For evaluation options, please check out:

`python3 eval.py --help`

## Citation
If you use this code for your research, please cite our paper.
```
@article{zhang2022onix,
  title={ONIX: an X-ray deep-learning tool for 3D reconstructions from sparse views},
  author={Zhang, Yuhe and Yao, Zisheng and Ritschel, Tobias and Villanueva-Perez, Pablo},
  journal={arXiv preprint arXiv:2203.00682},
  year={2022}
}
```
## Acknowledgments
Parts of the code were based on [nerf-pytorch](https://github.com/krrish94/nerf-pytorch) and [pixelNeRF](https://github.com/sxyu/pixel-nerf).

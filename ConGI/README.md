



# Identifying Spatial Domain by Adapting Transcriptomics with Histology through Contrastive Learning




## Usage

I used this code to try it on our ST data. This readme is the original readme from the authors of this code. I made some modifications for the code to fit with our data. These modifications are explained in the "Modifications" part of this readme.

## Overview
We propose a novel method ConGI to accurately decipher spatial domains by integrating gene expression and histopathologi-cal images, where the gene expression is adapted to image infor-mation through contrastive learning. We introduce three contrastive loss functions within and between modalities to learn the common semantic representations across all modalities while avoiding their meaningless modality-private noise information. The learned rep-resentations are then used for deciphering spatial domains through a clustering method. By comprehensive tests on tumor and normal spatial transcriptomics datasets, ConGI was shown to outperform existing methods in terms of spatial domain identification. More importantly, the learned representations from our model have also been used efficiently for various downstream tasks, including trajectory inference, clustering, and visualization.

![(Variational) gcn](framework.bmp)


## Requirements
Please ensure that all the libraries below are successfully installed:

- **torch 1.7.1**
- CUDA Version 10.2.89
- scanpy 1.8.1
- mclust








## Run ConGI on the example data.

```

python train.py --dataset SpatialLIBD  --name 151509 

```


## output

The clustering labels will be stored in the dir `output` /dataname_pred.csv. 


## Tutorial

We also provide a [Tutorial](https://github.com/biomed-AI/ConGI/blob/main/tutorial.ipynb) script for users. 



## Datasets

The spatial transcriptomics datasets that support the findings of this study are available here:
(1) human HER2-positive breast tumor ST data https://github.com/almaan/HER2st/. 
(2) The LIBD human dorsolateral prefrontal cortex (DLPFC) data was acquired with 10X Visium composed of spatial transcriptomics data acquired from twelve tissue slices (http://research.libd.org/spatialLIBD/).
(3) The mouse brain anterior section from 10X Visium (https://www.10xgenomics.com/resources/datasets). 
(4) the human epidermal growth factor receptor (HER) 2-amplified (HER+) invasive ductal carcinoma (IDC) (https://support.10xgenomics.com/spatial-gene-expression/datasets). 




## Citation

Please cite our paper:

```

@article{zengys,
  title={Deciphering Spatial Domains by Integrating Histopathological Image and Tran-scriptomics via Contrastive Learning },
  author={Yuansong Zeng1,#, Rui Yin3,#, Mai Luo1, Jianing Chen1, Zixiang Pan1, Yutong Lu1, Weijiang Yu1* , Yuedong Yang1,2*},
  journal={biorxiv},
  year={2022}
 publisher={Cold Spring Harbor Laboratory}
}

```
## Modifications

Modification that I made on the script :

In dataset.py :

Add a variable pathI to open my tiles from the image.

Add a part to open my data when the dataset variable is “Space_Ranger_DB” to run the code with our preprocessed data. 

Add a part to open my position_list file and add it to the anndata object (in a try/except block).

Save the new anndata object with the same name but with a _2 at the end.

Add a part to open my data when the dataset variable is “Space_Ranger_DB2” to run the code with our raw data. 

In utils.py :

Add a part to open my data (with the position_list file, so it’s the anndata object with “_2” at the end) when the dataset variable is “Space_Ranger_DB” and to perform the get_predicted_results function.

Add a part to the get_predicted_results in “Space_Ranger_DB2” to make a choice for using refine function for clusterisation.

In model.py :

Import these packages : from torch.cuda.amp import autocast, GradScaler.

Add a scaler, an autocast, and delete of memory after each loop to optimize the memory of my GPU in the train function.

Add a bloc to load my dataset in TrainerSpaCLR class.

In metrics.py:

os.environ['R_HOME'] = '/GPUFS/sysu_ydyang_10/.conda/envs/r-base/lib/R'
os.environ['R_USER'] = '/GPUFS/sysu_ydyang_10/.conda/envs/r-base/lib/python3.9/site-packages/rpy2'

Change these 2 previous lines by the 2 following lines in the mclust function to redirect them to my R path.

export R_HOME=/disk2/user/cormey/miniconda3/bin/R
export R_LIBS_USER=/disk2/user/cormey/miniconda3/bin/Rscript

In tutorial :

Adjust R path to my environment to use some R packages in the code.

Add new parser argument “pathI” to stock my path to my tiles.

Add pathI in all the function that need it (for the training mainly).

Change the path to save the embeddings.
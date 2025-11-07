# cell-comparative-MIL-guided-by-IQA-for-cervical-WSI
Pytorch implementation for the model described in the paper **A cell comparative multiple instance learning network guided by image quality assessment for cervical whole slide image classification**

### Installation

Install the required packages

```
conda env create --name pytorch --file env.yml
conda activate pytorch
```

Install [pytoch](https://pytorch.org/get-started/locally/)

Install [openslide](https://pypi.org/project/openslide-python/)

### Training on your own datasets

1. Place WSI files as `WSI\[DATASET_NAME]\[CATEGORY_NAME]\[SLIDE_FOLDER_NAME] (optional)\SLIDE_NAME.svs`
2. Crop patches.

`python deepzoom_tiler.py -m 0 -b 20 -d [DATASET_NAME]`

3. Train an embedder.

  `cd simclr`
  `python run.py --dataset=[DATASET_NAME]`

4. Compute features using the embedder.

```
python compute_feats.py --dataset=[DATASET_NAME]
```

5. Training.

```
python train_cell.py --dataset=[DATASET_NAME]
```

6.   Testing 

`python attention_map.py --bag_path test/patches --map_path test/output --thres 0.73 0.28`



**Our code refers to DSMIL:**

@inproceedings{li2021dual,
  title={Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning},
  author={Li, Bin and Li, Yin and Eliceiri, Kevin W},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14318--14328},
  year={2021}
}

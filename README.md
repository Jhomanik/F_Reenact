# F_Reenact
## Set up Anaconda environment:

```
conda create -n hyperreenact_env python=3.8
conda activate hyperreenact_env
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

## Download all pretrained models
Add to ./pretrained_models

1) Style_encoder 
https://drive.google.com/file/d/1ZLJAuagW46QHtAnhqiuFtYxVdNbJ-QH-/view?usp=sharing

2) StyleGAN2 trained on VoxCeleb1 dataset 
https://drive.google.com/file/d/1cBwIFwq6cYIA5iR8tEvj6BIL7Ji7azIH/view

3) Face detector 
https://drive.google.com/file/d/1IWqJUTAZCelAZrUzfU38zK_ZM25fK32S/view

4) ArcFACE 
https://drive.google.com/file/d/1F3wrQALEOd1Vku8ArJ_Gn4T6U3IX7Pz7/view

Extract data.tar.gz under  ./libs 
https://drive.google.com/file/d/1BHVJAEXscaXMj_p2rOsHYF_vaRRRHQbA/view

## Dataset
We use Voxceleb1 dataset for training (~100k images pairs) and Voxceleb2 dataset for evaluation (~5k images pairs):
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

For dataset preprocessing, we refer to 
https://github.com/StelaBou/voxceleb_preprocessing 


## Run script
```
python run_trainer.py  --batch_size 8 --lr 1e-4  --experiment_path "..." --type "cross"  --train_dataset_path  "Voxceleb1" --test_dataset_path  "Voxceleb2" --cherry_dataset_path  'Cherry images' 
```

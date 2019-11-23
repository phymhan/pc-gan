# Robust Conditional GAN from Uncertainty-Aware Pairwise Comparisons

[[arXiv]](https://arxiv.org/abs/1911.09298)
[[Supplementary]](https://github.com/phymhan/pc-gan/blob/master/pdf/pc-gan_supplementary.pdf)

The code is heavily based on the PyTorch implementation of [CycleGAN-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

### Pretrained Models
Coming soon...

### Setup
#### Data Preparation
Take UTKFace for example,
```
DATASET=UTK
ln -s path/to/dataset/folder/that/contains/all/images $(pwd)/datasets/${DATASET}
ls datasets/${DATASET} > ${DATASET}_all.txt
```

##### Split dataset
This will create `${DATASET}_train.txt`, `${DATASET}_val.txt`, and `${DATASET}_test.txt` under `./sourcefiles/`, each of which contains 10000, 1000, and 10000 filenames respectively.
Then sample 1000 images from train and val for visualization.
```
# split dataset
python scripts/split_dataset.py --input sourcefiles/${DATASET}_all.txt --output sourcefiles/${DATASET} --num_samples 10000 1000 10000

# sample a subset for visualization
python scripts/sample_sourcefile.py --input sourcefiles/${DATASET}_train.txt sourcefiles/${DATASET}_val.txt --output sourcefiles/${DATASET}_sample.txt --num_samples 1000
```

##### Simulate pairs
Generate pairs for train and val,
```
# generate pairs
python scripts/generate_pairs.py --input sourcefiles/${DATASET}_train.txt --output sourcefiles/${DATASET}_pairs_train.txt --num_pairs 20000 --margin 10
python scripts/generate_pairs.py --input sourcefiles/${DATASET}_val.txt --output sourcefiles/${DATASET}_pairs_val.txt --num_pairs 1000 --margin 10

# extract different pairs for training PC-GAN
python scripts/reweight_sourcefile.py --input sourcefiles/${DATASET}_pairs_train.txt --output sourcefiles/${DATASET}_diff_pairs_train.txt --sample_weight 1 0 1
```

#### Elo Rating Network
Open a new terminal to start a `visdom` server with `python -m visdom.server`.

Train a basic CNN Elo rating network,
```
python siamese.py \
--dataroot datasets/${DATASET} \
--datafile sourcefiles/${DATASET}_pairs_train.txt \
--dataroot_val datasets/${DATASET} \
--datafile_val sourcefiles/${DATASET}_pairs_val.txt \
--datafile_emb sourcefiles/${DATASET}_sample.txt \
--mode train \
--name elo_${DATASET}_cnn
```

Train a Bayesian Elo rating network,
```
python siamese.py \
--dataroot datasets/${DATASET} \
--datafile sourcefiles/${DATASET}_pairs_train.txt \
--dataroot_val datasets/${DATASET} \
--datafile_val sourcefiles/${DATASET}_pairs_val.txt \
--datafile_emb sourcefiles/${DATASET}_sample.txt \
--mode train \
--name elo_${DATASET}_bnn \
--noisy true \
--bayesian true \
--bnn_dropout 0.2
```

#### PC-GAN Model
Get the mean (`embedding_mean`), std (`embedding_std`), and quantization clusters (`embedding_bins`) of the ratings (or 'embeddings' called in the code) using `get_embedding_cluster.m` in `./scripts/`. In the following commands, replace the corresponding argument values with the computed values.
To train a basic PC-GAN with a basic encoder,
```
python train.py \
--model wsgan_emb \
--dataroot datasets/${DATASET} \
--sourcefile_A sourcefiles/${DATASET}_diff_pairs_train.txt \
--name emb_${DATASET}_cnn \
--pretrained_model_path_E checkpoints/elo_${DATASET}_cnn/latest_net.pth \
--embedding_mean 0 \
--embedding_std 1 \
--embedding_bins "[-2, -1, 0, 1, 2]"
```

To train a PC-GAN with a Bayesian encoder,
```
python train.py \
--model wsgan_emb \
--dataroot datasets/${DATASET} \
--sourcefile_A sourcefiles/${DATASET}_diff_pairs_train.txt \
--name emb_${DATASET}_bnn \
--noisy_var_type ae \
--noisy true \
--bayesian true \
--bnn_dropout 0.2 \
--pretrained_model_path_E checkpoints/elo_${DATASET}_bnn/latest_net.pth \
--embedding_mean 0 \
--embedding_std 1 \
--embedding_bins "[-2, -1, 0, 1, 2]"
```

# chinese_im2text.pytorch

This project is based on ruotian's [neuraltalk2.pytorch](https://github.com/ruotianluo/neuraltalk2.pytorch).

## Requirements

### Software enviroment
Python 2.7 and Python 3 ([coco-caption](https://github.com/tylin/coco-caption)), PyTorch 0.2 (along with torchvision). 

### Dataset
You need to download pretrained resnet model for both training and evaluation, and you need to register the ai challenger, and then download the training and validation dataset.

## Pretrained models

TODO

## Train your own network on AI Challenger
### Download AI Challenger dataset and preprocessing
First, download the 图像中文描述数据库 from [link](https://challenger.ai/datasets).
or
```
  https://pan.baidu.com/s/1zG-qwf8otow-QZk7XsrERw
  code: 1s4q
```  
We need training images (210,000) and val images (30,000). You should put the `ai_challenger_caption_train_20170902/` and `ai_challenger_caption_train_20170902/` in the same directory, denoted as `$IMAGE_ROOT`. Once we have these, we can now invoke the `json_preprocess.py` and `prepro_ai_challenger.py` script, which will read all of this in and create a dataset (two feature folders, a hdf5 label file and a json file).

```bash
$ python scripts/json_preprocess.py
$ python -m scripts.prepro_ai_challenger
```

`json_preprocess.py` will first transform the AI challenger Image Caption_json to mscoco json format. Then map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `coco_ai_challenger_raw.json`.

`prepro_ai_challenger.py` extract the resnet101 features (both fc feature and last conv feature) of each image. The features are saved in `coco_ai_challenger_talk_fc.h5` and `coco_ai_challenger_talk_att.h5`, and resulting files are about 359GB.


### Start training
The following training procedure are adopted from ruotian's project, and if you need REINFORCEMENT-based approach, you can clone from [here](https://github.com/ruotianluo/self-critical.pytorch). For ai challenger, they provide large number of validation size, you can set `--val_images_use` to a bigger size.

```bash
$ python train.py --id st --caption_model show_tell --input_json data/cocotalk.json --input_fc_h5 data/coco_ai_challenger_talk_fc.h5 --input_att_h5 data/coco_ai_challenger_talk_att.h5 --input_label_h5 data/coco_ai_challenger_talk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_st --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 25
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `save/`). We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

If you have tensorflow, the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

The current command use scheduled sampling, you can also set scheduled_sampling_start to -1 to turn off scheduled sampling.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

For more options, see `opts.py`. 

I am training this model with stack-captioning, and the training loss is as follows:
![](./vis/training_log_mine.png)

Some predicted descriptions are as follows (image xxx, xxx is the image ID):
```bash
Beam size: 5, image 3750: 球场上有一个穿着运动服的男人在说话
Beam size: 5, image 3751: 一个右手拿着包的男人在T台场上走秀
Beam size: 5, image 3752: 一个穿着西装的男人和一个穿着西装的男人站在室外
Beam size: 5, image 3753: 道路上手有一个穿着着包的女人和一个男人
Beam size: 5, image 3754: 两个穿着运动服的男人在运动场上奔跑
Beam size: 5, image 3755: 房间里有一个穿着短袖的男人在给一个穿着短袖的孩子
Beam size: 5, image 3756: 舞台上手拿着话筒的男人在舞台上唱歌
Beam size: 5, image 3757: 两个穿着运动服的男人在球场上踢足球
Beam size: 5, image 3758: 室外有着长起的女人坐在椅子上
Beam size: 5, image 3759: 两个穿着球衣的男人在运动场上打篮球
Beam size: 5, image 3760: 两个穿着球衣的男人在球在球场上奔跑
Beam size: 5, image 3761: 一个右手拿着包的女人站在道路上
Beam size: 5, image 3762: 一个穿着裙子的女人和一个穿着裙子的女人走在道路上
Beam size: 5, image 3763: 宽敞油的球服的男人旁有一个穿着球
Beam size: 5, image 3764: 一个穿着裙子的女人站在广告牌前
Beam size: 5, image 3765: 球场上有两个穿着球衣的男人在打篮球
Beam size: 5, image 3766: 室外有一个人前有一个戴着帽子的男人在给一个孩子
Beam size: 5, image 3767: 一个穿着西装的男人和一个双手站在道路上
Beam size: 5, image 3768: 一个右手拿着话筒的男人站在广告牌前的红毯上
Beam size: 5, image 3769: 球场上的球场上有一个穿动场上踢足球
Beam size: 5, image 3770: 两个人旁有一个人旁边有一个戴着帽子的孩的男人在道
Beam size: 5, image 3771: 一个穿着裙子的女人站在广告牌前
Beam size: 5, image 3772: 运动场上拿着球拍的女人在打羽毛球
Beam size: 5, image 3773: 广告牌前有一个机的男人旁有一个双手拿话筒的男人在说话
Beam size: 5, image 3774: 一个人旁有一个穿着裙子的女人坐在室内的椅子上
Beam size: 5, image 3775: 一个右手拿着手机的女人在道路上
Beam size: 5, image 3776: 一个戴着帽子的男人在室内
Beam size: 5, image 3777: 道路上有一个右手拿着包的女人在走秀
Beam size: 5, image 3778: 一个戴着墨镜的女人走在道路上
Beam size: 5, image 3779: 一个双手拿着话筒的女人站在广告牌前
Beam size: 5, image 3780: 一个戴着墨镜的女人走在大厅里
Beam size: 5, image 3781: 室内一个人旁有一个右手拿着笔的男人在下围棋
Beam size: 5, image 3782: 一个右手拿着东西的女人站在道路上
Beam size: 5, image 3783: 屋子里有一个右手拿着手机的女人坐在椅子上
Beam size: 5, image 3784: 一个右手拿着话筒的女人和一个穿着裙子的女人站在舞台上
Beam size: 5, image 3785: 一个戴着墨镜的女人和一个戴着墨镜的女人坐在船上
Beam size: 5, image 3786: 绿油的草地上有两个戴着帽子的男人在草地上
Beam size: 5, image 3787: 球场上有一个右手拿着球拍的男人在打羽毛球
Beam size: 5, image 3788: 两个穿着球衣的男人走在球场上奔跑
Beam size: 5, image 3789: 球场上有两个穿着球衣的运动员在打排球
Beam size: 5, image 3790: 一个穿着裙子的女人坐在室内的沙发上
Beam size: 5, image 3791: 大厅里有一个人旁有一起的男人坐在室
Beam size: 5, image 3792: 舞台上两个的舞着旁有一个男人
Beam size: 5, image 3793: 一个穿着裙子的女人站在大厅内
Beam size: 5, image 3794: 球场上的球的男人旁有一个穿着球衣的男人在踢足球
Beam size: 5, image 3795: 两个穿着球衣穿着球衣的男人在运动场上争抢足球
Beam size: 5, image 3796: 运动场的前面上有着运动服的男人和一个
Beam size: 5, image 3797: 一个穿着裙子的衣服的女人站在道路上
Beam size: 5, image 3798: 室内有一个左腿上坐在一个坐发上
Beam size: 5, image 3799: 屋子里有一个右手拿着衣服的男人在下围棋
```

## Generate image captions

### Evaluate on raw images
Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```bash
$ python eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```

This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size`. Use `--num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```bash
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

### Evaluate on validation split

```bash
$ python eval.py --dump_images 0 --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1 
```

The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_max 1`), to sample from the posterior, set `--sample_max 0`.

**Beam Search**. Beam search can increase the performance of the search for greedy decoding sequence by ~5%. However, this is a little more expensive. To turn on the beam search, use `--beam_size N`, N should be greater than 1 (we set beam size to 5 in our eval).

## Acknowledgements

Thanks the original [neuraltalk2](https://github.com/karpathy/neuraltalk2), and the pytorch-based [neuraltalk2.pytorch](https://github.com/ruotianluo/neuraltalk2.pytorch) and awesome PyTorch team.

## Paper

1. Jiuxiang Gu, Gang Wang, Jianfei Cai, and Tsuhan Chen. ["An Empirical Study of Language CNN for Image Captioning."](https://arxiv.org/pdf/1612.07086.pdf) ICCV, 2017.
```
@article{gu2016recurrent,
  title={An Empirical Study of Language CNN for Image Captioning},
  author={Gu, Jiuxiang and Wang, Gang and Cai, Jianfei and Chen, Tsuhan},
  journal={ICCV},
  year={2017}
}
```
2. Jiuxiang Gu, Jianfei cai, Gang Wang, and Tsuhan Chen. ["stack-Captioning: Coarse-to-Fine Learning for Image Captioning."](https://arxiv.org/abs/1709.03376) arXiv preprint arXiv:1709.03376 (2017).
```
@article{gu2017stack_cap,
  title={Stack-Captioning: Coarse-to-Fine Learning for Image Captioning},
  author={Gu, Jiuxiang and Cai, Jianfei and Wang, Gang and Chen, Tsuhan},
  journal={arXiv preprint arXiv:1709.03376},
  year={2017}
}
```

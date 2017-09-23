#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: github.com/duinodu

from __future__ import print_function
import os
import argparse
import json
from PIL import Image

# lets download the annotations from http://mscoco.org/dataset/#download
def coco_preprocess():
    import os
    os.system('wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip')  # ~19MB
    os.system('unzip captions_train-val2014.zip')

    import json
    val = json.load(open('annotations/captions_val2014.json', 'r'))
    train = json.load(open('annotations/captions_train2014.json', 'r'))

    print(val.keys())
    print(val['info'])
    print(len(val['images']))
    print(len(val['annotations']))
    print(val['images'][0])
    print(val['annotations'][0])

    import json
    import os

    # combine all images and annotations together
    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    # for efficiency lets group annotations by image
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if not imgid in itoa: itoa[imgid] = []
        itoa[imgid].append(a)

    # create the json blob
    out = []
    for i, img in enumerate(imgs):
        imgid = img['id']

        # coco specific here, they store train/val images separately
        loc = 'train2014' if 'train' in img['file_name'] else 'val2014'

        jimg = {}
        jimg['file_path'] = os.path.join(loc, img['file_name'])
        jimg['id'] = imgid

        sents = []
        annotsi = itoa[imgid]
        for a in annotsi:
            sents.append(a['caption'])
        jimg['captions'] = sents
        out.append(jimg)

    json.dump(out, open('coco_raw.json', 'w'))

def ai_challenger_preprocess():
    import os
    import json
    val = json.load(open('/home/jxgu/github/chinese_im2text.pytorch/data/ai_challenger/ai_challenger_caption_validation_20170910/coco_caption_validation_annotations_20170910.json', 'r'))
    train = json.load(open('/home/jxgu/github/chinese_im2text.pytorch/data/ai_challenger/ai_challenger_caption_train_20170902/coco_caption_train_annotations_20170902.json', 'r'))

    print(val.keys())
    print(val['info'])
    print(len(val['images']))
    print(len(val['annotations']))
    print(val['images'][0])
    print(val['annotations'][0])

    import json
    import os

    # combine all images and annotations together
    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    # for efficiency lets group annotations by image
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if not imgid in itoa: itoa[imgid] = []
        itoa[imgid].append(a)

    # create the json blob
    out = []
    for i, img in enumerate(imgs):
        imgid = img['id']

        # coco specific here, they store train/val images separately
        loc = 'ai_challenger_caption_train_20170902' if 'train' in img['file_name'] else 'ai_challenger_caption_validation_20170910'
        #loc = 'all_images'

        jimg = {}
        jimg['file_path'] = os.path.join(loc, img['file_name'])
        jimg['id'] = imgid

        sents = []
        annotsi = itoa[imgid]
        for a in annotsi:
            sents.append(a['caption'])
        jimg['captions'] = sents
        out.append(jimg)

    json.dump(out, open('coco_ai_challenger_raw.json', 'w'))

def convert2coco(caption_json, img_dir):
    dataset = json.load(open(caption_json, 'r'))
    imgdir = img_dir

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()

    for ind, sample in enumerate(dataset):
        img = Image.open(os.path.join(imgdir, sample['image_id']))
        width, height = img.size

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = os.path.split(img_dir)[-1]+'/'+sample['image_id']
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        coco_img['id'] = ind

        coco_anno = {}
        coco_anno[u'image_id'] = ind
        coco_anno[u'id'] = ind
        coco_anno[u'caption'] = sample['caption']

        coco[u'images'].append(coco_img)
        coco[u'annotations'].append(coco_anno)

        print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join(os.path.dirname(caption_json), 'coco_'+os.path.basename(caption_json))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))

if __name__ == "__main__":
    train_caption_json = '/home/jxgu/github/chinese_im2text.pytorch/data/ai_challenger/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
    train_img_dir = '/home/jxgu/github/chinese_im2text.pytorch/data/ai_challenger/ai_challenger_caption_train_20170902/caption_train_images_20170902'
    val_caption_json = '/home/jxgu/github/chinese_im2text.pytorch/data/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
    val_img_dir = '/home/jxgu/github/chinese_im2text.pytorch/data/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_images_20170910'
    convert2coco(train_caption_json, train_img_dir)
    convert2coco(val_caption_json, val_img_dir)
    ai_challenger_preprocess()
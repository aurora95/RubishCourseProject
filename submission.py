from __future__ import absolute_import

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model import *
from PSDataset import PSDataset
from PIL import Image
from keras import preprocessing
from keras.preprocessing.image import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('model_path', help='.', default=None, type=str)
    parser.add_argument('--iterations', dest='iterations', default=10, type=int)

    parser.add_argument('--gpu', dest='gpus',
                        nargs='*',
                        help='GPU device id to use',
                        default=[0], type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

args = parse_args()
data_path = '/home/xing/.kaggle/competitions/plant-seedlings-classification/test/'
image_list = sorted(os.listdir(data_path))
datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last'
    )
train_dataset = PSDataset('/home/xing/.kaggle/competitions/plant-seedlings-classification/train', 'train', 0.9)
index_label_dict = train_dataset.get_index_label_info()
stat_mean = [0.485*255, 0.456*255, 0.406*255]
stat_std = [0.229*255, 0.224*255, 0.225*255]
iteraions = args.iterations

def get_model_prediction(model_path):
    weights_path = os.path.join(model_path, 'checkpoint_best.params')
    model_name = model_path.split('/')[1]
    net = globals()[model_name]()
    net.cuda()
    net = nn.DataParallel(net, device_ids=args.gpus)
    model_dict = torch.load(weights_path)
    net.load_state_dict(model_dict)
    cudnn.benchmark = True
    net.eval()
    
    pred_dict = {}
    for num, image_name in enumerate(image_list):
        print('{}/{}: {}'.format(num, len(image_list), image_name))
        image = Image.open(os.path.join(data_path, image_name))
        image = image.resize((320, 320), Image.BILINEAR)
        data = img_to_array(image, 'channels_last')[:, :, :3]
        pred_list = np.zeros((12,))
        for i in range(iteraions):
            transform_parameters = datagen.get_random_transform(img_shape=(320, 320))
            transformed_data = datagen.apply_transform(data, transform_parameters)
            transformed_data[..., 0] -= stat_mean[0]
            transformed_data[..., 1] -= stat_mean[1]
            transformed_data[..., 2] -= stat_mean[2]
            transformed_data[..., 0] /= stat_std[0]
            transformed_data[..., 1] /= stat_std[1]
            transformed_data[..., 2] /= stat_std[2]
            transformed_data = transformed_data.transpose(1, 2, 0).reshape((3, 320, 320))
            transformed_data = np.expand_dims(transformed_data, axis=0)
            input_data = torch.tensor(transformed_data).cuda()
            pred = net(input_data).detach().cpu().numpy()
            pred_list += np.squeeze(pred)
        final_pred = np.argmax(pred_list)
        pred_dict[image_name] = final_pred
    return pred_dict

pred_dict = get_model_prediction(args.model_path)

print('saving...')
image_list = [k for k in pred_dict.keys()]
sub = pd.DataFrame({'file': image_list, 'species': [index_label_dict[pred_dict[img]] for img in image_list]})
sub.to_csv('submission.csv', index=False, header=True)
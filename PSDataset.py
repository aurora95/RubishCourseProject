import os
import random
import numpy as np

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from keras import preprocessing
from keras.preprocessing.image import *


class PSDataset(Dataset):

    def __init__(self, root, state, train_ratio=0.8,
                 image_size=320,
                 rotation_range=180,
                 width_shift_range=0.1,
                 height_shift_range=0.1,
                 zoom_range=0.3,
                 horizontal_flip=True,
                 vertical_flip=True,
                 data_format='channels_last'):
        print('Init PSDataset {}...'.format(state))
        assert state in ['train', 'valid'], 'state error'
        self.root = root
        self.state = state
        self.image_size = image_size
        self.datainfo = {}
        self.image_path_label_tuple_list = []
        self.index_label_dict = {}
        class_list = sorted(os.listdir(self.root))
        assert len(class_list) == 12, '\'root\' should contain 12 subfolders, got {}'.format(len(class_list))
        for i, item in enumerate(class_list):
            self.datainfo.setdefault(item, [])
            self.index_label_dict[i] = item
            class_path = os.path.join(self.root, item)
            for image_name in sorted(os.listdir(class_path)):
                self.image_path_label_tuple_list.append((os.path.join(class_path, image_name), i))
        random.seed(42)
        random.shuffle(self.image_path_label_tuple_list)
        split_point = int(len(self.image_path_label_tuple_list) * train_ratio)
        self.train_list = self.image_path_label_tuple_list[:split_point]
        self.valid_list = self.image_path_label_tuple_list[split_point:]
        if state == 'train':
            print('train data num {}'.format(len(self.train_list)))
        if state == 'valid':
            print('valid data num {}'.format(len(self.valid_list)))
        self.datagen = ImageDataGenerator(rotation_range=rotation_range,
                                          width_shift_range=width_shift_range,
                                          height_shift_range=height_shift_range,
                                          zoom_range=zoom_range,
                                          horizontal_flip=horizontal_flip,
                                          vertical_flip=vertical_flip,
                                          data_format=data_format)
        self.stat_mean_std()

    def stat_mean_std(self):
        # mean_list = []
        # std_list = []
        # print('stat image ...')
        # pbar = tqdm(total=len(self.image_path_label_tuple_list))
        # for i, item in enumerate(self.image_path_label_tuple_list):
        #     image = Image.open(item[0])
        #     data = np.asarray(image, dtype=np.float32)[:, :, :3]
        #     mean_list.append(np.mean(data, axis=(0, 1)).reshape(1, -1))
        #     std_list.append(np.std(data, axis=(0, 1)).reshape(1, -1))
        #     print(np.shape(np.std(data, axis=(0, 1)).reshape(1, -1)))
        #     pbar.update(1)
        # pbar.close()
        # self.stat_mean = np.mean(np.concatenate(mean_list, axis=0), axis=0)
        # self.stat_std = np.mean(np.concatenate(std_list, axis=0), axis=0)
        self.stat_mean = [83.84732, 73.78362, 52.863792]
        self.stat_std = [23.832254, 24.810072, 27.171442]

    def get_index_label_info(self):
        return self.index_label_dict

    def __len__(self):
        return len(self.train_list) if self.state == 'train' else len(self.valid_list)

    def __getitem__(self, idx):
        data_list = self.train_list if self.state == 'train' else self.valid_list
        image = Image.open(data_list[idx][0])
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        data = img_to_array(image, 'channels_last')[:, :, :3]
        transform_parameters = self.datagen.get_random_transform(img_shape=(self.image_size, self.image_size))
        data = self.datagen.apply_transform(data, transform_parameters)
        data[..., 0] -= self.stat_mean[0]
        data[..., 1] -= self.stat_mean[1]
        data[..., 2] -= self.stat_mean[2]
        data[..., 0] /= self.stat_std[0]
        data[..., 1] /= self.stat_std[1]
        data[..., 2] /= self.stat_std[2]
        data = data.transpose(1, 2, 0).reshape((3, self.image_size, self.image_size))
        label = data_list[idx][1]
        return data, label


if __name__ == '__main__':
    train_dataset = PSDataset('./train', 'train', 0.8)
    print(train_dataset.__len__())
    print(train_dataset.get_index_label_info())
    valid_dataset = PSDataset('./train', 'valid', 0.8)
    print(valid_dataset.__len__())
    train_loader = DataLoader(dataset=train_dataset, batch_size=20, num_workers=4, shuffle=True)
    for i, (data, label) in enumerate(train_loader):
        print(np.shape(data))
        print(np.mean(data.numpy(), axis=(0, 2, 3)))
        print(np.std(data.numpy(), axis=(0, 2, 3)))
        print(np.shape(label))
        print(label)
        exit()

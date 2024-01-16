import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation, VisionDataset
import numpy as np


class BaseSemanticDataset(VisionDataset):
    """
    if you want to customize a new dataset to train the segmentation task,
    the img and mask file need be arranged as this sturcture.
        ├── data
        │   ├── my_dataset
        │   │   ├── img
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann
        │   │   │   ├── train
        │   │   │   │   ├── xxx{ann_suffix}
        │   │   │   │   ├── yyy{ann_suffix}
        │   │   │   │   ├── zzz{ann_suffix}
        │   │   │   ├── val
    """

    def __init__(self, metainfo, dataset_dir, transform, target_transform,
                 image_set='train',
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 data_prefix: dict = dict(img_path='img', ann_path='ann'),
                 return_dict=False):
        '''

        :param metainfo: meta data in original dataset, e.g. class_names
        :param dataset_dir: the path of your dataset, e.g. data/my_dataset/ by the stucture tree above
        :param image_set: 'train' or 'val'
        :param img_suffix: your image suffix
        :param ann_suffix: your annotation suffix
        :param data_prefix: data folder name, as the tree shows above, the data_prefix of my_dataset: img_path='img' , ann_path='ann'
        :param return_dict: return dict() or tuple(img, ann)
        '''
        super(BaseSemanticDataset, self).__init__(root=dataset_dir, transform=transform,
                                                  target_transform=target_transform)

        self.class_names = metainfo['class_names']
        self.img_path = os.path.join(dataset_dir, data_prefix['img_path'], image_set)
        self.ann_path = os.path.join(dataset_dir, data_prefix['ann_path'], image_set)
        print('img_folder_name: {img_folder_name}, ann_folder_name: {ann_folder_name}'.format(
            img_folder_name=self.img_path, ann_folder_name=self.ann_path))
        self.img_names = [img_name.split(img_suffix)[0] for img_name in os.listdir(self.img_path) if
                          img_name.endswith(img_suffix)]
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.return_dict = return_dict

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_names[index] + self.img_suffix))
        ann = Image.open(os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
        if self.transforms is not None:
            img, ann = self.transforms(img, ann)
        ann = np.array(ann)

        if self.return_dict:
            data = dict(img_name=self.img_names[index], img=img, ann=ann,
                        img_path=os.path.join(self.img_path, self.img_names[index] + self.img_suffix),
                        ann_path=os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
            return data
        return img, ann

    def __len__(self):
        return len(self.img_names)


class VOCSemanticDataset(Dataset):
    def __init__(self, root_dir, domain, transform, with_id=False, with_mask=False):
        super(VOCSemanticDataset, self).__init__()
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'
        self.xml_dir = self.root_dir + 'Annotations/'
        self.mask_dir = self.root_dir + 'SegmentationClass/'

        self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt' % domain).readlines()]
        self.transform = transform
        self.with_id = with_id
        self.with_mask = with_mask
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_mask:
            data_list.append(self.get_mask(image_id))

        return data_list


class TorchVOCSegmentation(VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        super(TorchVOCSegmentation, self).__init__(root=root, year=year, image_set=image_set, download=download,
                                                   transform=transform, target_transform=target_transform)
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target = np.array(target)
        return img, target


class TorchVOCTextSegmentation(VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        super().__init__(root=root, year=year, image_set=image_set, download=download,
                                                   transform=transform, target_transform=target_transform)
        self.class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.rs = np.random.RandomState(42)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, selected_class_name) where target is the image segmentation.
        """
        # print(f'[Get] i={index}, returning {self.images[index]}')
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        # only select class names that has value
        unique_labels = set(np.unique(np.array(target)))
        unique_labels.remove(255)
        # remove bg, add random label in case nothing's left
        if (0 in unique_labels):
            unique_labels.remove(0)
        if (len(unique_labels) == 0):
            unique_labels.add(self.rs.randint(1, 21))
        # print(unique_labels)
        selected_class_idx = self.rs.choice(list(unique_labels))
        selected_class_name = self.class_names[selected_class_idx]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target = (np.array(target) == selected_class_idx).astype(np.uint8)

        # print(img.shape, target.shape, target.dtype)

        # print(target.min(), target.max())
        # xx
        return img, target, selected_class_name

class TorchVOCTextSegmentationFull(VOCSegmentation):
    ''' full dataset without random sampling labels '''
    def __init__(
            self, root, year='2012', image_set='train', 
            *,
            download=False, transform=None, target_transform=None,
            filter_keywords=(), filter_class_idx=()
            ):
        super().__init__(root=root, year=year, image_set=image_set, download=download,
                                                   transform=transform, target_transform=target_transform)
        self.class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # self.rs = np.random.RandomState(42)

        if (filter_keywords is None):
            filter_keywords = ()
        self.filter_keywords = filter_keywords
        if (filter_class_idx is None):
            filter_class_idx = ()
        self.filter_class_idx = filter_class_idx

        self.build_index()
    
    def build_index(self):
        self.index_table = [] # (original_idx, label_idx)
        for original_index in range(super().__len__()):

            target = Image.open(self.masks[original_index])

            unique_labels = set(np.unique(np.array(target)))
            unique_labels.remove(255) # in VOC, 255 = invalid

            # filter by idx
            for remove_idx in self.filter_class_idx:
                if (remove_idx in unique_labels):
                    unique_labels.remove(remove_idx)
            
            # filter by keywords in class name
            unique_labels_filtered = []
            for unique_idx in unique_labels:
                label_text = self.class_names[unique_idx]
                has_keyword = False
                for keyword in self.filter_keywords:
                    if keyword in label_text:
                        has_keyword = True
                if (not has_keyword):
                    unique_labels_filtered.append(unique_idx)
            

            for unique_idx in unique_labels_filtered:
                self.index_table.append((original_index, unique_idx))
        
        print(f'index building done, {super().__len__()} -> {len(self)}')

    def __len__(self):
        return len(self.index_table)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, selected_class_name) where target is the image segmentation.
        """
        original_index, class_idx = self.index_table[index]
        # print(f'[Get] i={index}, returning {self.images[index]}')
        img = Image.open(self.images[original_index]).convert('RGB')
        target = Image.open(self.masks[original_index])

        selected_class_idx = class_idx
        selected_class_name = self.class_names[selected_class_idx]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target = (np.array(target) == selected_class_idx).astype(np.uint8)

        # print(img.shape, target.shape, target.dtype)

        # print(target.min(), target.max())
        # xx
        return img, target, selected_class_name


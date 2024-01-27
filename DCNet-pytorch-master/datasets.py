import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DrivetrainDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DrivetrainDataset, self).__init__()
        data_root = os.path.join(root, "train" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        manual_names = [i for i in os.listdir(os.path.join(data_root, "1st_manual"))]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i) for i in manual_names]


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 值为0和255
        mask = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)


# 下面两个函数是图片弄成batch批次、 FCN里面讲了
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

class DrivetestDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DrivetestDataset, self).__init__()
        self.flag = "aug" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        # os.listdir(path)中有一个参数，就是传入相应的路径，将会返回那个目录下的所有文件名
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        manual_names = [i for i in os.listdir(os.path.join(data_root, "1st_manual"))]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "mask"))]

        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i) for i in manual_names]
        self.roi_mask = [os.path.join(data_root, "mask",  i) for i in mask_names]


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        manual = Image.open(self.manual[idx]).convert('L')  # L表示灰度图片
        # 标签中感兴趣的从白色255变成1
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # mask中感兴趣的从白色255变成0
        roi_mask = 255 - np.array(roi_mask)

        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        # 将两者相加 np.clip设置上下限 mask前景区域也就是目标区域像素值是1，
        # 对于背景区域像素值是0 ，对于不感兴趣的区域像素值是255

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)
        from torchvision import datasets, transforms
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    # 下面两个函数是图片弄成batch批次、 FCN里面讲了
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



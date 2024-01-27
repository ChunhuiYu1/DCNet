import time
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import torch.utils.data
import transforms as T
from datasets import DrivetrainDataset
from model.Net import Net
from train_utils.train_and_eval import evaluate
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pad import InputPadder
from train_utils.disturtd_utils import ConfusionMatrix
import compute_mean_std
import os
from torchvision.utils import save_image

os.makedirs('output_images', exist_ok=True)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalizetrain(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main():
    classes = 1  # exclude background
    weights_path = "save_weights/best.pth"
    name = os.listdir(os.path.join("CHASEDB1/test/images"))
    os.makedirs('output_images', exist_ok=True)
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    num_classes = 1
    mean1, std1 = compute_mean_std.compute("CHASEDB1/test/images")
    # get devices
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # create model
    model = Net(in_channels=3, num_classes=num_classes, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # model.load_state_dict(torch.load(weights_path)['model'])
    model.to(device)
    val_dataset = DrivetrainDataset('CHASEDB1',
                                    train=False,
                                    transforms=SegmentationPresetEval(mean1, std1))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=1,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    acc, se, sp, F1, mIou, pr, AUC_ROC = evaluate(model, val_loader, device=device, num_classes=num_classes, name=name)
    print(f"AUC: {AUC_ROC:.6f}")
    print(f"acc: {acc:.6f}")
    print(f"se: {se:.6f}")
    print(f"sp: {sp:.6f}")
    print(f"mIou: {mIou:.6f}")
    print(f"F1: {F1:.6f}")


def evaluate(model, data_loader, device, num_classes, name):
    model.eval()
    confmat = ConfusionMatrix(num_classes + 1)
    data_loader = tqdm(data_loader)
    mask = None
    predict = None
    with torch.no_grad():
        for index, (image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)  # 传入cuda
            # 对图像进行填充，使得尺寸能够被16整除
            padder = InputPadder(image.shape, num=8)
            image, target = padder.pad(image, target)
            image, target = image.to(device), target.to(device)
            # (B,1,H,W)
            output = model(image)
            truth = output.clone()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            confmat.update(target.flatten(), output.long().flatten())

            save_image(output, os.path.join('output_images', str(name[index])))
            # dice.update(output, target)
            mask = target.flatten() if mask is None else torch.cat((mask, target.flatten()))
            # 它是概率集合，不能是0，1集合
            predict = truth.flatten() if predict is None else torch.cat((predict, truth.flatten()))

    mask = mask.cpu().numpy()
    predict = predict.cpu().numpy()
    assert mask.shape == predict.shape, f"维度不对"
    AUC_ROC = roc_auc_score(mask, predict)

    return confmat.compute()[0], confmat.compute()[1], confmat.compute()[2], confmat.compute()[3], confmat.compute()[
        4], confmat.compute()[
        5], AUC_ROC


if __name__ == '__main__':
    main()

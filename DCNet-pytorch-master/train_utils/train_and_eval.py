import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from .pad import InputPadder
from .disturtd_utils import ConfusionMatrix
from .dice_cofficient_loss import dice_loss
from .Focal_loss import FocalLoss
from .EGloss import EGloss


def criterion(inputs, target, dice: bool = True, bce: bool = True, pib: bool = True):
    loss1 = 0
    if dice:
        loss1 = dice_loss(inputs, target)
    loss2 = 0
    target = target.unsqueeze(1).float()
    if bce:
        loss2 = nn.BCELoss()(inputs, target)
    loss3 = 0
    if pib:
        loss3 = EGloss(inputs, target)

    return loss1 + loss2 + loss3


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes + 1)
    data_loader = tqdm(data_loader)
    mask = None
    predict = None
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            padder = InputPadder(image.shape,num=8)
            image, target = padder.pad(image, target)
            image, target = image.to(device), target.to(device)
            output = model(image)
            truth = output.clone()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            confmat.update(target.flatten(), output.long().flatten())
            mask = target.flatten() if mask is None else torch.cat((mask, target.flatten()))
            predict = truth.flatten() if predict is None else torch.cat((predict, truth.flatten()))

    mask = mask.cpu().numpy()
    predict = predict.cpu().numpy()
    assert mask.shape == predict.shape, f"维度不对"
    AUC_ROC = roc_auc_score(mask, predict)

    return confmat.compute()[0], confmat.compute()[1], confmat.compute()[2], confmat.compute()[3], confmat.compute()[
        4], confmat.compute()[
        5], AUC_ROC


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler,
                    scaler=None):
    model.train()
    total_loss = 0

    data_loader = tqdm(data_loader)  # 显示加载进度条
    for image, target in data_loader:
        image, target = image.to(device), target.to(device)  # 传入cuda
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # output的输出通道为1,已经使用了sigmoid层，输出的是概率图
            # loss2 = criterion(thin1, target_thin, True, True, False)
            loss = criterion(output, target, True, False, True)

        total_loss += loss.item()

        data_loader.set_description(f"Epoch[{epoch}/250]-train,train_loss:{loss.item()}")
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # chasedb1
        scheduler.step()
    return total_loss / len(data_loader)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

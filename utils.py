import argparse
from configs import *
from resnets import rf_lw50, rf_lw101, rf_lw152
import torch
import torch.nn as nn
import os
import time
from aux_utils import AverageMeter
import cv2
from miou_utils import compute_iu, fast_cm


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    # Dataset
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR,
                        help="Path to the training set directory.")
    parser.add_argument("--val_dir", type=str, default=VAL_DIR,
                        help="Path to the validation set directory.")
    parser.add_argument("--train_list", type=str, nargs='+', default=TRAIN_LIST,
                        help="Path to the training set list.")
    parser.add_argument("--val_list", type=str, nargs='+', default=VAL_LIST,
                        help="Path to the validation set list.")
    parser.add_argument("--shorter_side", type=int, nargs='+', default=SHORTER_SIDE,
                        help="Shorter side transformation.")
    parser.add_argument("--crop_size", type=int, nargs='+', default=CROP_SIZE,
                        help="Crop size for training,")
    parser.add_argument("--normalise_params", type=list, default=NORMALISE_PARAMS,
                        help="Normalisation parameters [scale, mean, std],")
    parser.add_argument("--batch_size", type=int, nargs='+', default=BATCH_SIZE,
                        help="Batch size to train the segmenter model.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument("--num_classes", type=int, nargs='+', default=NUM_CLASSES,
                        help="Number of output classes for each task.")
    parser.add_argument("--low_scale", type=float, nargs='+', default=LOW_SCALE,
                        help="Lower bound for random scale")
    parser.add_argument("--high_scale", type=float, nargs='+', default=HIGH_SCALE,
                        help="Upper bound for random scale")
    parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL,
                        help="Label to ignore during training")

    # Encoder
    parser.add_argument("--enc", type=str, default=ENC,
                        help="Encoder net type.")
    parser.add_argument("--enc_pretrained", type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument("--evaluate", type=bool, default=EVALUATE,
                        help='If true, only validate segmentation.')
    parser.add_argument("--freeze_bn", type=bool, nargs='+', default=FREEZE_BN,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument("--num_segm_epochs", type=int, nargs='+', default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for segmentation network.')
    parser.add_argument("--print_every", type=int, default=PRINT_EVERY,
                        help='Print information every often.')
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Path to directory for storing checkpoints.")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH,
                        help="Path to the checkpoint file.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to the model file.")
    parser.add_argument("--val_every", nargs='+', type=int, default=VAL_EVERY,
                        help="How often to validate current architecture.")

    # Optimisers
    parser.add_argument("--lr_enc", type=float, nargs='+', default=LR_ENC,
                        help="Learning rate for encoder.")
    parser.add_argument("--lr_dec", type=float, nargs='+', default=LR_DEC,
                        help="Learning rate for decoder.")
    parser.add_argument("--mom_enc", type=float, nargs='+', default=MOM_ENC,
                        help="Momentum for encoder.")
    parser.add_argument("--mom_dec", type=float, nargs='+', default=MOM_DEC,
                        help="Momentum for decoder.")
    parser.add_argument("--wd_enc", type=float, nargs='+', default=WD_ENC,
                        help="Weight decay for encoder.")
    parser.add_argument("--wd_dec", type=float, nargs='+', default=WD_DEC,
                        help="Weight decay for decoder.")
    parser.add_argument("--optim_dec", type=str, default=OPTIM_DEC,
                        help="Optimiser algorithm for decoder.")
    return parser.parse_args()


def create_segmenter(net, pretrained, num_classes, model_path=MODEL_PATH):
    """Create Encoder; for now only ResNet [50,101,152]"""
    if str(net) == '50':
        return rf_lw50(num_classes, model_path, pretrained=pretrained)
    elif str(net) == '101':
        return rf_lw101(num_classes, model_path, pretrained=pretrained)
    elif str(net) == '152':
        return rf_lw152(num_classes, model_path, pretrained=pretrained)
    else:
        raise ValueError("{} is not supported".format(str(net)))


def create_loaders(
        train_dir, val_dir, train_list, val_list,
        shorter_side, crop_size, low_scale, high_scale,
        normalise_params, batch_size, num_workers, ignore_label
):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from datasets import NYUDataset as Dataset
    from datasets import Pad, RandomCrop, RandomMirror, ResizeShorterScale, ToTensor, Normalise

    ## Transformations during training ##
    composed_trn = transforms.Compose([ResizeShorterScale(shorter_side, low_scale, high_scale),
                                       Pad(crop_size, [123.675, 116.28, 103.53], ignore_label),
                                       RandomMirror(),
                                       RandomCrop(crop_size),
                                       Normalise(*normalise_params),
                                       ToTensor()])
    composed_val = transforms.Compose([Normalise(*normalise_params),
                                       ToTensor()])
    ## Training and validation sets ##
    trainset = Dataset(data_file=train_list,
                       data_dir=train_dir,
                       transform_trn=composed_trn,
                       transform_val=composed_val)

    valset = Dataset(data_file=val_list,
                     data_dir=val_dir,
                     transform_trn=None,
                     transform_val=composed_val)
    print(" Created train set = {} examples, val set = {} examples".format(len(trainset), len(valset)))
    ## Training and validation loaders ##
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(valset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader


def create_optimisers(
        lr_enc, lr_dec,
        mom_enc, mom_dec,
        wd_enc, wd_dec,
        param_enc, param_dec,
        optim_dec):
    """Create optimisers for encoder, decoder and controller"""
    optim_enc = torch.optim.SGD(param_enc, lr=lr_enc, momentum=mom_enc,
                                weight_decay=wd_enc)
    if optim_dec == 'sgd':
        optim_dec = torch.optim.SGD(param_dec, lr=lr_dec,
                                    momentum=mom_dec, weight_decay=wd_dec)
    elif optim_dec == 'adam':
        optim_dec = torch.optim.Adam(param_dec, lr=lr_dec, weight_decay=wd_dec, eps=1e-3)
    return optim_enc, optim_dec


def load_ckpt(ckpt_path, ckpt_dict):
    best_val = epoch_start = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k])
        best_val = ckpt.get('best_val', 0)
        epoch_start = ckpt.get('epoch_start', 0)
        print(" Found checkpoint at {} with best_val {:.4f} at epoch {}".format(
            ckpt_path, best_val, epoch_start
        ))
    return best_val, epoch_start


def train_segmenter(
        segmenter, train_loader, optim_enc, optim_dec,
        epoch, segm_crit, freeze_bn, print_every):
    """Training segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      train_loader (DataLoader) : training data iterator
      optim_enc (optim) : optimiser for encoder
      optim_dec (optim) : optimiser for decoder
      epoch (int) : current epoch
      segm_crit (nn.Loss) : segmentation criterion
      freeze_bn (bool) : whether to keep BN params intact

    """
    train_loader.dataset.set_stage('train')
    segmenter.train()
    if freeze_bn:
        for m in segmenter.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    for i, sample in enumerate(train_loader):
        start = time.time()
        input = sample['image'].cuda()
        target = sample['mask'].cuda()
        input_var = torch.autograd.Variable(input).float()
        target_var = torch.autograd.Variable(target).long()
        # Compute output
        output = segmenter(input_var)
        # print("test")
        # print(output.size())
        # print(target_var.size())
        output = nn.functional.interpolate(output, size=target_var.size()[1:], mode='bilinear', align_corners=False)
        # print(output.size())
        soft_output = nn.LogSoftmax()(output)
        # Compute loss and backpropagate
        loss = segm_crit(soft_output, target_var)
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        loss.backward()
        optim_enc.step()
        optim_dec.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)
        if i % print_every == 0:
            print(' Train epoch: {} [{}/{}]\t'
                  'Avg. Loss: {:.3f}\t'
                  'Avg. Time: {:.3f}'.format(
                epoch, i, len(train_loader),
                losses.avg, batch_time.avg
            ))


def validate(segmenter, val_loader, epoch, num_classes=-1, print_every=PRINT_EVERY):
    """Validate segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      val_loader (DataLoader) : training data iterator
      epoch (int) : current epoch
      num_classes (int) : number of classes to consider
      print_every
    Returns:
      Mean IoU (float)
    """
    val_loader.dataset.set_stage('val')
    segmenter.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            start = time.time()
            input = sample['image']
            # print(input.size())
            target = sample['mask']
            input_var = torch.autograd.Variable(input).float().cuda()
            # print(input_var.size())
            # Compute output
            output = segmenter(input_var)
            output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                target.size()[1:][::-1],
                                interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
            # Compute IoU
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes  # Ignore every class index larger than the number of classes
            cm += fast_cm(output[gt_idx], gt[gt_idx], num_classes)

            if i % print_every == 0:
                print(' Val epoch: {} [{}/{}]\tMean IoU: {:.3f}'.format(
                    epoch, i, len(val_loader),
                    compute_iu(cm).mean()
                ))

    ious = compute_iu(cm)
    print(" IoUs: {}".format(ious))
    miou = np.mean(ious)
    print(' Val epoch: {}\tMean IoU: {:.3f}'.format(
        epoch, miou))
    return miou


def test_img(segmenter, img_path, num_classes, normalise_params):
    from PIL import Image
    from datasets import Normalise_test, ToTensor_test

    img_arr = np.array(Image.open(img_path))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    SCALE_temp = normalise_params[0]
    MEAN_temp = normalise_params[1]
    STD_temp = normalise_params[2]
    img_arr = Normalise_test(SCALE_temp, MEAN_temp, STD_temp, img_arr)
    img = ToTensor_test(img_arr)

    with torch.no_grad():
        input_var = torch.autograd.Variable(img).float().cuda()
        input_var = input_var.unsqueeze(0)
        output = segmenter(input_var)
        output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                            img_arr.shape[:2][::-1],
                            interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
    output_3d = np.zeros(img_arr.shape, dtype=np.uint8)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if output[i][j] < num_classes:
                for z in range(3):
                    output_3d[i, j, z] = palette[output[i][j]][z]
    return output_3d

from torch.utils.data import DataLoader
from geoseg.losses import *
import torch.nn as nn
from geoseg.datasets.UVdataset import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils
# from geoseg.models.WavefuseNet import WavefuseNet
# from geoseg.models.segformer import SegFormer
from geoseg.models.UANet import UANet_VGG
# from geoseg.models.UVSENet import UVNet
# training hparam
max_epoch = 300
ignore_index = 255
train_batch_size = 6
val_batch_size = 6
lr = 1e-3
weight_decay = 0.0025
backbone_lr = 1e-3
backbone_weight_decay = 0.0025
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES


model_infos = {
    # vgg16_bn, resnet50, resnet18
    'backbone': 'resnet50',
    'pretrained': True,
    'out_keys': ['block1','block2','block3' ,'block4'],
    'in_channel': 64,
    'n_classes': 2,
    'encoder_pos': True,
    'decoder_pos': True,
    'model_pattern': ['X', 'A', 'S', 'C'],

    'BATCH_SIZE': 12,
    'IS_SHUFFLE': True,
    'NUM_WORKERS': 0,
    'DATASET': 'generate_dep_info/train_data.csv',
    'model_path': 'Checkpoints',
    'log_path': 'Results',
    # if you need the validation process.
    'IS_VAL': True,
    'VAL_BATCH_SIZE': 12,
    'VAL_DATASET': 'generate_dep_info/val_data.csv',
    # if you need the test process.
    'IS_TEST': True,
    'TEST_DATASET': 'generate_dep_info/test_data.csv',
    'IMG_SIZE': [256, 256],
    'PHASE': 'seg',

#         # China Dataset
#         'PRIOR_MEAN': [0.3761870736217567, 0.38140236772409364, 0.390793712429784],
#         'PRIOR_STD': [0.03322610681388147, 0.030094983534503315, 0.02801492565287637],
    # # # WHU Dataset
    'PRIOR_MEAN': [0.4380193123577935, 0.44732773473491755, 0.41931856414745494],
    'PRIOR_STD': [0.02739904138457403, 0.026904002077546566, 0.028657592029524774],

    # if you want to load state dict
#         'load_checkpoint_path': 'Checkpoints/best_model.pt',
    'load_checkpoint_path': '',
    # if you want to resume a checkpoint
    'resume_checkpoint_path': '',
}

weights_name = "UANet_VGG"
weights_path = "model_weights/UVData/{}".format(weights_name)
test_weights_name = "UANet_VGG"
log_name = 'UVData/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
# pretrained_ckpt_path = 'model_weights/UVData/UNet/UNet.ckpt'
pretrained_ckpt_path = None
resume_ckpt_path = None

# net = UVNet(64,2,True)
# net = SegFormer(num_classes=2, phi="b2", pretrained=False)
# net = WavefuseNet(**model_infos)
net = UANet_VGG(channel=32,num_classes=num_classes)
# define the loss

loss = nn.CrossEntropyLoss()
use_aux_loss = False

# define the dataloader

train_dataset = UVDataset(data_root='/root/autodl-tmp/data/UVData/train', mode='train', mosaic_ratio=0.25, transform=train_aug)
val_dataset = UVDataset(data_root='/root/autodl-tmp/data/UVData/val', mode='val', transform=val_aug)
test_dataset = UVDataset(data_root='/root/autodl-tmp/data/UVData/test', mode='test', transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-4)

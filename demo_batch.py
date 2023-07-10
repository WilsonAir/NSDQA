import logging
import sys, os

sys.path.append('.')
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import OrderedDict
from utils.metrics import accuracy, measure_bers, measure_bers_for_ori, AverageMeter, IQAPerformance
import timm
# from inplace_abn import InPlaceABNSync
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

need_inv = True
need_rank = False


class CASC_C(nn.Module):
    def __init__(self, backbone='tf_mobilenetv3_small_100'):
        super(CASC_C, self).__init__()
        self.backbone = timm.models.create_model(backbone, features_only=True, pretrained=True, in_chans=5).cuda()
        # self.backbone_2 = timm.models.create_model(backbone, features_only=True, pretrained=True, in_chans=5).cuda()
        channel_config = self.backbone.feature_info.channels()
        self.channel_config = channel_config

        self.layer_fuse = nn.AvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.channel_config[-1] * 3, out_features=4096, bias=True),
            # nn.Linear(in_features=544, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048, out_features=6, bias=True)
        )

    def forward(self, input, detections, refference):

        for i in range(detections.shape[1]):
            detections_temp = detections[:, i:i + 1, :, :]
            fuse_input = torch.cat((detections_temp, refference, input), dim=1)
            features_A = self.backbone(fuse_input)

            if i == 0:
                features_fuse = features_A[-1]
            else:
                features_fuse = torch.cat((features_fuse, features_A[-1]), dim=1)

        features_fuse = self.layer_fuse(features_fuse)
        features = torch.flatten(features_fuse, 1)

        scores_mul = self.classifier(features)
        ber = scores_mul[:, :3]
        mae = scores_mul[:, 3:]
        return ber, mae


def eval_net(net_D, net_G, loader, device, args=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net_D.eval()
    # net_G.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion_reg = torch.nn.MSELoss()
    # criterion_inter = InterDistanceLoss()
    criterion_rank = nn.MarginRankingLoss()

    losses_reg_m = AverageMeter()
    loss_out_m = AverageMeter()
    loss_ber_m = AverageMeter()
    loss_mae_m = AverageMeter()
    loss_inter_m = AverageMeter()
    top1_m = AverageMeter()

    evaluator_A = IQAPerformance()
    evaluator_B = IQAPerformance()

    selection_eval = {'shadow': 0, 'nonshadow': 0, 'ber': 0, 'Np': 0, 'Nn': 0, 'Tp': 0, 'Tn': 0}

    dir_checkpoint = 'ckpt/' + args.ablation_arch + '/'
    logdir3 = dir_checkpoint + f'{args.arch}' + '_ber_results.txt'
    writer_res = open(logdir3, 'a')

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for input, target in loader:
            with torch.no_grad():
                input = input.cuda()
                predictions = target['detection_results_crf_for_train'].cuda(non_blocking=True).float()
                mask = target['mask_resized'].cuda(non_blocking=True).float()
                mask_size = target['ori_size']
                bers = target['bers'].cuda(non_blocking=True).float()

                # bers = (bers - ber_label_MEAN) / ber_label_STD

                maes = target['maes'].cuda(non_blocking=True).float()

                label_ber = target['label_ber'].cuda().long()
                cls_label = label_ber

                predictions_ori = target['detection_results_crf'].cuda().float()
                mask_ori = target['mask'].cuda().float()

                imgs = input.to(device=device, dtype=torch.float32)

                loss = 0
                x_reg_mul = None
                x_reg_mul_second = None
                pre_b = 0

                loss_rank = 0
                loss_ber = 0
                loss_mae = 0

                reg_label_A = bers
                reg_label_B = maes

                prediction_sum = torch.sum(predictions, dim=1)
                prediction_intreaction = (prediction_sum >= (0.5 * 3)) * 1.0
                prediction_intreaction = prediction_intreaction.unsqueeze(1).cuda()
                pre_fp = target['fp'].cuda(non_blocking=True).float()
                pre_fn = target['fn'].cuda(non_blocking=True).float()
                detections_revise_fake = prediction_intreaction - prediction_intreaction * pre_fp + (
                        1 - prediction_intreaction) * pre_fn

                imgs = input.to(device=device, dtype=torch.float32)

                pre_b, pre_m = net_D(imgs, predictions, detections_revise_fake)

                loss_ber = criterion_reg(pre_b, reg_label_A)
                loss_mae = criterion_reg(pre_m, reg_label_B)

                output_ber = pre_b  # + x_reg_mul_second) / 2
                # loss_out = criterion_reg(output_ber, bers)
                # loss_inter = criterion_inter(output_ber, bers)
                #
                loss = loss_ber + loss_mae  # + loss_rank + loss_inter
                losses_reg_m.update(loss, input.size(0))
                # loss_out_m.update(loss_out.item(), input.size(0))
                # loss_inter_m.update(loss_inter.item(), input.size(0))
                # loss_mae_m.update(loss_mae.item(), input.size(0))
                loss_ber_m.update(loss_ber.item(), input.size(0))

                output = 100 * torch.ones(output_ber.shape, dtype=output_ber.dtype,
                                          device=output_ber.device) - output_ber
                acc1, acc5 = accuracy(output, cls_label, topk=(1, predictions.shape[1]))
                top1_m.update(acc1.item(), output.size(0))
                selection_eval = measure_bers(input, output, predictions_ori, mask_ori, mask_size, selection_eval)
                evaluator_A.update(pred=output_ber, y=bers)

                writer_res.writelines('{},{},{},{} \n'.format(
                    'eval', target['image_name'], output_ber, bers))
                writer_res.flush()
            # evaluator_B_result = evaluator_B.compute()
            pbar.update()

    writer_res.close()
    srocc, krocc, plcc, rmse, mae = evaluator_A.compute()
    # net_G.train()
    net_D.train()

    return OrderedDict(
        [('Dice_coff', tot / n_val),
         ('ber', selection_eval['ber']),
         ('shadow', selection_eval['shadow']),
         ('nonshadow', selection_eval['nonshadow']),
         ('loss_reg', losses_reg_m.avg),
         ('loss_out', loss_out_m.avg),
         ('loss_inter', loss_inter_m.avg),
         ('loss_ber', loss_ber_m.avg),
         ('loss_mae', loss_mae_m.avg),
         ('srocc', srocc),
         ('krocc', krocc),
         ('plcc', plcc),
         ('rmse', rmse),
         ('mae', mae),
         # ('evaluator_B_result', evaluator_B_result),
         ('top1', top1_m.avg)]
    )


def train_net(net_D,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              net_G=None,
              img_scale=0.5):
    dir_checkpoint = 'ckpt/' + args.ablation_arch + '/'

    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    # For shadow
    from data.shadow_dataloader import dataset_init
    test_loader = dataset_init(args)

    # n_train = len(train_loader)
    n_val = len(test_loader)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    val_score = eval_net(net_D, net_G, test_loader, device, args)
    print(val_score)
    return 0



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    from config import get_args

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net_D = CASC_C(backbone=args.backbone)
    # net_G = SegNet_v2(n_channels=4, n_classes=1)

    # net_G.load_state_dict(torch.load('checkpoints/SBU_SegNet_v2/CP_epoch111.pth', map_location=device), strict=False)

    if torch.cuda.device_count() > 1:
        net_D = net_D.cuda()
        net_D = nn.DataParallel(net_D)
    else:
        net_D.to(device=device)
        # net_G.to(device=device)

    if args.load:
        net_D.load_state_dict(torch.load(args.load, map_location=device), strict=False)
        logging.info(f'Model loaded from {args.load}')
    try:
        train_net(net_D=net_D,
                  # net_G=net_G,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net_D.state_dict(), '../INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
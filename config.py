import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('--local_rank', type=int)

    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--lr_policy', default='poly', type=str, help='Learning rate step policy.')
    parser.add_argument('--lr_stepsize', default=1000, type=int, help='Learning rate step size.')
    parser.add_argument('--lr_gamma', default=99e-2, type=int, help='Learning rate step size.')

    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    parser.add_argument("--eval_to_list", default=False, type=bool)
    parser.add_argument("--need_fp_fn", default=False, type=bool)

    parser.add_argument("--multi_path", default=2, type=int)
    parser.add_argument("--data", default='/media/wilson/Wilson/DE/Data/', type=str)
    parser.add_argument("--dataset", default='CUHK', type=str)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--workers", default=8, type=int)

    parser.add_argument("--arch", default='Demo', type=str)
    parser.add_argument("--generator", action='store_true', default=False)
    parser.add_argument("--seperate", action='store_true', default=False)

    parser.add_argument("--train_eva_only", action='store_true', default=False)
    parser.add_argument("--device", default='2080', type=str)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--eval_metric", type=str, default='ber', choices=['ber', 'shadow', 'nonshadow', 'iou', 'mae'],
                        help='try to use deferent metric')

    parser.add_argument("--ablation_arch", type=str, help='what are you verifying for')

    parser.add_argument("--eval", action='store_true', default=False)

    # for Swin transformer config files
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to transformer config file', )
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+',
                        )

    # for vit
    parser.add_argument('--model_type', default='R50-ViT-B_16', type=str)
    parser.add_argument('--pretrained_dir', default='models/IQT/imagenet21k+imagenet2012_R50+ViT-B_16.npz', type=str)

    # for PIQT
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='efficientnet_b3', type=str,
                        help="Name of the convolutional backbone to use",
                        choices=['resnext101_32x8d', 'resnext50_32x4d', 'resnet50', 'inception_v3',
                                 'efficientnetv2_rw_m', 'efficientnetv2_rw_s', 'efficientnetv2_rw_t',
                                 'tf_efficientnetv2_s_in21ft1k', 'tf_mobilenetv3_small_100', 'tf_mobilenetv3_large_100'])
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    #
    # * Transformer
    parser.add_argument('--enc_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--lr_backbone', default=1e-5, type=float, help="Learn rate of feature extract backbone, if > 0 than train")
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # for PIQT end
    return parser.parse_args()
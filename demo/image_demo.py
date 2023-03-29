# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='11.6_2600.jpg', help='Image file')
    parser.add_argument('--config', default='/home/zzw/SegNeXt-main/tools/work_dirs/segnext.large.512x512.coco_trash.160k/segnext.large.512x512.coco_trash.160k.py', help='Config file')
    parser.add_argument('--checkpoint', default='/home/zzw/SegNeXt-main/tools/work_dirs/segnext.large.512x512.coco_trash.160k/best_mIoU_iter_160000.pth', help='Checkpoint file')
    parser.add_argument('--out-file', default='11.6_2600_result.jpg', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='my_dataset',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity,
        out_file=args.out_file)


if __name__ == '__main__':
    main()

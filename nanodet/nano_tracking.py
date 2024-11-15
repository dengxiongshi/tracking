# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results

# 定义一个回调函数，用于初始化对象跟踪器
def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    # 检查所选择的跟踪方法是否在支持的跟踪方法列表中
    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    # 构建跟踪配置文件的路径
    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []

    # 初始化指定数量的对象跟踪器
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # 如果跟踪器具有模型，则执行模型的预热
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers      # 将初始化的跟踪器保存到预测器对象中


# 主要的处理函数，用于运行目标检测和跟踪
@torch.no_grad()
def run(args):
    # 创建 YOLO 模型，yolo.predictor 是预测器对象
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    # 对视频流或图像序列执行目标检测和跟踪
    results = yolo.track(
        source=args.source, # 输入视频流或图像序列的路径
        conf=args.conf, # 置信度阈值
        iou=args.iou,   # IoU（交并比）阈值，用于非极大值抑制（NMS）
        show=args.show, # 是否显示跟踪结果
        stream=True,    # 是否在流式模式下运行
        device=args.device, # 使用的计算设备，可以是GPU或CPU
        show_conf=args.show_conf,    # 是否显示目标置信度
        save_txt=args.save_txt,     # 是否将跟踪结果保存为文本文件
        show_labels=args.show_labels,   # 是否显示标签
        save=args.save,     # 是否保存跟踪结果
        verbose=args.verbose,   # 是否打印详细信息
        exist_ok=args.exist_ok, # 是否允许覆盖已存在的结果
        project=args.project,   # 结果保存的项目名称
        name=args.name,         # 结果保存的名称
        classes=args.classes,    # 要过滤的目标类别
        imgsz=args.imgsz,         # 推理图像的大小
        vid_stride=args.vid_stride, # 视频帧率步长
        line_width=args.line_width  # 框的线宽
    )

    # 在预测开始时添加一个回调函数，用于初始化跟踪器
    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # 如果不使用 yolov8 模型，根据 yolo_model 参数加载另一个目标检测模型
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # 将自定义参数保存到预测器对象中
    yolo.predictor.custom_args = args

    # 遍历每一帧的检测和跟踪结果
    for frame_idx, r in enumerate(results):
        # 如果检测框数据中包含7列数据
        if r.boxes.data.shape[1] == 7:

            # 根据源类型或源文件格式创建 MOT 结果的保存路径
            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            # 如果设置了保存 MOT 结果的选项，则将结果写入文件
            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

             # 如果设置了保存目标裁剪图像的选项，则将每个目标的裁剪图像保存到相应的文件夹中
            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                        BGR=True
                    )

    # 如果设置了保存 MOT 结果的选项，打印结果保存路径
    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default=r'D:\animal_project\yolo_tracking\examples\mp4\detect1.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=1, type=int,
                        help='video frame-rate stride')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)

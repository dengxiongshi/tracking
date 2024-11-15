import os
import sys

from tracker.timer import Timer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracker'))


import cv2
import time
import torch
import argparse
import platform
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadScreenshots
from utils.general import (LOGGER, Profile, check_file, check_img_size,
                           check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                           scale_boxes, strip_optimizer, xyxy2xywh, xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync, smart_inference_mode

from tracker.basetrack import BaseTracker  # for framework
from tracker.deepsort import DeepSORT
from tracker.bytetrack import ByteTrack
from tracker.deepmot import DeepMOT
from tracker.botsort import BoTSORT
from tracker.c_biou_tracker import C_BIoUTracker
from tracker.uavmot import UAVMOT
from tracker.strongsort import StrongSORT

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ---------------Object Tracking---------------
import skimage

# -----------Object Blurring-------------------
blurratio = 40

# .................. Tracker Functions .................
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


TRACKER_DICT = {
        'sort': BaseTracker,
        'deepsort': DeepSORT,
        'bytetrack': ByteTrack,
        'deepmot': DeepMOT,
        'botsort': BoTSORT,
        'uavmot': UAVMOT,
        'strongsort': StrongSORT,
        'c_biou': C_BIoUTracker,
    }  # dict for trackers, key: str, value: class(BaseTracker)

class Opts:
    pass


def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


"""" Calculates the relative bounding box from absolute pixel values. """


def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, identities=None, categories=None,
               names=None, color_box=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        [255, 255, 255], 1)
            cv2.circle(img, data, 3, color, -1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 191, 0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255, 191, 0), -1)
    return img

def plot_img(img, results):
    """
    img: np.ndarray: (H, W, C)
    frame_id: int
    results: [tlwhs, ids, clses]
    save_dir: sr

    plot images with bboxes of a seq
    """
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    for tlwh, id, cls in zip(tlwhs, ids, clses):
        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(img, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'id: {id}'
        cv2.putText(img, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=(255, 164, 0), thickness=2)

    # cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)
    return img


# ..............................................................................


@smart_inference_mode()
def run(
        tracker='c_biou',
        reid_model_path='',
        dhn_path='',
        track_buffer=30,
        gamma=0.1,
        nms_thresh=0.7,
        kalman_format='',
        min_area=150,
        detect_per_frame=1,
        track_eval=True,

        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        blur_obj=False,
        color_box = False
):
    # NOTE: ATTENTION: make kalman and tracker compatible
    if tracker == 'botsort':
        kalman_format = 'botsort'
    elif tracker == 'strongsort':
        kalman_format = 'strongsort'

    opts = Opts()
    setattr(opts, 'img_size', imgsz)
    setattr(opts, 'conf_thresh', conf_thres)
    setattr(opts, 'iou_thresh', iou_thres)
    setattr(opts, 'track_buffer', track_buffer)
    setattr(opts, 'kalman_format', kalman_format)
    setattr(opts, 'reid_model_path', reid_model_path)
    setattr(opts, 'dhn_path', dhn_path)

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # .... Initialize SORT ....
    track = TRACKER_DICT[tracker](opts, frame_rate=30, gamma=gamma)

    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    seq_fps = []
    frame_id = 0
    average_time = []
    timer = Timer()
    t0 = time.time()

    bbox = [858, 479, 115, 165]
    bbox = xywh2xyxy(np.array(bbox))
    conf_cls = np.array([0, 0])
    bbox = np.append(bbox, conf_cls)
    select_object = True

    for path, im, im0s, vid_cap, s in dataset:
        timer.tic()  # start timing this img

        string_ = f'----------processing frame {frame_id}----------'

        t1 = time_sync()

        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)
        save_path = str(save_dir / p.name)
        s += '%gx%g ' % im.shape[2:]

        t4 = time_sync()
        # Run SORT
        if select_object:
            current_tracks = track.update(bbox, im0)
            select_object = False
        else:
            current_tracks = track.update_without_detection(None, im0)
        t5 = time_sync()
        string_ += 'tracker time: ' + str(round(t5 - t4, 3)) + 'ms  '

        # save results
        cur_tlwh, cur_id, cur_cls = [], [], []
        for trk in current_tracks:
            bbox = trk.tlwh
            id = trk.track_id
            cls = trk.cls

            # filter low area bbox
            if bbox[2] * bbox[3] > min_area:
                cur_tlwh.append(bbox)
                cur_id.append(id)
                cur_cls.append(cls)

        plot_img(im0, [cur_tlwh, cur_id, cur_cls])

        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
        # print("Frame Processing!")
        timer.toc()  # end timing this image
        t6 = time_sync()
        frame_id += 1

        seq_fps.append(frame_id / timer.total_time)  # cal fps for current seq
        timer.clear()  # clear for next seq

        totaltime = t6 - t1
        string_ += 'total time: ' + str(round(totaltime, 3)) + 'ms'
        average_time.append(round(totaltime, 3))
        print(string_)

    print(f'average fps: {np.mean(seq_fps)}')
    print(f'average time: {round(np.mean(average_time), 3)}ms')
    print("Video Exported Success")

    if update:
        strip_optimizer(weights)

    if vid_cap:
        vid_cap.release()



def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracker', type=str, default='c_biou', help='sort, deepsort, etc')
    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default=ROOT / 'weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default=ROOT / 'weights/DHN.pth', help='path of DHN path for DeepMOT')
    # other options
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--kalman_format', type=str, default='default', help='use what kind of Kalman, default, naive, strongsort or bot-sort like')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')
    parser.add_argument('--detect_per_frame', type=int, default=1, help='choose how many frames per detect')
    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')

    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/AnFang_20240125.onnx', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'datasets/test.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[384, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold, IOU thresh to filter tracks')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track/c_biou', help='save results to project/name')
    parser.add_argument('--name', default='test', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--blur-obj', action='store_true', help='Blur Detected Objects')
    parser.add_argument('--color-box', action='store_true', help='Change color of every box and track')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

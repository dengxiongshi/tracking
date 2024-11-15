"""
main code for track
"""
import numpy as np
import torch
import cv2
from PIL import Image
import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracker'))

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
# print('ROOT=', ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse
import time
from tracker.timer import Timer
import yaml

from tracker.basetrack import BaseTracker  # for framework
from tracker.deepsort import DeepSORT
from tracker.bytetrack import ByteTrack
from tracker.deepmot import DeepMOT
from tracker.botsort import BoTSORT
from tracker.c_biou_tracker import C_BIoUTracker
from tracker.uavmot import UAVMOT
from tracker.strongsort import StrongSORT

from models.common import DetectMultiBackend
from tracker.evaluate import evaluate
from utils.general import check_img_size, non_max_suppression, scale_boxes, check_file, increment_path, \
    check_imshow, Profile
from utils.torch_utils import select_device, smart_inference_mode, TracedModel
from utils.dataloaders import LoadStreams, LoadImages, LoadScreenshots

print('Note: running yolo v5 detector')


from tracker import tracker_dataloader

from tracker import trackeval


class my_queue:
    """
    implement a queue for image seq reading
    """

    def __init__(self, arr: list, root_path: str) -> None:
        self.arr = arr
        self.start_idx = 0
        self.root_path = root_path

    def push_back(self, item):
        self.arr.append(item)

    def pop_front(self):
        ret = cv2.imread(os.path.join(self.root_path, self.arr[self.start_idx]))
        self.start_idx += 1
        return not self.is_empty(), ret

    def is_empty(self):
        return self.start_idx == len(self.arr)


def set_basic_params(cfgs):
    global CATEGORY_DICT, DATASET_ROOT, CERTAIN_SEQS, IGNORE_SEQS, YAML_DICT
    CATEGORY_DICT = cfgs['CATEGORY_DICT']
    DATASET_ROOT = cfgs['DATASET_ROOT']
    CERTAIN_SEQS = cfgs['CERTAIN_SEQS']
    IGNORE_SEQS = cfgs['IGNORE_SEQS']
    YAML_DICT = cfgs['YAML_DICT']


def post_process_v5(out, img_size, ori_img_size):
    """ post process for v5 and v7

    """

    out = non_max_suppression(out, conf_thres=0.01, )[0]
    out[:, :4] = scale_boxes(img_size, out[:, :4], ori_img_size, ratio_pad=None).round()

    # out: tlbr, conf, cls

    return out


def save_results(folder_name, seq_name, results, data_type='mot17'):
    """
    write results to txt file

    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: write data format, default or mot17 format.
    """
    assert len(results)

    if not os.path.exists(f'./tracker/results/{folder_name}'):
        os.makedirs(f'./tracker/results/{folder_name}')

    with open(os.path.join('./tracker/results', folder_name, seq_name + '.txt'), 'w') as f:
        for frame_id, target_ids, tlwhs, clses in results:
            if data_type == 'default':

                for id, tlwh, cls in zip(target_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{int(cls)}\n')

            elif data_type == 'mot17':
                for id, tlwh, cls in zip(target_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n')
    f.close()

    return folder_name


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

    img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    for tlwh, id, cls in zip(tlwhs, ids, clses):
        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'id: {id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=(255, 164, 0), thickness=2)

    # cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)
    return img_

def save_videos(obj_name, save_floder):
    """
    convert imgs to a video

    seq_names: List[str] or str, seqs that will be generated
    """

    if not isinstance(obj_name, list):
        obj_name = [obj_name]

    for seq in obj_name:
        if 'mp4' in seq: seq = seq[:-4]
        images_path = os.path.join(save_floder, 'result_images', seq)
        images_name = sorted(os.listdir(images_path))

        to_video_path = os.path.join(images_path, '/', seq + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        img0 = Image.open(os.path.join(images_path, images_name[0]))
        vw = cv2.VideoWriter(to_video_path, fourcc, 30, img0.size)

        for img in images_name:
            if img.endswith('.jpg'):
                frame = cv2.imread(os.path.join(images_path, img))
                vw.write(frame)

    print('Save videos Done!!')


def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_v5(ori_img, model_size, model_stride):
    """ simple preprocess for a single image

    """
    img_resized = _letterbox(ori_img, new_shape=model_size, stride=model_stride)[0]

    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img_resized = np.ascontiguousarray(img_resized)

    img_resized = torch.from_numpy(img_resized).float()
    img_resized /= 255.0

    return img_resized, ori_img


timer = Timer()
seq_fps = []  # list to store time used for every seq


def main(opts):
    # set_basic_params(cfgs)  # NOTE: set basic path and seqs params first

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

    # NOTE: ATTENTION: make kalman and tracker compatible
    if opts.tracker == 'botsort':
        opts.kalman_format = 'botsort'
    elif opts.tracker == 'strongsort':
        opts.kalman_format = 'strongsort'

    # NOTE: if save video, you must save image
    # if opts.save_videos:
    #     opts.save_images = True

    """
    1. load model
    """
    bs = 1
    device = select_device(opts.device)
    model = DetectMultiBackend(opts.weights, device=device, dnn=opts.dnn, data=None, fp16=opts.half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(opts.img_size, s=stride)  # check image size

    # if opts.trace:
    #     print(opts.img_size)
    #     model = TracedModel(model, device, opts.img_size)
    # else:
    model.to(device)

    model.eval()
    # warm up
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    """
    2. load videos or images
    """
    if opts.demo == "video":
        obj_name = opts.obj
        # check path
        assert os.path.exists(obj_name), 'the path does not exist! '
    # if read video, then put every frame into a queue
    # if read image seqs, the same as video
    resized_images_queue = []  # List[torch.Tensor] store resized images
    images_queue = []  # List[torch.Tensor] store origin images

    current_time = time.localtime()
    obj, get_next_frame = None, None  # init obj
    if opts.demo == "video" or opts.demo == "webcam":  # if it is a video
        obj = cv2.VideoCapture(opts.obj if opts.demo == "video" else opts.camid)
        width = obj.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = obj.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = obj.get(cv2.CAP_PROP_FPS)

        get_next_frame = lambda _: obj.read()

        # if os.path.isabs(obj_name):
        #     obj_name = obj_name.split('/')[-1][:-4]
        # else:
        #     obj_name = obj_name[:-4]
        save_folder = os.path.join(
            opts.save_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = (os.path.join(save_folder, opts.obj.replace("\\", "/").split("/")[-1])
                    if opts.demo == "video"
                    else os.path.join(save_folder, "camera.mp4")
        )
        print(f"save_path is {save_path}")

        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    else:
        obj = my_queue(os.listdir(obj_name), obj_name)
        get_next_frame = lambda _: obj.pop_front()

        # if os.path.isabs(obj_name): obj_name = obj_name.split('/')[-1]
        name = os.path.basename(obj_name)
        obj_name = os.path.splitext(name)[0]

    """
    3. start tracking
    """
    tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)  # instantiate tracker  TODO: finish init params

    results = []  # store current seq results
    frame_id = 0

    while True:
        # print(f'----------processing frame {frame_id}----------')
        string_ = f'----------processing frame {frame_id}----------'
        # end condition
        is_valid, img0 = get_next_frame(None)  # img0: (H, W, C)

        if not is_valid:
            break  # end of reading

        img, img0 = preprocess_v5(ori_img=img0, model_size=imgsz, model_stride=stride)

        # timer.tic()  # start timing this img
        img = img.unsqueeze(0)  # （C, H, W) -> (bs == 1, C, H, W)

        timer.tic()  # start timing this img
        start_time = time.time()

        yolov5_out = model(img.to(device))
        yolov5_out = yolov5_out[0]

        yolov5_out = post_process_v5(yolov5_out, img_size=img.shape[2:], ori_img_size=img0.shape)

        string_ += 'yolov5-time: ' + str(round(time.time() - start_time, 3))

        current_tracks = tracker.update(yolov5_out, img0)  # List[class(STracks)]

        # save results
        cur_tlwh, cur_id, cur_cls = [], [], []
        for trk in current_tracks:
            bbox = trk.tlwh
            id = trk.track_id
            cls = trk.cls

            # filter low area bbox
            if bbox[2] * bbox[3] > opts.min_area:
                cur_tlwh.append(bbox)
                cur_id.append(id)
                cur_cls.append(cls)
                # results.append((frame_id + 1, id, bbox, cls))

        results.append((frame_id + 1, cur_id, cur_tlwh, cur_cls))
        timer.toc()  # end timing this image

        result_img = plot_img(img0, [cur_tlwh, cur_id, cur_cls])

        vid_writer.write(result_img)

        frame_id += 1
        string_ += '----total-time: ' + str(round(time.time() - start_time, 3))
        print(string_)

        seq_fps.append(frame_id / timer.total_time)  # cal fps for current seq
        timer.clear()  # clear for next seq
    # thirdly, save results
    # every time assign a different name
        if opts.save_txt:
            save_results(obj_name, results)

    ## finally, save videos
    # save_videos(obj_name, opts.save_folder)

    print(f'average fps: {np.mean(seq_fps)}')


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument('--obj', type=str, default='/data/tracking/datasets/test.mp4', help='video NAME or images FOLDER NAME')
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--show_image", action="store_true", default=False, help="whether to save the inference result of image/video")
    # parser.add_argument('--dataset', type=str, default='visdrone', help='visdrone, visdrone_car, uavdt or mot')
    # parser.add_argument('--data_format', type=str, default='origin', help='format of reading dataset')
    # parser.add_argument('--det_output_format', type=str, default='yolo', help='data format of output of detector, yolo or other')

    parser.add_argument('--tracker', type=str, default='c_biou', help='sort, deepsort, etc')

    parser.add_argument('--weights', type=str, default='/data/tracking/weights/best.pt', help='model path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_txt', type=bool, default=False, help='save results to *.txt')

    parser.add_argument('--img_size', nargs='+', type=int, default=[384, 640], help='inference size h,w')
    parser.add_argument('--save_folder', default=ROOT / 'results', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')

    # threshs
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='filter tracks')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU thresh to filter tracks')

    # other options
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--kalman_format', type=str, default='default',
                        help='use what kind of Kalman, default, naive, strongsort or bot-sort like')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')

    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')

    # detect per several frames
    parser.add_argument('--detect_per_frame', type=int, default=1, help='choose how many frames per detect')

    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')

    opts = parser.parse_args()
    opts.img_size *= 2 if len(opts.img_size) == 1 else 1  # expand

    return opts


if __name__ == '__main__':
    opt = parse_opt()

    # NOTE: read path of datasets, sequences and TrackEval configs
    # with open(f'./tracker/config_files/{opts.dataset}.yaml', 'r') as f:
    #     cfgs = yaml.load(f, Loader=yaml.FullLoader)
    # main(opts, cfgs)
    main(opt)

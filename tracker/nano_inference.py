import argparse
import os
import time
import cv2
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir


# def load
def run_detector(config_path, model_path, demo_type, demo_path, cam_id=0, save_result=True):
    def get_image_list(path):
        image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_names.append(apath)
        return image_names

    def visualize(dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img

    args = argparse.Namespace(
        demo=demo_type,
        config=config_path,
        model=model_path,
        path=demo_path,
        camid=cam_id,
        save_result=save_result
    )

    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)

    def inference(img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = pipeline(None, meta, cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = model.inference(meta)
            return meta, results

    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    model = build_model(cfg.model)
    ckpt = torch.load(args.model, map_location=lambda storage, loc: storage)
    load_model_weight(model, ckpt, logger)
    if cfg.model.arch.backbone.name == "RepVGG":
        deploy_config = cfg.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)
    device = "cuda:0"
    model = model.to(device).eval()
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    if args.demo == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = inference(image_name)
            result_image = visualize(res[0], meta, cfg.class_names, 0.35)
            if args.save_result:
                save_folder = os.path.join(
                    cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
    elif args.demo == "video" or args.demo == "webcam":
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(
            cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        mkdir(local_rank, save_folder)
        save_path = (
            os.path.join(save_folder, args.path.replace("\\", "/").split("/")[-1])
            if args.demo == "video"
            else os.path.join(save_folder, "camera.mp4")
        )
        print(f"save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = inference(frame)
                result_frame = visualize(res[0], meta, cfg.class_names, 0.35)
                if args.save_result:
                    vid_writer.write(result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break

# Example usage:
# run_detector("path/to/config.yaml", "path/to/model.pth", "image", "path/to/images", save_result=True)
run_detector(r"/data/nanotracking/src/nanodet/nanodet-plus-m_416-yolo.yml", r"/data/nanotracking/src/nanodet/nanodet_model_best.pth", "video", r"/data/nanotracking/src/nanodet/test.mp4", save_result=True)

from __future__ import division
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import pickle as pkl
from tqdm import tqdm
import glob
import shutil

from darknet import Darknet
from dataset import YoloDataset
from util import *


def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        type=str)
    parser.add_argument("--output", dest='output', help="Image / Directory to store detections to",
                        default="output", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=50)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--resolution", dest='resolution', default="256", type=str,
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed")
    parser.add_argument("--image_size", dest='image_size', help="Size of evaluated images", default=256, type=int)
    parser.add_argument('--iou', dest='iou', action='store_true')

    parser.add_argument('--gpu', dest='gpu', type=str, default="0")

    return parser.parse_args()


def run_yolo(args):
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    img_size = args.image_size

    CUDA = torch.cuda.is_available()

    classes = load_classes('data/coco.names')

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.resolution
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU available, put the model on GPU
    if CUDA:
        _gpu = int(args.gpu)
        torch.cuda.set_device(_gpu)
        model.cuda()
        print("Using GPU: {}".format(_gpu))

    # Set the model in evaluation mode
    model.eval()
    print("saving to {}".format(args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # go through all folders of generated images
    for dir in tqdm(os.listdir(images)):
        full_dir = os.path.join(images, dir)

        # check if there exists a ground truth file (which would contain bboxes etc to calculate IoU)
        ground_truth_file = [_file for _file in os.listdir(full_dir) if _file.endswith(".pkl")]
        if len(ground_truth_file) > 0:
            shutil.copyfile(os.path.join(full_dir, ground_truth_file[0]),
                            os.path.join(args.output, "ground_truth_{}.pkl".format(dir)))

        # create dataset
        image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1, 1, 1))])
        dataset = YoloDataset(full_dir, transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 drop_last=False, shuffle=False, num_workers=4)

        num_batches = len(dataloader)
        dataloader = iter(dataloader)
        output_dict = {}

        for idx in tqdm(range(num_batches)):
            data = dataloader.next()
            imgs, filenames = data
            if CUDA:
                imgs = imgs.cuda()

            with torch.no_grad():
                predictions = model(imgs, CUDA)
                predictions = non_max_suppression(predictions, confidence, nms_thresh)

            for img, preds in zip(filenames, predictions):
                img_preds_name = []
                img_preds_id = []
                img_bboxs = []
                if preds is not None and len(preds) > 0:
                    for pred in preds:
                        pred_id = int(pred[-1])
                        pred_name = classes[pred_id]

                        bbox_x = pred[0] / img_size
                        bbox_y = pred[1] / img_size
                        bbox_width = (pred[2] - pred[0]) / img_size
                        bbox_height = (pred[3] - pred[1]) / img_size

                        img_preds_id.append(pred_id)
                        img_preds_name.append(pred_name)
                        img_bboxs.append([bbox_x.cpu().numpy(), bbox_y.cpu().numpy(),
                                          bbox_width.cpu().numpy(), bbox_height.cpu().numpy()])
                output_dict[img.split("/")[-1]] = [img_preds_name, img_preds_id, img_bboxs]

        # print("saving to {}/{}".format(args.output, dir))
        with open(os.path.join(args.output, "detected_{}.pkl".format(dir)), "wb") as f:
            pkl.dump(output_dict, f)


def calc_recall(predicted_bbox, label):
    correctly_recognized = 0
    num_images_total = len(predicted_bbox.keys())
    for key in predicted_bbox.keys():
        predictions = predicted_bbox[key]
        for recognized_label in predictions[1]:
            if recognized_label == label:
                correctly_recognized += 1
                break
    if num_images_total == 0:
        return -1, -1, -1
    accuracy = float(correctly_recognized) / num_images_total
    return accuracy, correctly_recognized, num_images_total


def calc_iou(predicted_bbox, gt_bbox, label):
    ious = []

    for key in predicted_bbox.keys():
        predicted_bboxes = []
        predictions = predicted_bbox[key]
        for recognized_label, pred_bbox in zip(predictions[1], predictions[2]):
            if recognized_label == label:
                predicted_bboxes.append(pred_bbox)

        gt_bboxes = []
        gts = gt_bbox[key]
        if gts[1] is None:
            continue
        elif len(gts[1]) == 0:
            continue
        if type(gts[1][0]) == np.ndarray:
            for real_label, real_bbox in zip(gts[1][0], gts[2][0]):
                if int(real_label) == label:
                    gt_bboxes.append(real_bbox)
        else:
            for real_label, real_bbox in zip(gts[1], gts[2]):
                if real_label == label:
                    gt_bboxes.append(real_bbox)

        all_current_ious = []
        for current_predicted_bbox in predicted_bboxes:
            for current_gt_bbox in gt_bboxes:
                current_iou = get_iou(current_predicted_bbox, current_gt_bbox)
                all_current_ious.append(current_iou)
        if len(all_current_ious) > 0:
            ious.append(max(all_current_ious))
    if len(ious) == 0:
        return 0.0
    avg_iou = sum(ious) / float(len(ious))
    return avg_iou


def calc_soa(args):
    results_dict = {}
    # find yolo results
    yolo_detected_files = [_file for _file in os.listdir(args.output)
                           if _file.endswith(".pkl") and _file.startswith("detected_")]

    for yolo_file in yolo_detected_files:
        yolo = load_file(yolo_file)
        label = get_label(yolo_file)
        acc, correctly_recog, num_imgs_total = calc_recall(yolo, label)

        results_dict[label] = {}
        results_dict[label]["accuracy"] = acc
        results_dict[label]["images_recognized"] = correctly_recog
        results_dict[label]["images_total"] = num_imgs_total

    if args.iou:
        ground_truth_files = [_file for _file in os.listdir(args.output)
                              if _file.endswith(".pkl") and _file.startswith("ground_truth_")]

        yolo_detected_files = sorted(yolo_detected_files)
        ground_truth_files = sorted(ground_truth_files)

        for yolo_file, gt_file in zip(yolo_detected_files, ground_truth_files):
            yolo = load_file(yolo_file)
            gt = load_file(gt_file)
            label = get_label(yolo_file)
            iou = calc_iou(yolo, gt, label)

            results_dict[label]["iou"] = iou

    # calculate SOA-C and SOA-I
    with open(os.path.join(args.output, "result_file.pkl")) as f:
        pkl.dump(results_dict, f)



if __name__ == '__main__':
    args = arg_parse()

    # use YOLOv3 on all images
    run_yolo(args)

    # calculate score
    calc_soa(args)


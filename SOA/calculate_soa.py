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

    # check that the given folder contains exactly 80 folders
    _all_dirs = os.listdir(images)
    _num_folders = 0
    for _dir in _all_dirs:
        if os.path.isdir(os.path.join(images, _dir)):
            _num_folders += 1
    if _num_folders != 80:
        print("")
        print("****************************************************************************")
        print("\tWARNING")
        print("\tDid not find exactly 80 folders ({} folders found) in {}.".format(_num_folders, images))
        print("\tFor the final calculation please make sure the folder {} contains one subfolder for each of the labels.".format(images))
        print("\tCalculating scores on {}/80 labels now, but results will not be conclusive.".format(_num_folders))
        print("****************************************************************************")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

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

    # go through all folders of generated images
    for dir in tqdm(os.listdir(images)):
        full_dir = os.path.join(images, dir)

        # check if there exists a ground truth file (which would contain bboxes etc to calculate IoU)
        ground_truth_file = [_file for _file in os.listdir(full_dir) if _file.endswith(".pkl")]
        if len(ground_truth_file) > 0:
            shutil.copyfile(os.path.join(full_dir, ground_truth_file[0]),
                            os.path.join(args.output, "ground_truth_{}.pkl".format(dir)))

        # check if detection was already run for this label
        if os.path.isfile(os.path.join(args.output, "detected_{}.pkl".format(dir))):
            print("Detection already run for {}. Continuing with next label.".format(dir))
            continue

        # create dataset from images in the current folder
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

        # get YOLO predictions for images in current folder
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

        with open(os.path.join(args.output, "detected_{}.pkl".format(dir)), "wb") as f:
            pkl.dump(output_dict, f)


def calc_recall(predicted_bbox, label):
    """Calculate how often a given object (label) was detected in the images"""
    correctly_recognized = 0
    num_images_total = len(predicted_bbox.keys())
    for key in predicted_bbox.keys():
        predictions = predicted_bbox[key]
        for recognized_label in predictions[1]:
            if recognized_label == label:
                correctly_recognized += 1
                break
    if num_images_total == 0:
        return 0, 0, 0
    accuracy = float(correctly_recognized) / num_images_total
    return accuracy, correctly_recognized, num_images_total


def calc_iou(predicted_bbox, gt_bbox, label):
    """Calculate max IoU between correctly detected objects and provided ground truths for each image"""
    ious = []

    # iterate over the predictions for all images
    for key in predicted_bbox.keys():
        predicted_bboxes = []
        # get predictions for the image
        predictions = predicted_bbox[key]
        # check if it recognized an object of the given label and if yes get its predicted bounding box
        # for all detected objects of the given label
        for recognized_label, pred_bbox in zip(predictions[1], predictions[2]):
            if recognized_label == label:
                predicted_bboxes.append(pred_bbox)

        gt_bboxes = []
        # get the ground truth information of the current image
        gts = gt_bbox[key]

        if gts[1] is None or len(gts[1]) == 0:
            continue
        else:
            # gts should e.g. be [[],
            #                     [7, 1], -> integer values for the object labels
            #                     [[0.1, 0.2, 0.3, 0.5], [0.4, 0.3, 0.3, 0.5]] -> bounding boxes
            assert type(gts[1]) is list and type(gts[2]) is list,\
                   "Expected lists as entries of the ground truth bounding box file"
            for real_label, real_bbox in zip(gts[1], gts[2]):
                if real_label == label:
                    assert all([_val >= 0 and _val <= 1 for _val in real_bbox]), \
                        "Bounding box entries should be between 0 and 1 but are: {}.".format(real_bbox)
                    gt_bboxes.append(real_bbox)

        # calculate all IoUs between ground truth bounding boxes of the given label
        # and predicted bounding boxes of the given label
        all_current_ious = []
        for current_predicted_bbox in predicted_bboxes:
            for current_gt_bbox in gt_bboxes:
                current_iou = get_iou(current_predicted_bbox, current_gt_bbox)
                all_current_ious.append(current_iou)
        # choose the maximum value as the IoU for this image
        if len(all_current_ious) > 0:
            ious.append(max(all_current_ious))
    if len(ious) == 0:
        return 0.0
    avg_iou = sum(ious) / float(len(ious))
    return avg_iou


def calc_overall_class_average_accuracy(dict):
    """Calculate SOA-C"""
    accuracy = 0
    for label in dict.keys():
        accuracy += dict[label]["accuracy"]
    overall_accuracy = accuracy / len(dict.keys())
    return overall_accuracy


def calc_image_weighted_average_accuracy(dict):
    """Calculate SOA-I"""
    accuracy = 0
    total_images = 0
    for label in dict.keys():
        num_images = dict[label]["images_total"]
        accuracy += num_images * dict[label]["accuracy"]
        total_images += num_images
    overall_accuracy = accuracy / total_images
    return overall_accuracy


def calc_split_class_average_accuracy(dict):
    """Calculate SOA-C-Top/Bot-40"""
    num_img_list = []
    for label in dict.keys():
        num_img_list.append([label, dict[label]["images_total"]])
    num_img_list.sort(key=lambda x: x[1])
    sorted_label_list = [x[0] for x in num_img_list]

    bottom_40_accuracy = 0
    top_40_accuracy = 0
    for label in dict.keys():
        if sorted_label_list.index(label) < 40:
            bottom_40_accuracy += dict[label]["accuracy"]
        else:
            top_40_accuracy += dict[label]["accuracy"]
    bottom_40_accuracy /= 0.5*len(dict.keys())
    top_40_accuracy /= 0.5*len(dict.keys())

    return top_40_accuracy, bottom_40_accuracy


def calc_overall_class_average_iou(dict):
    """Calculate SOA-C-IoU"""
    iou = 0
    for label in dict.keys():
        if dict[label]["iou"] is not None and dict[label]["iou"] >= 0:
            iou += dict[label]["iou"]
    overall_iou = iou / len(dict.keys())
    return overall_iou


def calc_image_weighted_average_iou(dict):
    """Calculate SOA-I-IoU"""
    iou = 0
    total_images = 0
    for label in dict.keys():
        num_images = dict[label]["images_total"]
        if dict[label]["iou"] is not None and dict[label]["iou"] >= 0:
            iou += num_images * dict[label]["iou"]
        total_images += num_images
    overall_iou = iou / total_images
    return overall_iou


def calc_split_class_average_iou(dict):
    """Calculate SOA-C-IoU-Top/Bot-40"""
    num_img_list = []
    for label in dict.keys():
        num_img_list.append([label, dict[label]["images_total"]])
    num_img_list.sort(key=lambda x: x[1])
    sorted_label_list = [x[0] for x in num_img_list]

    bottom_40_iou = 0
    top_40_iou = 0
    for label in dict.keys():
        if sorted_label_list.index(label) < 40:
            if dict[label]["iou"] is not None and dict[label]["iou"] >= 0:
                bottom_40_iou += dict[label]["iou"]
        else:
            if dict[label]["iou"] is not None and dict[label]["iou"] >= 0:
                top_40_iou += dict[label]["iou"]
    bottom_40_iou /= 0.5*len(dict.keys())
    top_40_iou /= 0.5*len(dict.keys())

    return top_40_iou, bottom_40_iou


def calc_soa(args):
    """Calculate SOA scores"""
    results_dict = {}

    # find detection results
    yolo_detected_files = [os.path.join(args.output, _file) for _file in os.listdir(args.output)
                           if _file.endswith(".pkl") and _file.startswith("detected_")]

    # go through yolo detection and check how often it detected the desired object (based on the label)
    for yolo_file in yolo_detected_files:
        yolo = load_file(yolo_file)
        label = get_label(yolo_file)
        acc, correctly_recog, num_imgs_total = calc_recall(yolo, label)

        results_dict[label] = {}
        results_dict[label]["accuracy"] = acc
        results_dict[label]["images_recognized"] = correctly_recog
        results_dict[label]["images_total"] = num_imgs_total

    # calculate SOA-C and SOA-I
    print("")
    class_average_acc = calc_overall_class_average_accuracy(results_dict)
    print("Class average accuracy for all classes (SOA-C) is: {:6.4f}".format(class_average_acc))

    image_average_acc = calc_image_weighted_average_accuracy(results_dict)
    print("Image weighted average accuracy (SOA-I) is: {:6.4f}".format(image_average_acc))

    top_40_class_average_acc, bottom_40_class_average_acc = calc_split_class_average_accuracy(results_dict)
    print("Top (SOA-C-Top40) and Bottom (SOA-C-Bot40) 40 class average accuracy is: {:6.4f} and {:6.4f}".
          format(top_40_class_average_acc, bottom_40_class_average_acc))

    # if IoU is true calculate the IoU scores, too
    if args.iou:
        ground_truth_files = [os.path.join(args.output, _file) for _file in os.listdir(args.output)
                              if _file.endswith(".pkl") and _file.startswith("ground_truth_")]

        yolo_detected_files = sorted(yolo_detected_files)
        ground_truth_files = sorted(ground_truth_files)

        for yolo_file, gt_file in zip(yolo_detected_files, ground_truth_files):
            yolo = load_file(yolo_file)
            gt = load_file(gt_file)
            label = get_label(yolo_file)
            iou = calc_iou(yolo, gt, label)

            results_dict[label]["iou"] = iou

        print("")
        class_average_iou = calc_overall_class_average_iou(results_dict)
        print("Class average IoU for all classes (SOA-C-IoU) is: {:6.4f}".format(class_average_iou))

        image_average_iou = calc_image_weighted_average_iou(results_dict)
        print("Image weighted average IoU (SOA-I-IoU) is: {:6.4f}".format(image_average_iou))

        top_40_class_average_iou, bottom_40_class_average_iou = calc_split_class_average_iou(results_dict)
        print("Top (SOA-C-Top40-IoU) and Bottom (SOA-C-Bot40-IoU) 40 class average IoU is: {:6.4f} and {:6.4f}".
              format(top_40_class_average_iou, bottom_40_class_average_iou))

    # store results
    with open(os.path.join(args.output, "result_file.pkl"), "wb") as f:
        pkl.dump(results_dict, f)


if __name__ == '__main__':
    args = arg_parse()

    # use YOLOv3 on all images
    print("Using YOLOv3 Network on Generated Images...")
    run_yolo(args)

    # calculate score
    print("Calculating SOA Score...")
    calc_soa(args)


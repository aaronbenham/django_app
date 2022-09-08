import base64

from channels.generic.websocket import WebsocketConsumer
import keyboard

import json
from random import randint
from time import sleep

import cv2
import threading
import numpy as np

from sklearn import preprocessing
import matplotlib.pyplot as plt

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion.fast_gradient import FastGradientMethod

import os


standard_scores = [8]
standard_deviation_score = 0


class WSConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

        # cam = Videocamera()
        # frame_jpeg, frame_array = cam.get_frame()
        # u, s, v, mean_img = calculate_cov(frame_array)
        # attack = False
        # frame = 0
        #
        # while True:
        #     score, average_confidence_score, above_threshold_scores_average, number_of_boxes, output_jpg, saliency = gen(
        #         cam, u, s, v, mean_img, attack)
        #
        #     img = base64.b64encode(output_jpg).decode()
        #     saliency = base64.b64encode(saliency).decode()
        #
        #     self.send(json.dumps(
        #         {'std_score': score, 'confidence_score': average_confidence_score, 'num_boxes': number_of_boxes,
        #          'above_threshold': above_threshold_scores_average, 'output': "data:image/jpg;base64," + img,
        #          'saliency': "data:image/jpg;base64," + saliency}))
        #
        #     if keyboard.is_pressed('q'):
        #         print("Stopping...")
        #         break
        #
        #     frame += 1
        #     if frame == 3:
        #         attack = True
        #     else:
        #         attack = False


    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        frame_to_attack = int(text_data_json['frame_to_attack'])
        type_of_attack = text_data_json['attack_type']
        eps = int(text_data_json['eps'])

        cam = Videocamera()
        frame_jpeg, frame_array = cam.get_frame()
        u, s, v, mean_img = calculate_cov(frame_array)
        attack = False
        frame = 1

        while True:
            score, average_confidence_score, above_threshold_scores_average, number_of_boxes, output_jpg, saliency = \
                gen(cam, u, s, v, mean_img, attack, type_of_attack, eps)

            img = base64.b64encode(output_jpg).decode()
            saliency = base64.b64encode(saliency).decode()

            self.send(json.dumps(
                {'std_score': score, 'confidence_score': average_confidence_score, 'num_boxes': number_of_boxes,
                 'above_threshold': above_threshold_scores_average, 'output': "data:image/jpg;base64," + img,
                 'saliency': "data:image/jpg;base64," + saliency}))

            if keyboard.is_pressed('q'):
                print("Stopping...")
                break

            frame += 1
            if frame == frame_to_attack:
                attack = True
            else:
                attack = False


class Videocamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame  # array format

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera, u, s, v, mean_img, attack, type_of_attack, eps):
    frame_jpeg, frame_array = camera.get_frame()

    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
    image = np.stack([frame_array], axis=0).astype(np.float32)

    if attack:
        if type_of_attack == "fast":
            attack = FastGradientMethod(estimator=frcnn, eps=eps, eps_step=2)
        elif type_of_attack == "projected":
            attack = ProjectedGradientDescent(estimator=frcnn, eps=eps, eps_step=2, max_iter=6)

        output = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite('clean_frame.jpg', output)

        os.system("python ../pytorch_grad_cam/cam.py --image-path clean_frame.jpg --method gradcam")

        image = attack.generate(x=image, y=None)

        output = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite('attacked_frame.jpg', output)

        predictions = frcnn.predict(x=image)
    else:
        predictions = frcnn.predict(x=image)

    for i in range(image.shape[0]):
        # Process predictions
        predictions_class, predictions_boxes, predictions_class, predictions_score, average_confidence_score, \
        above_threshold_scores_average = extract_predictions(predictions[i])

        # Plot predictions
        output_jpg, number_of_boxes = plot_image_with_boxes(img=image[i].copy(), boxes=predictions_boxes,
                                                                          pred_cls=predictions_class, pred_scr=predictions_score)

    x = image.reshape((-1, 8 * 8 * 3))
    img = np.dot(x - mean_img, u)
    score = calc_standard_deviation(img, s)
    saliency = update_saliency(img, s, u)

    return score, average_confidence_score, above_threshold_scores_average, number_of_boxes, output_jpg, saliency


def update_saliency(img, s, u):
    whitened_image = np.dot(img / np.sqrt(s + 1e-4), u.T)

    img_float32 = np.float32(whitened_image.reshape(480, 640, 3))

    saliency = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR)
    y = np.array(saliency)
    saliency_mean_img = np.mean(y, axis=0)
    saliency = saliency - saliency_mean_img

    output = cv2.cvtColor(saliency.astype(np.uint8), cv2.COLOR_BGR2RGB)
    output_saliency = output.astype(np.uint8)

    _, output_saliency = cv2.imencode('.jpg', output_saliency)

    return output_saliency.tobytes()


def calc_standard_deviation(img, s):
    r = img / np.sqrt(s + 1e-11)
    r = r.reshape(921600)
    standard_scores.append(np.std(r))

    # standard_deviation_score = standard_scores[-1] / max(standard_scores)

    normalized = preprocessing.normalize([standard_scores])
    standard_deviation_score = normalized[0][len(standard_scores) - 1]

    return standard_deviation_score


def calculate_cov(image):
    image = image.reshape((-1, 8 * 8 * 3))

    mean_img = np.mean(image, axis=0)
    cov = np.dot((image - mean_img).T, (image - mean_img)) / image.shape[0]
    u, s, v = np.linalg.svd(cov)

    return u, s, v, mean_img

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

frcnn = PyTorchFasterRCNN(
            clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"])

above_threshold_scores_average = []
above_threshold_scores = []

def extract_predictions(predictions_):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    # print("\npredicted classes:", predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    # print("predicted score:", predictions_score)
    threshold = 0.8


    for score in predictions_score:
        if score > threshold:
           above_threshold_scores.append(score)

    above_threshold_scores_average = np.sum(above_threshold_scores) / len(above_threshold_scores)
    average_confidence_score = np.sum(predictions_score) / len(predictions_score)


    # Get a list of index with score greater than threshold
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

    predictions_boxes = predictions_boxes[: predictions_t + 1]
    predictions_class = predictions_class[: predictions_t + 1]
    prediction_plots = predictions_score[: predictions_t + 1]

    return predictions_class, predictions_boxes, predictions_class, prediction_plots, average_confidence_score, above_threshold_scores_average


def plot_image_with_boxes(img, boxes, pred_cls, pred_scr, scale=2):
    text_size = 0.6*(int(scale/2))
    text_th = 2*(int(scale/2))
    rect_th = 2*(int(scale/2))

    number_of_boxes = len(boxes)

    for i in range(len(boxes)):
        c1 = boxes[i][0]
        c2 = boxes[i][1]
        # Draw Rectangle with the coordinates
        cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color=(0, 255, 0), thickness=rect_th)

        # Write the prediction class
        cv2.putText(img, pred_cls[i], (int(c1[0]), int(c1[1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, color=(0, 255, 0), thickness=text_th)
        cv2.putText(img, str(round(pred_scr[i], 2)), (int(c2[0]), int(c1[1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, color=(0, 255, 0), thickness=text_th)


    output = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    _, jpeg = cv2.imencode('.jpg', output)

    return jpeg.tobytes(), number_of_boxes
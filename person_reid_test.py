from copy import deepcopy
from dataclasses import dataclass

import cv2 as cv
import rospy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes

import utils
from models import Market1501TripletMiniVGGEval


@dataclass
class InitData:
    init_image: torch.tensor
    init_vector: torch.tensor


def box_callback(msg):
    global flow_boxes
    flow_boxes = msg


def get_box():
    global flow_boxes
    return deepcopy(flow_boxes)


def keep_only_person_boxes(boxes):
    return list(filter(lambda x: x.label == 'person', boxes))


bridge = CvBridge()
flow_boxes = None


def main():
    box_topic = '/YD/boxes'
    device = 'cuda:0'

    input_shape = (1, 3, 128, 64)
    model = Market1501TripletMiniVGGEval(
        input_shape=input_shape,
        embedding_dim=32,
        conv_blocks=2,
        conv_kernel_size=(3, 3),
        max_pool_kernel_size=(2, 2),
        dropout_rate=0.03,
        filters=64
    )

    model.load_state_dict(torch.load(
        '/media/rootadminwalker/DATA/outputs/Market1501_triplet_outputs/model_name(Mini-VGG)_embedding_dim(32)_ep30_loss(triplet)_margin(2)/model_checkpoints/ep1_ilNone_train-loss0.1475_val-loss0.2292.pth'))
    model.to(device)
    model.eval()

    rospy.init_node('test')

    rospy.Subscriber(
        box_topic,
        ObjectBoxes,
        box_callback,
        queue_size=1
    )
    rospy.wait_for_message(box_topic, ObjectBoxes)

    __isCaptured = False
    init_data = InitData(None, None)

    while not rospy.is_shutdown():
        boxes = get_box()
        if boxes is None:
            continue

        frame = bridge.compressed_imgmsg_to_cv2(boxes.source_img)
        if __isCaptured:
            person_boxes = keep_only_person_boxes(boxes.boxes)
            for person_box in person_boxes:
                crop_image = bridge.compressed_imgmsg_to_cv2(person_box.source_img)
                crop_image = cv.resize(crop_image, (input_shape[-1:1:-1]))
                blob = utils.cv2_to_torch(crop_image, device=device) / 255
                embedding = model(blob)
                dist = F.pairwise_distance(embedding, init_data.init_vector)
                print(dist)
                if dist <= 10:
                    color = (32, 255, 0)
                else:
                    color = (32, 0, 255)

                cv.rectangle(frame, (person_box.x1, person_box.y1), (person_box.x2, person_box.y2), color, 5)

        cv.imshow('test', frame)
        key = cv.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break

        elif key == ord('c'):
            rx, ry, rw, rh = cv.selectROI('test', frame)
            init_image = frame[ry:ry + rh, rx:rx + rw, :].copy()
            init_image = cv.resize(init_image, (input_shape[-1:1:-1]))
            blob = utils.cv2_to_torch(init_image, device=device) / 255
            init_data.init_image = blob
            init_data.init_vector = model(blob)
            __isCaptured = True


if __name__ == '__main__':
    main()

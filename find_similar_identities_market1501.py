import json
import os
import random
from copy import deepcopy
from typing import List, Tuple

import cv2 as cv
import numpy as np

from datasets import TripletMarket1501Dataset


class ImageSelector:
    def __init__(self, image_size):
        self.image_size = image_size
        self.total_images = []
        self.backup_total_images = []
        self.selection_status = []
        self.images_len = 0

    def callback(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            width_per_image = self.image_size[1]
            for i in range(self.images_len):
                min_x = width_per_image * i
                max_x = width_per_image * (i + 1)
                crop_image = self.total_images[:, min_x:max_x, :].copy()
                if min_x < x < max_x:
                    if not self.selection_status[i]:
                        print(f"Unselected image number {i + 1}")
                        blurred_crop_image = cv.GaussianBlur(crop_image, (35, 35), cv.BORDER_DEFAULT)
                        blurred_crop_image = cv.rectangle(
                            blurred_crop_image, (0, 0), (image.shape[1], image.shape[0]), (32, 0, 255), 6
                        )
                        self.total_images[:, min_x:max_x, :] = blurred_crop_image
                        self.selection_status[i] = True
                    else:
                        print(f"Re-selected image number {i + 1}")
                        self.total_images[:, min_x:max_x, :] = self.backup_total_images[i]
                        self.selection_status[i] = False

                    return

    def select(self, original_image, total_images) -> Tuple[List[bool], bool]:
        force_end = False
        self.images_len = len(total_images)
        self.backup_total_images = deepcopy(total_images)
        self.total_images = np.hstack(total_images)
        self.selection_status = [False] * self.images_len
        cv.namedWindow('test')
        cv.setMouseCallback('test', self.callback)
        try:
            while True:
                cv.imshow('frame', original_image)
                cv.imshow('test', self.total_images)
                key = cv.waitKey(16) & 0xFF
                if key == ord('o'):
                    break
                if key in [ord('q'), 27]:
                    print("User decided to end")
                    force_end = True
                    break

        except KeyboardInterrupt:
            print("User force interrupt the selection progress")

        # cv.destroyAllWindows()
        return self.selection_status, force_end


class PersonReidentification:
    def __init__(self, bin_path, xml_path):
        self.person_extractor = cv.dnn.readNet(bin_path, xml_path)

    def parse_descriptor(self, person_image, crop=False) -> np.array:
        blob = cv.dnn.blobFromImage(
            person_image,
            size=(128, 256),
            scalefactor=1.0,
            mean=(0, 0, 0),
            swapRB=False,
            crop=crop
        )

        self.person_extractor.setInput(blob)
        descriptor = self.person_extractor.forward()
        return descriptor

    @staticmethod
    def compare_descriptors(desc1, desc2):
        return np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))


if __name__ == '__main__':
    start = 0
    if os.path.exists('./current_count.json'):
        with open('./current_count.json') as f:
            start = json.load(f)
            print(f"User decide to start at {start}")

    device = 'cuda:0'
    train_dataset = TripletMarket1501Dataset('/media/rootadminwalker/DATA/datasets/Market-1501-v15.09.15',
                                             device='cuda:0',
                                             batch_size=32,
                                             triplets_per_anchor=12, transforms=None)

    person_reidentification = PersonReidentification(
        bin_path='/home/rootadminwalker/workspace/ROS_projects/src/The_Essense_of_the_Grey_Region/models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.bin',
        xml_path='/home/rootadminwalker/workspace/ROS_projects/src/The_Essense_of_the_Grey_Region/models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml'
    )

    image_selector = ImageSelector(image_size=(128, 64, 3))

    output_file = './similar_identities.json'
    output_dict = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as cfg:
            output_dict = json.load(cfg)

    query_descriptors = {}
    print('Loading the query descriptors...\n')
    for idx, data in enumerate(train_dataset.dataset['labels_to_path_idx'].items()):
        if (idx + 1) < start:
            continue

        label, relatives = data
        random_idx = random.choice(relatives)
        query_image_path = train_dataset.dataset['image_paths'][random_idx]
        query_image = cv.imread(query_image_path)
        query_descriptor = person_reidentification.parse_descriptor(query_image).squeeze(0)
        query_descriptors.update({label: query_descriptor})
        print('\r', end='')
        print(f'{idx + 1}/{len(train_dataset.dataset["labels_to_path_idx"].keys())}', end='')

    current_count = 0
    for idx, search_id in enumerate(train_dataset.dataset["labels_to_path_idx"].keys()):
        current_count = idx + 1
        if search_id.lower() in ['exit', 'quit']:
            break

        print(f"{current_count}: Searching for identity {search_id}...")
        if search_id not in query_descriptors:
            print('Identity not found in query')
            continue

        random_idx = random.choice(train_dataset.dataset['labels_to_path_idx'][search_id])
        image_path = train_dataset.dataset['image_paths'][random_idx]
        search_image = cv.imread(image_path)
        search_descriptor = person_reidentification.parse_descriptor(search_image)
        calculated_dists = {}
        for token, query_descriptor in query_descriptors.items():
            if token == search_id:
                continue

            similarity = person_reidentification.compare_descriptors(search_descriptor, query_descriptor)
            calculated_dists.update({f"with {token}": similarity})

        top_k = 20
        print('\r', end='')
        print(f"Found top {top_k} ones, from left to right:")
        top_k_results = {
            k: v for k, v in sorted(calculated_dists.items(), key=lambda item: item[1], reverse=True)[:top_k]
        }
        ranked_images = {}
        for with_another, result in top_k_results.items():
            token = with_another.split()[-1]
            random_idx = random.choice(train_dataset.dataset['labels_to_path_idx'][token])
            image_path = train_dataset.dataset['image_paths'][random_idx]
            image = cv.imread(image_path)
            ranked_images[token] = image
            print(token, end=' ')
        print()

        selection_status, force_end = image_selector.select(search_image, list(ranked_images.values()))
        if force_end:
            break

        keep_identities = []
        for identity, selection in zip(ranked_images.keys(), selection_status):
            if not selection:
                keep_identities.append(identity.split()[-1])

        output_dict[search_id] = keep_identities

    with open(output_file, 'w+') as f:
        json.dump(output_dict, f, indent=4)

    if current_count < len(train_dataset.dataset["labels_to_path_idx"]):
        with open('./current_count.json', 'w+') as f:
            json.dump(current_count, f)

    cv.destroyAllWindows()

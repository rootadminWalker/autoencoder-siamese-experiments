import random

import cv2 as cv
import numpy as np

from datasets import SiameseMarket1501Dataset


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
    device = 'cuda:0'
    train_dataset = SiameseMarket1501Dataset('/media/rootadminwalker/DATA/datasets/Market-1501-v15.09.15',
                                             device='cuda:0',
                                             batch_size=32,
                                             pairs_per_image=12, transforms=None)

    person_reidentification = PersonReidentification(
        bin_path='/home/rootadminwalker/workspace/ROS_projects/src/The_Essense_of_the_Grey_Region/models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.bin',
        xml_path='/home/rootadminwalker/workspace/ROS_projects/src/The_Essense_of_the_Grey_Region/models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml'
    )
    query_descriptors = {}
    print('Loading the query descriptors...\n')
    for idx, data in enumerate(train_dataset.dataset['labels_to_path_idx'].items()):
        label, relatives = data
        random_idx = random.choice(relatives)
        query_image_path = train_dataset.dataset['image_paths'][random_idx]
        query_image = cv.imread(query_image_path)
        query_descriptor = person_reidentification.parse_descriptor(query_image).squeeze(0)
        query_descriptors.update({label: query_descriptor})
        print('\r', end='')
        print(f'{idx + 1}/{len(train_dataset.dataset["labels_to_path_idx"].keys())}', end='')

    for search_token in train_dataset.dataset["labels_to_path_idx"].keys():
        if search_token.lower() in ['exit', 'quit']:
            break

        print(f"Searching for identity {search_token}...")
        if search_token not in query_descriptors:
            print('Identity not found in query')
            continue

        random_idx = random.choice(train_dataset.dataset['labels_to_path_idx'][search_token])
        image_path = train_dataset.dataset['image_paths'][random_idx]
        search_image = cv.imread(image_path)
        search_descriptor = person_reidentification.parse_descriptor(search_image)
        calculated_dists = {}
        for token, query_descriptor in query_descriptors.items():
            if token == search_token:
                continue

            similarity = person_reidentification.compare_descriptors(search_descriptor, query_descriptor)
            calculated_dists.update({f"with {token}": similarity})

        k = 20
        print('\r', end='')
        print(f"Found top {k} ones, from left to right:")
        top_k_results = {k: v for k, v in sorted(calculated_dists.items(), key=lambda item: item[1], reverse=True)[:k]}
        ranked_images = []
        for with_another, result in top_k_results.items():
            token = with_another.split()[-1]
            random_idx = random.choice(train_dataset.dataset['labels_to_path_idx'][token])
            image_path = train_dataset.dataset['image_paths'][random_idx]
            image = cv.imread(image_path)
            ranked_images.append(image)
            print(token, end=' ')

        cv.imshow('original', search_image)
        cv.imshow('similar ones', np.hstack(ranked_images))
        cv.waitKey(0)
        remove_idxes = input("\nWhich are those DON't want to keep? (Separate by space, use order from 1) -- ").split()
        for remove_idx in remove_idxes:
            del top_k_results[list(top_k_results.keys())[int(remove_idx) - 1]]

        print(f"{search_token}: {list(top_k_results.keys())}")
        cv.destroyAllWindows()

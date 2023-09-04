# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import onnxruntime as ort

from projectp.inference import InferenceONNX


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="yolov5s-no-nms-2022-07-07-last.onnx")
        self.labels = labels

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = InferenceONNX(model, providers=providers, sess_options=so)
            # self.output_details = [i.name for i in self.model.get_outputs()]
            # self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def infer(self, image, threshold):
        image = np.array(image)  # PIL -> numpy
        image = image[:, :, ::-1].copy()  # RGB -> BGR
        h, w, _ = image.shape
        detections, _, latency = self.model.process_image(image, confidence=threshold, save=False)

        results = []
        print(f"Detections shape = {detections.shape}, done in {latency['total']:.2f} sec")
        if len(detections):
            boxes = detections[:, 1:5]
            labels = detections[:, 6]
            scores = detections[:, 5]

            for label, score, box in zip(labels, scores, boxes):
                xtl = max(int(box[0]), 0)
                ytl = max(int(box[1]), 0)
                xbr = min(int(box[2]), w)
                ybr = min(int(box[3]), h)
                print(f"DEBUG: label = {label}")

                results.append({
                    "confidence": str(score),
                    "label": self.labels.get(int(label), "unknown"),
                    "points": [xtl, ytl, xbr, ybr],
                    "type": "rectangle",
                })
            print(f"DEBUG: labels = {self.labels}")

        return results

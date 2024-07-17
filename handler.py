"""A custom model handler. You should generally derive from `BaseHandler` (or sometimes `ImageClassifier` or
`VisionHandler` handlers for image classification models) and ONLY override methods whose behavior needs to change. Most
of the time you only need to override `preprocess` or `postprocess`. For illustrative purposes, we derive from the
`BaseHandler` to give examples of both custom `preprocess` and `postprocess` methods.

For more information, refer to https://pytorch.org/serve/custom_service.html#custom-handlers and the `ImageClassifier`
and `VisionHandler` implementations in the TorchServe repository:

    - https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py
    - https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py
"""

import base64
import io

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from PIL import Image
from ts.handler_utils.timer import timed
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import map_class_to_label


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    topk = 5
    # These are the standard Imagenet dimensions and statistics
    image_processing = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: Initial context contains model server system properties.
        :return:
        """
        super(ModelHandler, self).initialize(context)

        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None

    @timed
    def preprocess(self, data):
        """
        Transform raw input into a float tensor and apply the image preprocessing steps.

        :param data: The list of raw requests to the model, should match batch size
        :return: A float tensor of preprocessed images
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data directly, but older versions of TorchServe
            # didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # If the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    @timed
    def postprocess(self, inference_output):
        """
        Return inference result. Take output from network and post-process to desired format.

        :param inference_output: list of inference output
        :return: list of predict results
        """
        ps = F.softmax(inference_output, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        return map_class_to_label(probs, self.mapping, classes)

import os
from typing import Tuple

import cv2
import torch
import numpy as np

from src import models


def pad_input(input: torch.Tensor, times: int=32):
    """
    Pad to multiple of 32 to ensure it fits the model

    :param input: Input to model [b, c, h, w]
    :param times: Enforce input to be multiple of this
    """
    input_h, input_w = input.shape[-2: ]
    pad_h = pad_w = 0
    if input_h % times != 0:
        pad_h = times - (input_h % times)
    if input_w % times != 0:
        pad_w = times - (input_w % times)
    input = torch.nn.functional.pad(input, (0, pad_w, 0, pad_h), mode='reflect')
    return input


def crop_result(result: torch.Tensor, input_hw: Tuple[int, int], times=32):
    """
    Removes pad added at the start during pre-processing

    :param result: Result produced by the model
    :param input_hw: Tuple containing input image height and width
    :param times: A value used to enforce the input shape to be a multiple of
    """
    crop_h = crop_w = 0
    input_h, input_w = input_hw

    if input_h % times != 0:
        crop_h = times - (input_h % times)
    if input_w % times != 0:
        crop_w = times - (input_w % times)

    if crop_h != 0:
        result = result[..., :-crop_h, :]
    if crop_w != 0:
        result = result[..., :-crop_w]
    return result


def pre_process(image: np.array, device: str) -> torch.Tensor:
    """
    Transforms input BGR image to something that is accepted by the model

    :param image: Input image to transform to the model input
    :param device: Device to send input to
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(image.astype(np.float32)) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1))
    image = image.to(device)
    image = image.unsqueeze(0)
    image = pad_input(image)
    return image


def post_process(model_output: torch.Tensor, input_hw: Tuple[int, int]) -> np.array:
    """
    Transforms model output into something accepted by OpenCV

    :param model_output: Output tensor produced by the model [b, c, h, w]
    :param input_hw: Tuple containing input image height and width
    :returns: Output image which can be displayed by OpenCV
    """
    model_output = crop_result(model_output, input_hw=input_hw)

    b, c, h, w = model_output.shape
    in_h, in_w = input_hw
    if h != in_h or w != in_w:
        model_output = torch.nn.functional.interpolate(model_output, input_hw, mode='bicubic')

    model_output = model_output * 255.0
    model_output = model_output.permute(0, 2, 3, 1)
    model_output = model_output.cpu().squeeze().numpy().clip(0, 255).astype(np.uint8)
    model_output = cv2.cvtColor(model_output, cv2.COLOR_RGB2BGR)

    return model_output


if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/29_hazy_video.mp4"
    device = "cpu"

    weights_path = "weights_reconstruct_REVIDE.pth"

    base_channel_num = 96 if "Light" not in weights_path else 64
    # Params according to checkpoints yml files
    model = models.HazeRemovalNet(
        base_channel_nums=base_channel_num,
        min_beta=0.2,
        max_beta=0.1,
        norm_type="instance",
    )
    weights = torch.load(weights_path, map_location="cpu")["net_h2c"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)

    while True:
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        h, w, _ = frame.shape

        in_tensor = pre_process(frame, device)

        with torch.no_grad():
            model_out = model(in_tensor, False)

        out_image = post_process(model_out, (h, w))

        cv2.imshow("frame", display_frame)
        cv2.imshow("out_image", out_image)

        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

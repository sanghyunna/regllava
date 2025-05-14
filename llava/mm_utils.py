from PIL import Image
from io import BytesIO
import base64
import os
from typing import List
import torch
from torchvision.transforms import Compose
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images: List[Image.Image], image_processor, model_cfg): # images 타입 힌트 추가
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []

    if image_aspect_ratio == 'pad':
        for image in images:
            background_color = tuple(int(x*255) for x in getattr(image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073]))
            image = expand2square(image, background_color)
            
            if hasattr(image_processor, 'preprocess'): # 표준 HF ImageProcessor
                # preprocess는 보통 단일 이미지를 받고, 결과를 딕셔너리로 반환하며, 그 안에 'pixel_values'가 배치 차원 없이 있음
                processed_image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif isinstance(image_processor, Compose): # torchvision.transforms.Compose 객체
                # Compose 객체는 보통 단일 이미지를 받아 단일 텐서(C, H, W)를 반환
                processed_image = image_processor(image)
                # reg_gated_transform이 (C,H,W) 텐서를 반환한다고 가정. 만약 다른 형태면 추가 처리 필요.
            else:
                raise TypeError(f"Unsupported image_processor type: {type(image_processor)}")
            new_images.append(processed_image)

    elif image_aspect_ratio == "anyres":
        for image in images:
            # process_anyres_image 내부에서도 image_processor 타입에 따른 분기 처리가 필요함
            # 여기서는 process_anyres_image가 이를 처리한다고 가정하고 호출
            # 또는 process_anyres_image에 image_processor가 Compose인지 여부를 전달할 수 있음
            processed_image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints, model_cfg) # model_cfg 전체 전달 고려
            new_images.append(processed_image)
            
    else: # 'pad'나 'anyres'가 아닌 경우 (예: None 또는 다른 값)
        if hasattr(image_processor, 'preprocess') and callable(getattr(image_processor, 'preprocess', None)):
            # 표준 HF ImageProcessor: 이미지 리스트를 받아 처리하고 'pixel_values'로 딕셔너리 반환 기대
            # (주의: HF ImageProcessor의 __call__ 메소드가 preprocess를 호출하는 경우가 많음)
            # 가장 안전한 것은 ImageProcessor의 __call__ 인터페이스를 따르는 것
            # image_processor(images, return_tensors='pt')는 보통 {'pixel_values': (B, C, H, W)}를 반환
            # 이 경우, return image_processor(images, return_tensors='pt')['pixel_values']가 맞음
            # 하지만 CLIPImageProcessor.__call__은 단일 이미지를 받아 pixel_values: (1,C,H,W) 를 반환 후 squeeze(0)
            # 만약 images가 리스트라면, 반복 처리 필요
            
            # 우선, image_processor가 리스트를 직접 처리할 수 있는지 확인 (HF ImageProcessor의 일반적인 사용법)
            try:
                # image_processor가 __call__ 메소드에서 리스트와 return_tensors를 지원한다고 가정
                return image_processor(images, return_tensors='pt')['pixel_values']
            except Exception: # 만약 실패하면, 개별 처리
                for image in images:
                    # 단일 이미지에 대한 preprocess 호출 (결과는 (C,H,W) 가정)
                    new_images.append(image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0])

        elif isinstance(image_processor, Compose): # torchvision.transforms.Compose 객체
            for image in images:
                # Compose 객체는 단일 이미지를 받아 단일 텐서(C, H, W)를 반환
                new_images.append(image_processor(image))
        else:
            raise TypeError(f"Unsupported image_processor type for general case: {type(image_processor)}")

    # 모든 이미지가 동일한 형태인지 확인 후 스택 (이 부분은 유지)
    if new_images and all(x.shape == new_images[0].shape for x in new_images):
        new_images_stacked = torch.stack(new_images, dim=0) # (B, C, H, W)
        return new_images_stacked
    elif not new_images: # 처리된 이미지가 없는 경우
        # 빈 리스트를 반환하거나, 빈 텐서를 반환하거나, 오류를 발생시킬 수 있음
        # 여기서는 빈 텐서를 반환하는 예시 (형태는 상황에 맞게 조정)
        # model_cfg에서 dtype 가져오기
        target_dtype = getattr(model_cfg, 'torch_dtype', torch.float32)
        return torch.empty((0, 3, getattr(image_processor, 'crop_size', {}).get('height', 224), getattr(image_processor, 'crop_size', {}).get('width', 224)), dtype=target_dtype)
    else: # 스택할 수 없는 경우 (이미지 크기가 다를 때) - 이 경우는 로직상 발생하면 안됨
        # 또는 리스트 그대로 반환 (LLaVA의 다른 부분이 리스트를 처리할 수 있다면)
        # raise ValueError("Processed images have different shapes and cannot be stacked.")
        return new_images # 혹은 오류 발생


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)

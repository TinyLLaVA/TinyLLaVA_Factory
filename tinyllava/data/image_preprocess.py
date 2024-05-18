import os

from PIL import Image, ImageFile
import torch
import ast

from ..utils.data_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 可能imagepreprocess需要继承一个huggingface的图像处理类？提供from_pretrained方法

class ImagePreprocess:
    def __init__(self, image_processor, data_args={}):
        self.image_aspect_ratio = getattr(data_args, 'image_aspect_ratio', None)
        self.image_processor = image_processor
        self.image_grid_pinpoints = getattr(data_args, 'image_grid_pinpoints', None)
    
    def __call__(self, image):
        if self.image_aspect_ratio == 'pad':
            image = self.expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        elif self.image_aspect_ratio == "anyres":
            image = self.process_anyres_image(image, self.image_processor, self.image_grid_pinpoints)
            return image
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        return image

    @classmethod
    def expand2square(cls, pil_img, background_color):
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
    
    @classmethod
    def process_anyres_image(cls, image, processor, grid_pinpoints):
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
        image_patches = [processor(image_patch, return_tensors='pt')['pixel_values'][0]
                        for image_patch in image_patches]
        return torch.stack(image_patches, dim=0)
    

import torch
from typing import Tuple, Union
import torchvision.transforms as T
from PIL import Image
from skimage import data


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: Tuple[float, float]):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different random resized crop to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images


class DeviceAgnosticRandomRotation(T.RandomRotation):
    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        """This is the same as T.RandomRotation but it only accepts batches of images and works on GPU"""
        super().__init__(degrees=degrees)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies random rotation to each image
        random_rotation = super(DeviceAgnosticRandomRotation, self).forward
        augmented_images = [random_rotation(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images


class DeviceAgnosticCutout:
    def __init__(self, size: int, p: float = 0.5, constant_fill: Union[int, Tuple[int, int, int]] = 0):
        self.size = size
        self.p = p
        self.constant_fill = constant_fill
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        mask = torch.ones((B, 1, H, W), dtype=torch.float32)
        if torch.rand(1) < self.p:
            # Generate random top-left coordinates for the cutout region
            x = torch.randint(0, W - self.size + 1, (B,))
            y = torch.randint(0, H - self.size + 1, (B,))
            # Apply cutout by setting the region to zeros or a constant value
            for i in range(B):
                mask[i, :, y[i]:y[i] + self.size, x[i]:x[i] + self.size] = self.constant_fill
        # Apply the cutout mask to the images
        masked_images = images * mask.to(images.device)
        return masked_images


if __name__ == "__main__":
    # Initialize augmentations
    random_crop = DeviceAgnosticRandomResizedCrop(size=(256, 256), scale=(0.5, 1))
    random_rotation = DeviceAgnosticRandomRotation(degrees=(-15, 15))
    color_jitter = DeviceAgnosticColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    cutout = DeviceAgnosticCutout(size=32, p=0.5, constant_fill=0)
    
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    
    # Apply augmentations (individually on each of the 2 images)
    augmented_batch = random_crop(images_batch)
    augmented_batch = random_rotation(augmented_batch)
    augmented_batch = color_jitter(augmented_batch)
    augmented_batch = cutout(augmented_batch)
    
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    
    # Visualize the original image, as well as the two augmented ones
    pil_image.show()
    augmented_image_0.show()
    augmented_image_1.show()
import glob
from torchvision import transforms
from diffusers import AutoencoderKL
import torch
from PIL import Image
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA

class ImagePlotter:
    def __init__ (self, rows, columns, width, height):
        self.rows = rows
        self.columns = columns
        plt.figure(figsize=(width, height))
        
    def plot_image(self, image, position, title, cmap=None, axis="off"):
        plt.subplot(self.rows, self.columns, position)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis(axis)
        
    def display_plot(self):
        plt.show()

class LatentImageModifier:
    def __init__(self, image_path, model_path="./models", device="mps"):
        self._device = device 
        self._vae = AutoencoderKL.from_pretrained(model_path).to(self._device)
        self.images = [Image.open(img_path).convert("RGB") for img_path in glob.glob(image_path)]
        self.latent_images = []
        
    def trim_images(self, target_size=None, resampling_filter=Image.Resampling.LANCZOS):
        if target_size is None:
            min_width, min_height = min(image.size for image in self.images)
            target_size = (min_width, min_height)

        self.images = [image.resize(target_size, resampling_filter) for image in self.images]   
        
    def encode_images(self):
        preprocess = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Normalize([0.5], [0.5])])
        for image in self.images:
            with torch.no_grad():
                self.latent_images.append(             
                    self._vae.encode(preprocess(image)
                                                .unsqueeze(0)
                                                .to(self._device)
                                                ).latent_dist.sample() * 0.18215)
  
    def decode_image(self, latent_image):
        with torch.no_grad():
            decoded_image = self._vae.decode(latent_image / 0.18215).sample
                
        image = Image.fromarray(( 
                    ( 
                        (decoded_image.cpu() / 2 + 0.5).clamp(0, 1) \
                        .permute(0, 2, 3, 1).float().numpy()[0]
                    ) * 255).round().astype("uint8"))
        
        self.images.append(image)
        self.latent_images.append(latent_image)
        return image

    def modify(self, modify_type="average", **kwargs):
        options = {
            "average": self._average,            
            "weighted_average": lambda: self._weighted_average(kwargs["latent_1"], kwargs["latent_2"], kwargs.get("ratio", 0.5)),
            "combine_latent_halves": lambda: self._combine_latent_halves(kwargs["latent_1"], kwargs["latent_2"], kwargs.get("split_direction", 'vertical')),
            "pca": lambda: self._pca(kwargs.get("n_components", 2), kwargs.get("latent_images", None))
        }
        
        if modify_type not in options:
            raise ValueError(f"not a modify_type: {modify_type}")
        
        return options[modify_type]()

    def _average(self):
        return torch.stack(self.latent_images).mean(dim=0).to(self._device)


    def _weighted_average(self, latent_1, latent_2, ratio):
        if latent_1.shape != latent_2.shape:
            raise ValueError("Latent shapes don't match")
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be between 0 and 1")
        return ((1 - ratio) * latent_1 + ratio * latent_2).to(self._device)


    def _combine_latent_halves(self, latent_1, latent_2, split_direction = 'vertical'):
        if latent_1.shape != latent_2.shape:
            raise ValueError(f"latent shapes don't match")
        
        b, c, h, w = latent_1.shape
        
        if split_direction == 'vertical':
            midpoint_w = w // 2
            combined = torch.zeros_like(latent_1)
            combined[..., :midpoint_w] = latent_1[..., :midpoint_w]
            combined[..., midpoint_w:] = latent_2[..., midpoint_w:]
        else:  #horizontal
            midpoint_h = h // 2
            combined = torch.zeros_like(latent_1)
            combined[..., :midpoint_h, :] = latent_1[..., :midpoint_h, :]
            combined[..., midpoint_h:, :] = latent_2[..., midpoint_h:, :]
        
        return combined.to(self._device)


    def _pca(self, n_components=2, latent_images=None):
        if latent_images is None:
            latent_images = self.latent_images
            
        shapes = [latent.shape for latent in self.latent_images]
        if len(set(shapes)) > 1:
            raise ValueError(f"latent shapes are not consistent: {shapes}")

        latents_tensor = torch.stack(self.latent_images)
        original_shape = latents_tensor.shape[1:]  # C, H, W
        latents_np = latents_tensor.cpu().numpy().reshape(len(latent_images), -1)

        if n_components > latents_np.shape[1]:
            raise ValueError(f"n_components ({n_components}) exceeds number of features.")

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(latents_np)

        reduced_avg = reduced.mean(axis=0)
        reconstructed_tensor = torch.tensor(pca.inverse_transform(reduced_avg), dtype=torch.float32).reshape(1, *original_shape).to(self._device)
    
        return reconstructed_tensor.squeeze(1).to(self._device)  


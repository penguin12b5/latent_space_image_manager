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
        plt.tight_layout()
        plt.show()

class LatentImageModifier:
    def __init__(self, image_path, model_path="./models", device="mps"):
        self._device = device 
        self._vae = AutoencoderKL.from_pretrained(model_path).to(self._device)
        self.images = [Image.open(img_path).convert("RGB") for img_path in glob.glob(image_path)]
        self.latent_images = []
        
    #resize image so that they are al the same size
    def resize_images(self, target_size=None, resampling_filter=Image.Resampling.LANCZOS):
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


    #dispatch table probably not nessesary
    def modify(self, modify_type="none", **kwargs):
        options = {
            "none": self._no_mod,
            "average": self._average,            
            
            "weighted_average": lambda: self._weighted_average(kwargs["latent_1"], 
                                                               kwargs["latent_2"], 
                                                               kwargs.get("ratio", 0.5)),
            
            "combine_latent_halves": lambda: self._combine_latent_halves(kwargs["latent_1"], 
                                                                         kwargs["latent_2"], 
                                                                         kwargs.get("split_direction", 'vertical')),
            
            "pca": lambda: self._pca(kwargs.get("n_components", 2), 
                                     kwargs.get("latent_images", None))
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


    #takes two opposite halves of latent images and combines them
    def _combine_latent_halves(self, latent_1, latent_2, split_direction = 'vertical'):
        if latent_1.shape != latent_2.shape:
            raise ValueError(f"latent shapes don't match")
        
        b, c, h, w = latent_1.shape
        
        #split and combine vertically
        if split_direction == 'vertical': 
            midpoint_w = w // 2
            combined = torch.zeros_like(latent_1)
            combined[..., :midpoint_w] = latent_1[..., :midpoint_w]
            combined[..., midpoint_w:] = latent_2[..., midpoint_w:]
        else:  #split anc combine horizontally
            midpoint_h = h // 2
            combined = torch.zeros_like(latent_1)
            combined[..., :midpoint_h, :] = latent_1[..., :midpoint_h, :]
            combined[..., midpoint_h:, :] = latent_2[..., midpoint_h:, :]
        
        return combined.to(self._device)


    #apply PCA to latent images
    def _pca(self, n_components, latent_images=None):
        if latent_images is None:
            latent_images = self.latent_images
             
        #check if all latent tensors have the same shape
        shapes = [latent.shape for latent in latent_images]
        if len(set(shapes)) > 1:
            raise ValueError(f"latent shapes are not consistent: {shapes}")
        
        #stack latent tensors into a single tensor 
        latents_tensor = torch.stack(latent_images)
        
        #save the original shape for reconstruction later
        original_shape = latents_tensor.shape[1:] 
        
        #convert tensor to NumPy and flatten each latent image into a 1D vector
        latents_np = latents_tensor.cpu().numpy().reshape(len(latent_images), -1)

        if n_components > latents_np.shape[1]:
            raise ValueError(f"n_components ({n_components}) exceeds number of features.")

        #apply PCA to reduce the dimensionality
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(latents_np)
        
        #combine the reduced representations 
        reduced_avg = reduced.mean(axis=0)
        
        #reconstruct vector
        reconstructed_tensor = torch.tensor(pca.inverse_transform(reduced_avg), dtype=torch.float32).reshape(1, *original_shape).to(self._device)
    
        return reconstructed_tensor.squeeze(1).to(self._device)  



    #encodes the first image
    def _no_mod(self):
        if not self.images:
            raise ValueError("No images loaded.")

        image = self.images[0]
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        with torch.no_grad():
            tensor = preprocess(image).unsqueeze(0).to(self._device)
            latent = self._vae.encode(tensor).latent_dist.sample() * 0.18215

        return latent





    '''
    compare diffusion process on pca compressed image vs latent representation
    other ideas:
    - encode latent images?
    - add noise to latent images
    - reduce color channels to see how the decoder treats values
    - compare latent img and pixel image on diffusion process and note quality differences, such as which parts have more or less details
      and possibly find a way to edit the latent representation, such as for example if areas with low contrast tend to lose detail 
      after being decoded, boost the contrast of those areas in the latent representation
      
    - instead of decoding image all at once, split the image into patches based on detail. for example, if you have an image
      of a beach that is 64 wide x 32 tall (small resolution to keep thing simpel for this exmaple), and in the top half 
      you have a sky, with little to no clouds, and no visual interest or focal point, we can treat that entire section as just
      a single blue block. whether we spend a lot of effort or no effort on trying to decode that single part doesnt matter
      since its just a flat color and will probably look the same either way. if the latent representation is 16 wide x 8 tall, 
      the top half would be 8 wide x 4 tall, but can be represetned with just 2 x 1 instead. other parts of the latent image, such as the
      sand on the beach can be simplified the same way. aditioannlly if there was an area of low contrast with a lot of details,
      the contrast could be boosted during the encoding process. for example, if pixel 1 had a value of 10 and pixel 2 had a value of 15,
      since these are similar you can boost the value of pixel 2 based on the difference for exmpale pixel 2 value += ((p2 - p1) divided by some number) 
      or just apply the normal contrast function in photo editing apps. i don't know how possible or efficient/ineficcent this is since you would 
      have to go through all pixels, but just looking at the latent representation of the dog image this seems kind of possible. also doing this would 
      utilize the resutls of the other experiments, especially the combing halves one. the pca can maybe be used to find parts of the latent 
      representation that have similat features, like the blue sky. this probably won't work for small image, and the size of the latent might have to be 
      increased.
    
    '''

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from diffusers import AutoencoderKL
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from PIL import ImageFilter


class ImageProcessor:
    def __init__(self, detection_threshold=0.8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detection_threshold = detection_threshold
        
        #load models
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained("./models").to(self.device).eval()

    #finds subject and returns all subjects and their coordinates
    def find_subjects(self, image, max_width=1000, max_height=1000):
        image_tensor = transforms.ToTensor()(image).to(self.device)

        with torch.no_grad():
            outputs = self.detection_model([image_tensor])[0]

        boxes, scores = outputs["boxes"], outputs["scores"]
        keep_indices = [i for i, s in enumerate(scores) if s > self.detection_threshold]

        #search for subjects
        subjects = []
        for i in keep_indices:
            x1, y1, x2, y2 = map(int, boxes[i].tolist())

            width = x2 - x1
            height = y2 - y1

            #crop subject if it is too big
            if width <= max_width and height <= max_height:
                subject = image.crop((x1, y1, x2, y2))
                subjects.append((subject, (x1, y1)))
            else:
                h_splits = (width + max_width - 1) // max_width  
                v_splits = (height + max_height - 1) // max_height

                tile_width = width / h_splits
                tile_height = height / v_splits

                for row in range(v_splits):
                    for col in range(h_splits):
                        left = int(x1 + col * tile_width)
                        upper = int(y1 + row * tile_height)
                        right = int(min(left + tile_width, x2))
                        lower = int(min(upper + tile_height, y2))

                        tile = image.crop((left, upper, right, lower))
                        subjects.append((tile, (left, upper)))

        if not subjects:
            print("no subjects found")
            return None

        return subjects

    def encode(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            return self.vae.encode(image_tensor).latent_dist.sample()
    
    def decode(self, latent):
         with torch.no_grad(): 
             return self.vae.decode(latent).sample
    
    def resize_image(self, image, scale=1.0):
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height))
    
    #makes an image/latent displayable on a plot
    def to_displayable(self, image, is_latent=False):
        if is_latent:
            return image.squeeze(0).mean(0).cpu()

        image = (image.clamp(-1, 1) + 1) / 2
        return image.squeeze(0).permute(1, 2, 0).cpu().numpy()   

class ImageDisplayer:
    def __init__(self):
        self.axes = None
        self.figure = None

    def create_plot(self, rows, columns):
        self.figure, self.axes = plt.subplots(rows, columns)
        
        if rows == 1 and columns == 1:
            self.axes = np.array([[self.axes]])
        elif rows == 1:
            self.axes = np.array([self.axes])
        elif columns == 1:
            self.axes = np.array([[ax] for ax in self.axes])

    def add_to_plot(self, image, row, column, title=None, cmap=None, axis='off'):
        ax = self.axes[row][column]
        print(f"{image.shape} {type(image)}")
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis(axis)

    def make_tight_layout(self):
        self.figure.tight_layout()

    def display_plot(self):
        plt.show()
        
def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu()
    return to_pil_image((tensor.clamp(-1, 1) + 1) / 2)
    
def load_image(file_path):
        image = Image.open(file_path).convert("RGB")
        return image
    
def edge_fade_mask(width, height, fade_px):
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]

    dist_left   = x
    dist_right  = (width - 1) - x
    dist_top    = y
    dist_bottom = (height - 1) - y

    dist_edge = np.minimum(np.minimum(dist_left, dist_right),
                           np.minimum(dist_top, dist_bottom)).astype(np.float32)

    alpha = np.clip(dist_edge / max(1.0, float(fade_px)), 0.0, 1.0)

    alpha = np.power(alpha, 1.5)

    mask = (alpha * 255.0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


if __name__ == "__main__":
    ''''
    test B: 
    image -> resize_down -> encode -> resize_up = full_latent_resized;
    subject -> encode -> subj_latent;
    paste subj_latent onto full_latent_resized (with or without blending);
    decode
    '''
    
    p = ImageProcessor()
    d = ImageDisplayer()
    scale = 0.25

    #find and store subjects
    image = load_image("images/dog2.jpeg")
    subjects = p.find_subjects(image)

    #backgound: encode resized, decode, upsize
    encoded_resized_image = p.encode(p.resize_image(image, scale))
    decoded_resized_image_tensor = p.decode(encoded_resized_image)
    decoded_resized_image_pil = tensor_to_pil(decoded_resized_image_tensor)
    merged_image = p.resize_image(decoded_resized_image_pil, 1 / scale).copy()

    d.create_plot(1, 2)
    d.add_to_plot(np.array(image), 0, 0, title="Original")

    for idx, (subject_image, (x1, y1)) in enumerate(subjects):
        encoded_subject = p.encode(subject_image)
        decoded_subject_tensor = p.decode(encoded_subject)
        decoded_subject_img = tensor_to_pil(decoded_subject_tensor)

        w, h = decoded_subject_img.size
        fade_px = int(0.08 * min(w, h)) 
        mask = edge_fade_mask(w, h, fade_px)

        merged_image.paste(decoded_subject_img, (int(x1), int(y1)), mask)
        print(f"subject {idx+1} blended at ({x1},{y1}), fade {fade_px}px")


    d.add_to_plot(np.array(merged_image), 0, 1, title="merged + fade")

    d.make_tight_layout()
    d.display_plot()


        
""" 
 #image -> resized_down(image) -> encode -> resize_up(latent) -> decode   
if __name__ == "__main__":
    p = ImageProcessor()
    d = ImageDisplayer()
    scale = 0.25

    image = load_image("images/dog2.jpeg")
    subjects = p.find_subjects(image)

    encoded_resized_image = p.encode(p.resize_image(image, scale))
    print(f"shape of encoded resized image: {encoded_resized_image.shape}")

    #esize latent back up to original resolution
    full_latent_resized = F.interpolate(
        encoded_resized_image,
        scale_factor=1/scale,
        mode="bilinear",
        align_corners=False
    )

    merged_latent = full_latent_resized.clone()

    for idx, (subject_image, (x1, y1)) in enumerate(subjects):
        encoded_subject = p.encode(subject_image)
        
        top = y1 // 8
        left = x1 // 8

        merged_latent = paste_latent(merged_latent, encoded_subject, top, left)
        print(f"subject latent {idx+1} pasted at coords ({top}, {left})")

  
    decoded_merged_tensor = p.decode(merged_latent)
    decoded_merged_pil = tensor_to_pil(decoded_merged_tensor)


    d.create_plot(1, 3)
   
    #original image
    d.add_to_plot(np.array(image), 0, 0, title="original")
    #latent visualization 
    merged_latent = p.to_displayable(merged_latent, is_latent=True)
    d.add_to_plot(merged_latent, 0, 1, title="merged latent", cmap="viridis")
    #final decoded image
    decoded_image = p.to_displayable(decoded_merged_tensor)
    d.add_to_plot(decoded_image, 0, 2, title="merged")

    d.make_tight_layout()
    d.display_plot()

if __name__ == "__main__":
    p = ImageProcessor()
    d = ImageDisplayer()
    scale = 0.25

    image = load_image("images/dog2.jpeg")
    subjects = p.find_subjects(image)
    
    encoded_resized_image = p.encode(p.resize_image(image, scale))
    print(f"shape of encoded resized image: {encoded_resized_image.shape} ")
    
    decoded_resized_image_tensor = p.decode(encoded_resized_image)
    decoded_resized_image_pil = tensor_to_pil(decoded_resized_image_tensor)
    resized_decoded_image_pil = p.resize_image(decoded_resized_image_pil, 1 / scale)

    merged_image = resized_decoded_image_pil.copy()

    d.create_plot(2, 1)

    d.add_to_plot(p.to_displayable(decoded_resized_image_tensor), 1, 0, title="image")

    for idx, (subject_image, (x1, y1)) in enumerate(subjects):
        encoded_subject = p.encode(subject_image)
        decoded_subject_tensor = p.decode(encoded_subject)
        decoded_subject_img = p.to_displayable(decoded_subject_tensor)
        merged_image.paste(subject_image, (int(x1), int(y1)))
        print(f"encoded_subject: {encoded_subject.shape}")
        print(f"subject chunk processed: {idx + 1}")

    merged_image_np = np.array(merged_image) / 255.0
    d.add_to_plot(merged_image_np, 0, 0, title="Merged Image")

    #alpha = 1/(a*dist_to_subj^2 + b)
    #if dist_to_subj > epsilon: alpha=0
    #pixel(x,y) = subject_pixel(x,y)*(1-alpha) + bg_pixel(x,y)*alpha

    #image -> resized_down(image) -> encode -> resize_up(latent) -> decode

    '''
    
    '''


    d.make_tight_layout()
    d.display_plot()
    
"""


import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from torchvision.transforms.functional import to_pil_image
from segment_anything import sam_model_registry, SamPredictor

class ImageProcessor:
    def __init__(self, detection_threshold=0.8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detection_threshold = detection_threshold
        
        #load models
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained("./models").to(self.device).eval()
        
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

class ImageDisplayer:
    def __init__(self, rows=None, cols=None):
        self.axes = None
        self.figure = None
        self.create_plot(rows, cols)

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
    
def get_fade_mask(width, height, fade_px):
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

def get_fade_mask_latent(width, height, fade_px, device="cpu"):
    y = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    x = torch.arange(width, device=device).unsqueeze(0).expand(height, width)

    dist_left   = x
    dist_right  = (width - 1) - x
    dist_top    = y
    dist_bottom = (height - 1) - y

    dist_edge = torch.minimum(torch.minimum(dist_left, dist_right),
                              torch.minimum(dist_top, dist_bottom)).float()

    alpha = torch.clamp(dist_edge / max(1.0, float(fade_px)), 0.0, 1.0)
    alpha = alpha ** 1.5 

    return alpha.unsqueeze(0).unsqueeze(0)  

def get_sam_mask(image, box, sam_checkpoint, sam_model_type="vit_h", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x1, y1, x2, y2 = map(int, box)
    #set full image for predictor
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(image))

    input_box = np.array([[x1, y1, x2, y2]])
    masks, scores, logits = predictor.predict(box=input_box, multimask_output=False)

    mask = masks[0].astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask, mode='L')
    # rop to box
    mask_cropped = mask_pil.crop((x1, y1, x2, y2)) if mask_pil.size != (x2 - x1, y2 - y1) else mask_pil
    return mask_cropped

def process_image_dod(image_processor, image_displayer, image_path, output_name, scale=0.25):
    #take image and find its subjects
    image = load_image(image_path)
    image_displayer.add_to_plot(np.array(image), 0, 0, title="Original")
    subjects = image_processor.find_subjects(image)

    #resize image by scale, then encode + decode
    decoded_image = image_processor.decode(image_processor.encode(image_processor.resize_image(image, scale)))
    #then resize image back to the original size
    decoded_image = image_processor.resize_image(tensor_to_pil(decoded_image), 1 / scale)

    #for each subject, encode + decode, 
    #then attach to the original image with blend
    for i, (subject_image, (x1, y1)) in enumerate(subjects):
        decoded_subject = p.decode(p.encode(subject_image))
        decoded_subject = tensor_to_pil(decoded_subject)

        w, h = decoded_subject.size
        fade_px = int(0.08 * min(w, h)) 
        mask = get_fade_mask(w, h, fade_px)

        decoded_image.paste(decoded_subject, (int(x1), int(y1)), mask)
        print(f"Subject {i + 1} placed at ({x1},{y1})")

    #display
    image_displayer.add_to_plot(np.array(decoded_image), 0, 1, title="Merged")
    image_displayer.make_tight_layout()
    image_displayer.display_plot()

    #save image
    os.makedirs("outputs", exist_ok=True)
    decoded_image.save(f"outputs/{output_name}.png")

def process_image_sam(image_processor, image_displayer, image_path, output_name, scale=0.25):
    #take image and find its subjects
    image = load_image(image_path)
    image_displayer.add_to_plot(np.array(image), 0, 0, title="Original")
    subjects = image_processor.find_subjects(image)

    #resize image by scale, then encode + decode
    decoded_image = image_processor.decode(image_processor.encode(image_processor.resize_image(image, scale)))
    #then resize image back to the original size
    decoded_image = image_processor.resize_image(tensor_to_pil(decoded_image), 1 / scale)

    #for each subject, encode + decode, then attach to the original image with blend
    for i, (subject_image, (x1, y1)) in enumerate(subjects):
        decoded_subject = image_processor.decode(image_processor.encode(subject_image))
        decoded_subject = tensor_to_pil(decoded_subject)

        # try to use SAM mask if available via environment variable SAM_CHECKPOINT
        sam_checkpoint = os.environ.get('SAM_CHECKPOINT', None)
        if sam_checkpoint:
            try:
                mask = get_sam_mask(image, (x1, y1, x1 + subject_image.width, y1 + subject_image.height), sam_checkpoint)
            except Exception:
                # fallback
                w, h = decoded_subject.size
                fade_px = int(0.08 * min(w, h))
                mask = get_fade_mask(w, h, fade_px)
        else:
            w, h = decoded_subject.size
            fade_px = int(0.08 * min(w, h))
            mask = get_fade_mask(w, h, fade_px)

        decoded_image.paste(decoded_subject, (int(x1), int(y1)), mask)
        print(f"Subject {i + 1} placed at ({x1},{y1})")

    #display
    image_displayer.add_to_plot(np.array(decoded_image), 0, 1, title="Merged")
    image_displayer.make_tight_layout()
    image_displayer.display_plot()

    #save image
    os.makedirs("outputs", exist_ok=True)
    decoded_image.save(f"images/outputs/{output_name}.png")

def process_image_lol(image_processor, image_displayer, image_path, output_name, scale=0.25):
    def paste_latent(base_latent, subject_latent, top, left, fade_ratio=0.08):
        _, _, h, w = subject_latent.shape
        fade_px = int(fade_ratio * min(h, w))

        # torch fade mask
        mask = get_fade_mask_latent(w, h, fade_px, device=subject_latent.device)

        # blend
        base_patch = base_latent[:, :, top:top+h, left:left+w]
        blended_patch = base_patch * (1 - mask) + subject_latent * mask
        base_latent[:, :, top:top+h, left:left+w] = blended_patch

        return base_latent

    #take image and find its subjects
    image = load_image(image_path)
    image_displayer.add_to_plot(np.array(image), 0, 0, title="Original")
    subjects = image_processor.find_subjects(image)

    encoded_image = image_processor.encode(image_processor.resize_image(image, scale))

    image_latent = F.interpolate(
        encoded_image,
        scale_factor=1/scale,
        mode="bilinear",
        align_corners=False
    )

    merged_latent = image_latent.clone()

    for i, (subject_image, (x1, y1)) in enumerate(subjects):
        #for each subject, encode + attach to original image latent
        encoded_subject = p.encode(subject_image)
        
        top = y1 // 8
        left = x1 // 8

        merged_latent = paste_latent(merged_latent, encoded_subject, top, left)
        print(f"subject latent {i + 1} placed at ({top}, {left})")

  
    #decode merged + make displayable
    decoded_merged = tensor_to_pil(p.decode(merged_latent))
    
    #dispaly
    image_displayer.add_to_plot(decoded_merged, 0, 1, title="Merged")
    image_displayer.make_tight_layout()
    image_displayer.display_plot()
    
    #save_image
    os.makedirs("outputs", exist_ok=True)
    decoded_merged.save(f"outputs/{output_name}.png")

p = ImageProcessor()
d = ImageDisplayer(1, 2)
process_image_sam(p, d, "images/car1.png", "car1_results")

  
    

   


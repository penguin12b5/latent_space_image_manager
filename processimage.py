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
        #plt.show()
        pass
        
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

    # Auto-detect model type from checkpoint filename if not explicitly provided
    if sam_model_type == "vit_h" and sam_checkpoint and "vit_b" in sam_checkpoint:
        sam_model_type = "vit_b"

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
    # crop to box
    mask_cropped = mask_pil.crop((x1, y1, x2, y2)) if mask_pil.size != (x2 - x1, y2 - y1) else mask_pil
    return mask_cropped

def save_image(image, output_name):
    output_dir = "images/output"
    os.makedirs(output_dir, exist_ok=True)
    image.save(f"{output_dir}/{output_name}.png")

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
        decoded_subject = image_processor.decode(image_processor.encode(subject_image))
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
    save_image(decoded_image, output_name)

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
    print("number of subjects found:", len(subjects))
    for i, (subject_image, (x1, y1)) in enumerate(subjects):
        decoded_subject = image_processor.decode(image_processor.encode(subject_image))
        decoded_subject = tensor_to_pil(decoded_subject)

        # try to use SAM mask if available via environment variable SAM_CHECKPOINT
        sam_checkpoint = os.environ.get('SAM_CHECKPOINT', None)
        if sam_checkpoint:
            print("Using SAM for mask generation")
            try:
                mask = get_sam_mask(image, (x1, y1, x1 + subject_image.width, y1 + subject_image.height), sam_checkpoint)
            except Exception:
                # fallback
                w, h = decoded_subject.size
                fade_px = int(0.08 * min(w, h))
                mask = get_fade_mask(w, h, fade_px)
        else:
            print("SAM_CHECKPOINT not set, using fade mask")
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
    save_image(decoded_image, output_name)

def process_image_lol(image_processor, image_displayer, image_path, output_name, scale=0.25, mask_type="fade"):

    def paste_latend_with_fade(base_latent, subject_latent, top, left, fade_ratio=0.08):
        _, _, h, w = subject_latent.shape
        fade_px = int(fade_ratio * min(h, w))

        # torch fade mask
        mask = get_fade_mask_latent(w, h, fade_px, device=subject_latent.device)

        # blend
        base_patch = base_latent[:, :, top:top+h, left:left+w]
        blended_patch = base_patch * (1 - mask) + subject_latent * mask
        base_latent[:, :, top:top+h, left:left+w] = blended_patch

        return base_latent

    def paste_latend_with_sam(base_latent, subject_latent, top, left, mask_sam, mode="bilinear"):
        _, _, h, w = subject_latent.shape

        # downsample SAM mask to latent size
        alpha = torch.from_numpy(np.array(mask_sam))[None, None, ...]
        alpha = alpha.to(device=base_latent.device, dtype=base_latent.dtype)
        mask = F.interpolate(
            alpha,
            size=(h, w),
            mode=mode,
            align_corners=False if mode in ("bilinear", "bicubic") else None
        )

        print("before normalization mask min/max:", mask.min().item(), mask.max().item())
        mask = mask / 255.0
        print("after normalization mask min/max:", mask.min().item(), mask.max().item())

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

        match mask_type:
            case "fade":
                merged_latent = paste_latend_with_fade(merged_latent, encoded_subject, top, left)
            case "sam":
                decoded_subject = tensor_to_pil(p.decode(encoded_subject))
                sam_checkpoint = os.environ.get('SAM_CHECKPOINT', None)
                print("Using SAM for mask generation")
                try:
                    mask_pil = get_sam_mask(image, (x1, y1, x1 + decoded_subject.width, y1 + decoded_subject.height), sam_checkpoint)
                    merged_latent = paste_latend_with_sam(merged_latent, encoded_subject, top, left, mask_pil)
                    print(f"Subject latent {i + 1} placed at ({top}, {left}) using SAM mask")
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError("SAM mask generation failed")
            case _:
                raise ValueError(f"Unknown mask type: {mask_type}")

        print(f"subject latent {i + 1} placed at ({top}, {left})")

  
    #decode merged + make displayable
    decoded_merged = tensor_to_pil(p.decode(merged_latent))
    
    #dispaly
    image_displayer.add_to_plot(decoded_merged, 0, 1, title="Merged")
    image_displayer.make_tight_layout()
    image_displayer.display_plot()
    
    #save_image
    save_image(decoded_merged, output_name)


import sys

# Set SAM checkpoint environment variable
if 'SAM_CHECKPOINT' not in os.environ:
    #os.environ['SAM_CHECKPOINT'] = 'models/sam_vit_b_01ec64.pth'
    os.environ['SAM_CHECKPOINT'] = 'models/sam_vit_h_4b8939.pth'

p = ImageProcessor()
d = ImageDisplayer(1, 2)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python processimage.py <method_name> <image_path> <output_name_without_extension> ")
        print("Example: python processimage.py lol_sam images/input/car1.png car1_results")
        sys.exit(1)
    
    method_name = sys.argv[1]
    image_path = sys.argv[2]
    output_name = sys.argv[3]

    if method_name == "sam":
        process_image_sam(p, d, image_path, output_name)
    elif method_name == "lol_fade":
        process_image_lol(p, d, image_path, output_name, mask_type="fade")
    elif method_name == "lol_sam":
        process_image_lol(p, d, image_path, output_name, mask_type="sam")
    elif method_name == "dod":
        process_image_dod(p, d, image_path, output_name)

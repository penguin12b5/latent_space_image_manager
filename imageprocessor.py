import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

class ImageProcessor:
    def __init__(self, detection_threshold=0.8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detection_threshold = detection_threshold
        
        #load models
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained("./models").to(self.device).eval()

    def find_subjects(self, image, max_width=200, max_height=200):
        image_tensor = transforms.ToTensor()(image).to(self.device)

        with torch.no_grad():
            outputs = self.detection_model([image_tensor])[0]

        boxes, scores = outputs["boxes"], outputs["scores"]
        keep_indices = [i for i, s in enumerate(scores) if s > self.detection_threshold]

        subjects = []
        for i in keep_indices:
            x1, y1, x2, y2 = map(int, boxes[i].tolist())

            width = x2 - x1
            height = y2 - y1

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

    def to_displayable(self, image, is_latent=False):
        if is_latent:
            return image.squeeze(0).mean(0).cpu()

        image = (image.clamp(-1, 1) + 1) / 2
        return image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    return to_pil_image(tensor)
    
def load_image(file_path):
        image = Image.open(file_path).convert("RGB")
        return image
    
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
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis(axis)

    def make_tight_layout(self):
        self.figure.tight_layout()

    def display_plot(self):
        plt.show()

if __name__ == "__main__":
    p = ImageProcessor()
    d = ImageDisplayer()
    scale = 0.25

    image = load_image("images/flower.jpeg")
    subjects = p.find_subjects(image)

    encoded_resized_image = p.encode(p.resize_image(image, scale))
    decoded_resized_image_tensor = p.decode(encoded_resized_image)
    decoded_resized_image_pil = tensor_to_pil(decoded_resized_image_tensor)
    resized_decoded_image_pil = p.resize_image(decoded_resized_image_pil, 1 / scale)

    merged_image = resized_decoded_image_pil.copy()

    d.create_plot(1, 1)

    for idx, (subject_image, (x1, y1)) in enumerate(subjects):
        encoded_subject = p.encode(subject_image)
        decoded_subject_tensor = p.decode(encoded_subject)
        decoded_subject_img = p.to_displayable(decoded_subject_tensor)
        merged_image.paste(subject_image, (int(x1), int(y1)))
        print(f"subject chunk processed: {idx + 1}")

    merged_image_np = np.array(merged_image) / 255.0
    d.add_to_plot(merged_image_np, 0, 0, title="Merged Image")

    d.make_tight_layout()
    d.display_plot()



from transformers import AutoImageProcessor, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

class DinoFeatureExtractor:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")

    def batch_extract(self,images) -> np.ndarray:
        """Extracts features from images using DINOv2-small"""
        vectors = []

        for image in tqdm(images,desc='Extracting features with DINOv2-small'):
            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                vectors.append(outputs.last_hidden_state[:, 0, :].cpu())

            del inputs
            del outputs
            del image
        vectors = torch.cat(vectors, dim=0).numpy()
        return vectors

    def batch_extract_paths(self, image_paths) -> np.ndarray:
        """Extract features from images one-by-one without keeping all images in RAM."""
        
        vectors = []

        for image_path in tqdm(image_paths, desc="Extracting features with DINOv2-small"):
            # ładowanie pojedynczego obrazu
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)

                # zapisujemy tylko wynik cech
                vectors.append(outputs.last_hidden_state[:, 0, :].cpu())

            # usuwamy referencje do obrazu i tensorów
            del image
            del inputs
            del outputs

        # łączymy dopiero na końcu
        vectors = torch.cat(vectors, dim=0).numpy()

        return vectors
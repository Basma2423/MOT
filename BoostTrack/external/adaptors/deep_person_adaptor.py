import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import torchreid

class DeepPersonReID:
    def __init__(self, model_name, weights_path=None, device="cuda", batch_size=8):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1, 
            pretrained=(weights_path is None)
        )

        if weights_path:
            # Load custom weights
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Define input image transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard Deep-Person-ReID input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image, detections):
        """Extracts person crops from the image and applies preprocessing."""
        H, W, _ = image.shape
        patches = []

        for d in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections[d, :4])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
            patch = image[y1:y2, x1:x2, :]

            patch = self.transform(patch).to(self.device)
            patches.append(patch)

        return torch.stack(patches) if patches else None

    def inference(self, image, detections):
        """Extracts deep features from person crops."""
        if detections is None or len(detections) == 0:
            return []

        patches = self.preprocess(image, detections)
        if patches is None:
            return []

        features = []
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch = patches[i:i + self.batch_size]
                batch_features = self.model(batch)
                batch_features = F.normalize(batch_features, p=2, dim=1)
                features.append(batch_features.cpu().numpy())

        return np.vstack(features)
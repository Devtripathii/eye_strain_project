import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MRLEyeDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        
        for subject in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject)
            
            if os.path.isdir(subject_path):
                for file in os.listdir(subject_path):
                    if file.endswith(".png"):
                        label = int(file.split("_")[2])  # 0 = closed, 1 = open
                        self.image_paths.append(os.path.join(subject_path, file))
                        self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
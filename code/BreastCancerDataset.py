import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


# Custom Dataset class to handle nested directory structure
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, magnifications=None, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_names = [
            "adenosis",
            "fibroadenoma",
            "phyllodes_tumor",
            "tubular_adenoma",
            "ductal_carcinoma",
            "lobular_carcinoma",
            "mucinous_carcinoma",
            "papillary_carcinoma",
        ]
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        if magnifications is None:
            magnifications = os.listdir(root_dir)
        for magnification in magnifications:
            magnification_dir = os.path.join(root_dir, magnification)
            if os.path.isdir(magnification_dir):
                for diagnosis in ["benign", "malignant"]:
                    diagnosis_dir = os.path.join(magnification_dir, diagnosis)
                    if os.path.isdir(diagnosis_dir):
                        for subtype in os.listdir(diagnosis_dir):
                            subtype_dir = os.path.join(diagnosis_dir, subtype)
                            if os.path.isdir(subtype_dir):
                                label = self.class_to_idx[subtype]
                                for img_name in os.listdir(subtype_dir):
                                    img_path = os.path.join(subtype_dir, img_name)
                                    self.image_paths.append(img_path)
                                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = default_loader(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

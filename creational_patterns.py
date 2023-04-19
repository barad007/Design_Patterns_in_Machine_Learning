from torch.utils.data import Dataset
from PIL import Image

class CustomDatasetsCreator(Dataset):
    def __int__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def load_image(self, image_path):
        return Image.open(image_path)

    def get_data(self, idx):
        path, class_idx = self.data[idx]
        img = self.load_image(path)
        return img, class_idx

    def transformation(self, sample):
        img, class_idx = sample
        return self.transform(img), class_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.get_data(idx)

        if self.transform:
            sample = self.transformation(sample)

        return sample

from abc import abstractmethod
import json
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):
    """
    Interfejs produktu deklarujący wszystkie działania, które konkretne produkty muszą zaimplementować.
    """
    @abstractmethod
    def __init__(self, data, transform):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class FolderDataset(Dataset):
    """Konkretne produkty posiadają różne implementacje interfejsu produktu.
    FolderDataset - dataset zawierajacy wszystkie zdjecia.

    :param data: Folder path.
    :type data: string
    :param transform: Optional transform to be applied on a sample.
    :type transform: callable, optional
    """
    def __init__(self, data, transform=None):
        self.path = data
        self.data = self.get_data()

        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def get_data(self):
        cats = glob.glob(self.path + "cats/*")
        dogs = glob.glob(self.path + "dogs/*")
        data = []
        for item in cats:
            data.append({"path": item, "cls": 0})
        for item in dogs:
            data.append({"path": item, "cls": 1})
        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]['path']
        image = Image.open(img_path)
        label = self.data[index]['cls']
        sample = {'image': image, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


class CustomImageDataset(ImageDataset):
    def __init__(self, data, transform=None):
        """
        CustomImageDataset - dataset zawierajacy zdjęcia określone przez plik json.

        :param data: Path to the json file with annotations.
        :type data: string
        :param transform: Optional transform to be applied on a sample.
        :type transform: callable, optional
        """
        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

        with open(data, 'r') as f:
          self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data[index]['cls']
        img_path = self.data[index]['path']
        image = Image.open(img_path)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


class InferenceDataset(ImageDataset):
    def __init__(self, data, transform=None):
        """
        InferenceDataset - dataset zawierajacy zdjęcia określone przez plik json.

        :param data: list of dictionaries - samples.
        :type data: List
        :param transform: Optional transform to be applied on a sample.
        :type transform: callable, optional
        """
        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

        self.data = data["samples"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx = self.data[index]["id"]
        img_path = self.data[index]['path']
        image = Image.open(img_path)
        sample = {'image': image, 'idx': idx, "img_path": img_path}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


class DatasetCreator:
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    @abstractmethod
    def create_dataset(self) -> ImageDataset:
        pass


class FolderDatasetCreator(DatasetCreator):
    def __init__(self, data, transform=None):
        """Konkretni kreatorzy nadpisują metodę wytwórczą w celu zmiany zwracanego typu produktu.
        FolderDataset - dataset zawierajacy wszystkie zdjecia.
        :param data: Folder path.
        :type data: string
        :param transform: Optional transform to be applied on a sample.
        :type transform: callable, optional
        """
        super().__init__(data, transform)
        self.data = data
        self.transform = transform

    def create_dataset(self) -> ImageDataset:
        folder_dataset = FolderDataset(data=self.data, transform=self.transform)
        return folder_dataset


class CustomImageDatasetCreator(DatasetCreator):
    def __init__(self, data, transform=None):
        """Konkretni kreatorzy nadpisują metodę wytwórczą w celu zmiany zwracanego typu produktu.
        CustomImageDatasetCreator - dataset zawierajacy zdjęcia określone przez plik json.
        :param data: Path to the json file with annotations.
        :type data: string
        :param transform: Optional transform to be applied on a sample.
        :type transform: callable, optional
        """
        super().__init__(data, transform)
        self.data = data
        self.transform = transform

    def create_dataset(self) -> ImageDataset:
        custom_image_dataset = CustomImageDataset(data=self.data, transform=self.transform)
        return custom_image_dataset


class InferenceDatasetCreator(DatasetCreator):
    def __init__(self, data, transform=None):
        """Konkretni kreatorzy nadpisują metodę wytwórczą w celu zmiany zwracanego typu produktu.
        InferenceDataset - dataset zawierajacy zdjęcia określone przez plik json.
        :param data: Path to the json file with annotations.
        :type data: string
        :param transform: Optional transform to be applied on a sample.
        :type transform: callable, optional
        """
        super().__init__(data, transform)
        self.data = data
        self.transform = transform

    def create_dataset(self) -> ImageDataset:
        inference_dataset = InferenceDataset(data=self.data, transform=self.transform)
        return inference_dataset


def get_dataset(creator: DatasetCreator) -> ImageDataset:
    """
    Kod klienta działający z instancją konkretnego kreatora poprzez jego interfejs bazową.
    """
    return creator.create_dataset()

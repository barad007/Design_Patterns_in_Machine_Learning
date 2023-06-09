import os
import json
from typing import Dict, List
import pathlib

class Target:
    """
    The Target defines interface.
    """
    def get_json_samples(self) -> Dict[str, List[Dict[str, str]]]:
        json_samples = {
            "samples": [
                {
                    "id": 0,
                    "path": "test/example.jpg"
                },
            ]
        }
        return json_samples

# Adaptee
class ImageFileReader:
    """
    ImageFileReader returns a list of files.
    """
    def __init__(self, directory):
        self.directory = directory

    def get_list_file_path(self) -> List:
        list_file_path = []
        for filename in os.listdir(self.directory):
            if pathlib.Path(filename).suffix in [".jpg", ".jpeg"]:
                path = os.path.join(self.directory, filename)
                list_file_path.append(path)
        return list_file_path


class Adapter(Target):
    """
    The adapter makes the ImageFileReader interface compatible with the Target interface.
    """

    def __init__(self, image_file_reader: ImageFileReader) -> None:
        self.image_file_reader = image_file_reader

    def get_json_samples(self) -> Dict[str, List[Dict[str, str]]]:
        list_file_path = self.image_file_reader.get_list_file_path()
        samples = []
        for  idx, path in enumerate(list_file_path):
            samples.append(
                {
                    "id": idx,
                    "path": path
                }
            )
        json_samples = {
            "samples": samples
        }
        return json_samples


def use_adapter(target: Target) -> Dict[str, List[Dict[str, str]]]:
    json_samples = target.get_json_samples()
    return json_samples



adapter = Adapter(ImageFileReader("data/test/"))
jsonx = use_adapter(adapter)


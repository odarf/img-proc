import numpy as np
from PIL import Image, ImageDraw


class MyImage:
    def __init__(self, path, name, data, data_type):
        self.path = path
        self.new_path = path
        self.name = name
        self.new_name = name
        self.file_number = 0
        self.data_type = data_type
        self.original_image = np.array(data, copy=True)
        self.new_image = np.array(data, copy=True)

    @classmethod
    def load(cls, path, name, data_type):
        return cls(path, name, np.array(Image.open(path+name).convert('L')), data_type)

    def save(self, data_type=np.uint8):
        Image.fromarray(self.new_image.astype(data_type)).save(self.new_path + self.new_name)

    def update_image(self, modified_image, postfix):
        self.new_image = np.array(modified_image, copy=True)
        self.file_number += 1
        self.new_name = self.name + '-' + str(self.file_number) + postfix + '.jpg'

    def reset_image(self):
        self.new_image = np.array(self.original_image, copy=True)
        self.new_name = self.name

    def copy_image(self):
        return MyImage(self.path, self.name, np.array(self.new_image, copy=True), self.data_type)

    def show(self):
        Image.fromarray(self.original_image).show()

    def colors(self):
        return int(np.iinfo(self.data_type).max + 1)

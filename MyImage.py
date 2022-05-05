import numpy as np
from PIL import Image, ImageDraw


class MyImage:
    def __init__(self, dir, fname, data, data_type, ext='.jpg'):
        self.dir = dir
        self.new_dir = dir
        self.fname = fname
        self.new_fname = fname
        self.ext = ext
        self.file_number = 0
        self.data_type = data_type
        self.original_image = np.array(data, copy=True)
        self.new_image = np.array(data, copy=True)
        self.width = self.new_image.shape[1]
        self.height = self.new_image.shape[0]

    @classmethod
    def load_image(cls, dir, fname, data_type, ext='.jpg'):
        return cls(dir, fname, np.array(Image.open(dir + fname + ext).convert('L')), data_type, ext)

    @classmethod
    def load_binary(cls, dir, fname, data_type, width, height, offset, ext='.jpg'):
        with open(dir + fname + '.xcr', 'rb') as file:
            output_data = np.fromfile(file, dtype=data_type)[offset:].reshape(height, width)
        return cls(dir, fname, output_data, data_type)

    def save_image(self, data_type=np.uint8):
        Image.fromarray(self.new_image.astype(data_type)).save(self.new_dir + self.new_fname + self.ext)
        print('File ' + self.new_dir + self.new_fname + self.ext + ' saved')

    def update_image(self, modified, postfix):
        self.new_image = np.array(modified, copy=True)
        self.file_number += 1
        self.new_fname = self.fname + '-' + str(self.file_number) + postfix

    def reset_image(self):
        self.new_image = np.array(self.original_image, copy=True)
        self.new_fname = self.fname

    def copy_image(self):
        return MyImage(self.dir, self.fname, np.array(self.new_image, copy=True), self.data_type, self.ext)

    def show_image(self):
        Image.fromarray(self.original_image).show()

    def rotate90_ccw(self):
        self.update_image(np.rot90(self.new_image), '-rot-ccw')

    def colors(self) -> int:
        return np.iinfo(self.data_type).max + 1

    def treshold(self, treshold):
        """
        Всё, что меньше значения, становится равным 0 \n
        :param treshold: порог
        """
        for i in range(self.height):
            for j in range(self.width):
                if self.new_image[i, j] < treshold:
                    self.new_image[i, j] = 0
                else:
                    self.new_image[i, j] = 255

    def treshold_inrange(self, left, right):
        """
        Всё, что не входит в left-right, становится равным 0 \n
        :param left: левая граница
        :param right: правая граница
        """
        for i in range(self.height):
            for j in range(self.width):
                if left < self.new_image[i, j] < right:
                    self.new_image[i, j] = 255
                else:
                    self.new_image[i, j] = 0

    def add_mask(self, mask):
        a = int((mask[0].shape[0] - 1) / 2)
        b = int((mask[0].shape[1] - 1) / 2)
        img = np.empty((self.height, self.width))
        for row in range(self.height):
            for col in range(self.width):
                sum1 = 0
                for i in range(len(mask)):
                    sum2 = 0
                    for j in range(-1 * a, a + 1):
                        sum3 = 0
                        for k in range(-1 * b, b + 1):
                            try:
                                sum3 += mask[i][j + 1, k + 1] * self.new_image[row + j, col + k]
                            except:
                                pass
                        sum2 += sum3
                    sum1 += np.abs(sum2)
                img[row, col] = sum1
        self.new_image = img

    def negative(self):
        self.new_image = 255 - self.new_image

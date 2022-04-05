import matplotlib.pyplot
import numba
import numpy as np
import math
from PIL import Image

from MyImage import *


class MyFourier:

    def __init__(self, fourier_data, real_part, complex_part, length: int, dt: int):
        self.fourier_data = fourier_data
        self.real_part = real_part
        self.complex_part = complex_part
        self.dt = dt
        self.length = length

    @classmethod
    def fourier_one_dimension(cls, input_data, dt):
        length = len(input_data)
        arg = 2.0 * math.pi / length
        real_values = []
        complex_values = []
        fourier_data = []
        for i in range(int(length)):
            real = 0.0
            complex = 0.0
            for j in range(length):
                real += input_data[j] * math.cos(arg * i * j)
                complex += input_data[j] * math.sin(arg * i * j)
            real /= length
            complex /= length
            fourier_data.append(real + complex)
            real_values.append(real)
            complex_values.append(complex)

        return cls(fourier_data, real_values, complex_values, length, dt)

    @staticmethod
    @numba.jit(nopython=True)
    def fourier_ampl(input_data):
        length = len(input_data)
        arg = 2.0 * math.pi / length
        real_values = []
        complex_values = []
        fourier_data = []
        for i in range(int(length)):
            real = 0.0
            complex = 0.0
            for j in range(length):
                real += input_data[j] * math.cos(arg * i * j)
                complex += input_data[j] * math.sin(arg * i * j)
            real /= length
            complex /= length
            fourier_data.append(real**2 + complex**2)
            real_values.append(real)
            complex_values.append(complex)
        return fourier_data

    @staticmethod
    @numba.jit(nopython=True)
    def fourier_complex(input_data):
        length = len(input_data)
        arg = 2.0 * math.pi / length
        real_values = []
        complex_values = []
        fourier_data = []
        for i in range(int(length)):
            real = 0.0
            complex = 0.0
            for j in range(length):
                real += input_data[j] * math.cos(arg * i * j)
                complex += input_data[j] * math.sin(arg * i * j)
            real /= length
            complex /= length
            fourier_data.append(real + complex)
            real_values.append(real)
            complex_values.append(complex)
        return fourier_data

    @staticmethod
    # @numba.jit(nopython=True)
    def fourier_two_dimension(self, image, dt=1):
        data = image
        m_rows = data.shape[0]
        n_cols = data.shape[1]
        for i in range(m_rows):
            stroka = []
            for j in range(n_cols):
                stroka.append(data[i][j])
            four_stroki = self.fourier_ampl(stroka)
            for j in range(len(four_stroki) - 1, int(len(four_stroki) / 2), -1):
                four_stroki.pop(j)
            rev = four_stroki[::-1]
            four_stroki = rev + four_stroki
            four_stroki.pop(n_cols)
            four_stroki.pop(0)

            # @numba.jit(nopython=True)
            # def foo():
            #     for k in range(len(four_stroki)):
            #         for j in range(n_cols):
            #             data[i][j] = four_stroki[k]
            #
            # foo()

            for j in range(n_cols):
                data[i][j] = four_stroki[j]
            print('1')

        print(str(data.shape[0]) + ' ' + str(data.shape[1]))

        for col in range(1, n_cols):
            stolbec = []
            for row in range(m_rows):
                stolbec.append(data[row][col])
            four_stolbca = self.fourier_ampl(stolbec)
            # print('1')
            for i in range(len(four_stolbca) - 1, int(len(four_stolbca) / 2), -1):
                four_stolbca.pop(i)
            # print('1')
            rev = four_stolbca[::-1]
            four_stolbca = rev + four_stolbca
            # print('1')
            for i in range(m_rows):
                data[i][col] = four_stolbca[i]

        # for i in range(1, n_cols):
        #     stolbec = np.array(data[:, i])
        #     four_stolbca = fourier(stolbec)
        #     for j in range(len(four_stolbca)-1, int(len(four_stolbca)/2), -1):
        #         four_stolbca.pop(j)
        #     rev = four_stolbca[::-1]
        #     four_stolbca.append(rev)
        #     data[:, i] = four_stolbca[0]

        print('end')
        #
        # for row in range(m_rows):
        #     for col in range(n_cols):
        #         data[row][col] += 50

        return data




    def sum_parts(self):
        sum = []
        for i in range(int(self.length/2)):
            sum.append(self.real_part[i] + self.complex_part[i])
        return sum

    def four_inverse_one_dimensional(self):
        data = self.fourier_data
        length = len(self.fourier_data)
        n = int(length/2)
        arg = 2.0 * math.pi / length
        output_data = []
        im = [0] * length
        re = [0] * length
        for i in range(length):
            for j in range(length):
                re[i] += data[j] * math.cos(arg * i * j)
                im[i] += data[j] * math.sin(arg * i * j)
            output_data.append(re[i] + im[i])

        return output_data

    def amplitude(self):
        output_data = []
        for i in range(int(len(self.real_part)/2)):
            output_data.append(math.sqrt(self.real_part[i]**2 + self.complex_part[i]**2))
        return output_data

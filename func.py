import math
import random
import struct

import matplotlib.pyplot
from PIL import Image, ImageDraw
import numpy as np

from MyFourier import *
from MyImage import *
import filters

import numba

# Устарело
def read_image_gray(file):
    """
    :param file: путь до файла-картинки
    """
    print('Started reading image')
    image = Image.open(file).convert('L')
    print('Done reading image')
    return image


def read_xcr(file, offset, width, height):
    """
    :param file: путь до файла
    :param offset: величина смещения/сдвига
    :param width: ширина
    :param height: высота
    """
    print('Started reading xcr')
    with open(file, "rb") as input_file:
        data = np.fromfile(input_file, dtype='>H')[offset:].reshape(height, width)
        print('Done reading xcr')
        return data


# Устарело
def save(data, saveas: str):
    """
    Функция сохранения\n
    :param data: входные данные
    :param saveas: путь до сохраняемого файла
    """
    Image.fromarray(np.array(data, copy=True).astype(np.uint8)).save_image(saveas)
    print('File ' + saveas + ' saved')


def scale_to_255(value):
    int(value)
    while value > 255:
        value -= 255
    while value < 0:
        value += 255
    return value


def grayscale(arr):
    print('Started making grayscale')
    x_min = np.amin(arr)
    x_max = np.amax(arr)

    grayscaled_data = []
    dx = x_max - x_min
    for row in arr:
            # uint8.max = 255
            calculated_x = [(int((x - x_min) * 255 / dx)) for x in row]
            grayscaled_data.append(calculated_x)
    print('Done making grayscale')
    output = np.array(grayscaled_data)
    output = normalization(output, 2)
    return output


"""
def grayscale(arr):
    print('Started making grayscale')
    gs_data = []
    if len(np.array(arr).shape) == 2:
        x_max = np.max(arr)
        x_min = np.min(arr)
        dx = x_max - x_min
        for i in arr:
            calculated_x = int((i[0] - x_min) * 255) / dx
            i = (calculated_x, calculated_x, calculated_x)
            gs_data.append(i)
        return gs_data
    if len(np.array(arr).shape) == 3:
        x_max = np.amax(arr)[0]
        x_min = np.amin(arr)[0]
        dx = x_max - x_min
        for i in arr:
            temp = []
            for j in i:
                calculated_x = int((i[0] - x_min) * 255) / dx
                i = [calculated_x, calculated_x, calculated_x]
                temp.append(i)
            gs_data.append(temp)
        return gs_data
"""


def normalization(arr: list, dim: int, N=255):
    """
    :param arr: список значений оттенков серости
    :param dim: размерность списка значений
    :param N: количество элементов
    :return: нормализованный набор
    """
    if dim == 1:
        normed = []
        x_min = min(arr)
        x_max = max(arr)
        for x in arr:
            normed.append(int(((x - x_min) / (x_max - x_min)) * N))
        return normed

    normed_arr = []
    width, height = len(arr[0]), len(arr)
    for row in range(height):
        for col in range(width):
            normed_arr.append(arr[row][col])
    normed_arr = normalization(normed_arr, dim=1)

    return np.array(normed_arr).reshape(height, width)


def new_image(pixel_matrix, width, height):
    image_new = Image.new('L', (width, height))
    draw = ImageDraw.Draw(image_new)
    new_image_normalized = normalization(pixel_matrix, dim=2)

    i = 0
    for y in range(height):
        for x in range(width):
            value = new_image_normalized[y][x]
            draw.point((x, y), int(value))
            i += 1

    return image_new


def resize(image, coefficient, resize_type, mode, width, height):
    """
    :param image: вход
    :param coefficient: коэффициент увеличения/уменьшения изображения
    :param resize_type: nearest - ближайший сосед, bilinear - билинейная интерполяция
    :param mode: increase - увеличение, decrease - уменьшение
    :return:
    """
    print('Started resizing')
    input_to_array = np.array(image)
    array_to_image = Image.fromarray(input_to_array.astype(np.uint8))
    pixel_matrix = array_to_image.load_image()
    asd = array_to_image.resize((int(width / coefficient), int(height / coefficient)), Image.NEAREST)

    w, h = width, height
    if mode == 'increase':
        new_width, new_hight = int(w * coefficient), int(h * coefficient)
    elif mode == 'decrease':
        new_width, new_hight = int(w / coefficient), int(h / coefficient)
    else:
        pass

    if resize_type == 'nearest':
        image_nearest_resized = Image.new('L', (new_width, new_hight))
        draw = ImageDraw.Draw(image_nearest_resized)
        for col in range(new_width):
            for row in range(new_hight):
                if mode == 'increase':
                    p = pixel_matrix[int(col / coefficient), int(row / coefficient)]
                elif mode == 'decrease':
                    p = pixel_matrix[int(col * coefficient), int(row * coefficient)]
                else:
                    pass
                draw.point((col, row), p)
        output_image = image_nearest_resized
    elif resize_type == 'bilinear':
        image_bilinear_rows = Image.new('L', (new_width, new_hight))
        draw = ImageDraw.Draw(image_bilinear_rows)
        for col in range(1, (new_width - 1)):
            for row in range(1, (new_hight - 1)):
                if mode == 'increase':
                    r1 = pixel_matrix[int(col / coefficient), int((row - 1) / coefficient)]
                    r2 = pixel_matrix[int(col / coefficient), int((row + 1) / coefficient)]
                elif mode == 'decrease':
                    r1 = pixel_matrix[int(col * coefficient), int((row - 1) * coefficient)]
                    r2 = pixel_matrix[int(col * coefficient), int((row + 1) * coefficient)]
                else:
                    pass
                p = int((r1 + r2) / 2)
                draw.point((col, row), p)
            if mode == 'increase':
                draw.point((col, 0), pixel_matrix[int(col / coefficient), int(0 / coefficient)])
                draw.point((col, new_hight), pixel_matrix[int(col / coefficient), int((new_hight - 1) / coefficient)])
            elif mode == 'decrease':
                draw.point((col, 0), pixel_matrix[int(col * coefficient), int(0 * coefficient)])
                draw.point((col, new_hight), pixel_matrix[int(col * coefficient), int((new_hight - 1) * coefficient)])
            else:
                pass
        pix_bilinear_rows = image_bilinear_rows.load_image()
        image_bilinear_resized = Image.new('L', (new_width, new_hight))
        draw_2 = ImageDraw.Draw(image_bilinear_resized)
        for row in range(1, (new_hight - 1)):
            for col in range(1, (new_width - 1)):
                r1 = pix_bilinear_rows[int(col - 1), int(row)]
                r2 = pix_bilinear_rows[int(col + 1), int(row)]
                p = int((r1 + r2) / 2)
                draw_2.point((col, row), p)
            draw_2.point((0, row), pix_bilinear_rows[int(0), int(row)])
            draw_2.point((new_width, row), pix_bilinear_rows[int(new_width - 1), int(row)])

        output_image = image_bilinear_resized
    else:
        pass
    print('Resizing complete')
    return output_image


def negative(data: MyImage):
    max_v = np.max(data.new_image)
    result_array = []

    for row in data.new_image:
        y_array = [max_v - 1 - y for y in row]
        result_array.append(y_array)

    data.update_image(np.array(result_array), '-negative')


def gamma_correction(arr, constant, gamma):
    print('Started gamma corr.')
    data = []

    for row in arr:
        buff = [constant * (np.power(x, gamma)) for x in row]
        data.append(buff)
    print('Gamma corr. done')
    return data


def logarithmic_correction(arr, constant):
    print('Started logarithmic corr.')
    width, height = len(arr[0]), len(arr)
    # data = grayscale(arr)
    data = arr

    for row in range(height):
        for col in range(width):
            data[row][col] = constant * np.log(data[row][col] + 1)

    # for row in arr:
    #    buff = [constant * (np.log((x + 1), 10)) for x in row]
    #    data.append(buff)
    print('Logarithmic corr. done')
    # return np.array(data)
    return data


def power_grad(arr, constant, power):
    width, height = len(arr[0]), len(arr)
    data = arr
    for row in range(height):
        for col in range(width):
            data[row][col] = constant * np.power(data[row][col], power)

    return data


def histogram_img(image: np.ndarray, colors: int):
    hist = [0] * colors
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            i = image[x, y]
            hist[i] += 1

    return np.array(hist), '-histogram'


def cdf_calc(histogram: np.ndarray):
    value = [0] * len(histogram)
    for i in range(len(histogram)):
        for j in range(i+1):
            value[i] += histogram[j]

    return np.array(value), '-cdf'


def eq(image: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    output_data = image
    cdf_min = cdf[cdf != 0].min()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output_data[x, y] = round((cdf[output_data[x, y]] - cdf_min) * 255 /
                                      (image.shape[0] * image.shape[1] - 1)
            )

    return output_data


def equalize_img(image: MyImage):
    hist = histogram_img(image.new_image, image.colors())[0]

    cdf = cdf_calc(hist)[0]

    eq_img = eq(image.new_image, cdf)
    image.update_image(eq_img, '-cdf-normed')
    #return output_data


def diff(input_data, dx=1):
    return np.diff(input_data) / dx


def convolution(x, h):
    n = len(x)
    m = len(h)
    output_data = [0]*(n+m)
    temp_n = n + m - 1
    for i in range(temp_n):
        # (if_test_is_false, if_test_is_true)[test]
        jmn = (0, i - (m - 1))[i >= m - 1]
        jmx = (n - 1, i)[i < n - 1]
        for j in range(jmn, jmx+1):
            output_data[i] += x[j] * h[i-j]

    return output_data


@numba.jit(nopython=True)
def auto_correlation(input_data):
    output_data = []
    mean = np.mean(input_data)
    disp = np.var(input_data)

    def autocorr(shift):
        value = 0
        for j in range(len(input_data) - shift):
            temp = (input_data[j] - mean) * (input_data[j + shift] - mean)
            value += temp
        return round(value / (disp * len(input_data)), 3)

    for i in range(len(input_data)):
        output_data.append(autocorr(i))

    return np.array(output_data)


@numba.jit(nopython=True)
def mutual_correlation(data1, data2) -> np.ndarray:
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    N = len(data1)
    output_data = []

    def autocorr(shift):
        auto_corr_value = 0
        for j in range(N - shift):
            buff = (data1[j] - mean1) * (data2[j + shift] - mean2)
            auto_corr_value += buff
        return round(auto_corr_value / N, 3)

    for i in range(N):
        output_data.append(autocorr(i))

    return np.array(output_data)


@numba.jit(nopython=True)
def four_spectrum(data, N):
    re = [0 for i in range(N)]
    im = [0 for i in range(N)]

    for i in range(N):
        for j in range(N):
            re[i] += (data[j] * math.cos((2 * math.pi * i * j) / N))
            im[i] += (data[j] * math.sin((2 * math.pi * i * j) / N))
        re[i] *= (1 / N)
        im[i] *= (1 / N)

    values = []
    for i in range(N):
        values.append(math.sqrt((re[i]) ** 2 + (im[i]) ** 2))
    return values


def one_d_four(data, dt) -> MyFourier:
    return MyFourier(np.fft.fft(data), dt)


def four_amplitude(data: np.ndarray, dt) -> MyFourier:
    return one_d_four(data, dt).amplitude()


def moire_detector(image: MyImage, m, q=0.7, dt=1):
    buff = []
    for i in range(0, len(image.new_image), m):
        row = image.new_image[i]
        buff.append(np.diff(row)/dt)

    auto_corr_frequencies = []
    for i in buff:
        auto_corr_values = auto_correlation(i)
        fourier_ac = four_amplitude(auto_corr_values, dt)
        fourier_ac_max_value_index = np.argmax(fourier_ac[1])
        max_four_ac_freqs = fourier_ac[0][fourier_ac_max_value_index] // 0.01 * 0.01

        auto_corr_frequencies.append(max_four_ac_freqs)

    mutual_corr_frequencies = []
    for i in range(0, len(buff)):
        row1 = buff[i]
        row2 = buff[(i + 1) % len(buff)]
        mutual_corr_values = mutual_correlation(row1, row2)
        fourier_mc = four_amplitude(mutual_corr_values, dt)
        fourier_mc_max_value_index = np.argmax(fourier_mc[1])
        max_freq_mc = fourier_mc[0][fourier_mc_max_value_index] // 0.01 * 0.01

        mutual_corr_frequencies.append(max_freq_mc)

    combined_frequencies = auto_corr_frequencies + mutual_corr_frequencies
    freqs_count = {}

    for freq in combined_frequencies:
        if freq in freqs_count:
            freqs_count[freq] += 1
        else:
            freqs_count[freq] = 1

    output_data = []

    for key in freqs_count:
        valuuue = freqs_count[key]  # debug variable
        if (valuuue / len(combined_frequencies)) >= q:
            output_data.append(key)

    return output_data


def moire_fixer(image: MyImage, freqs, apply_vertical_fix, shift, m) -> bool:
    if len(freqs) == 0:
        print('Can\'t find any artifacts on image, size of freqs = 0')
        return False

    first_top_freq = freqs[0]
    filter = filters.bsw_filter(first_top_freq - shift, first_top_freq + shift, 1, m)[1]

    print('Working frequency ' + str(first_top_freq))

    result = []
    for row in image.new_image:
        conv = convolution(np.array(row), np.array(filter), 1)[1]
        result.append(conv)

    if apply_vertical_fix:
        rotated = np.rot90(result)
        result = []
        for row in rotated:
            conv = convolution(np.array(row), np.array(filter), 1)[1]
            result.append(conv)
        image.update_image(np.rot90(result, k=3), '-moireFixed')
    else:
        image.update_image(result, '-moireFixed')

    print('Image ' + image.dir + image.fname + image.ext + ' is fixed')

    return True


def salt_and_pepper(image: MyImage, amount_of_dots):
    output_data = []
    for i in image.new_image:
        output_data.append(salt_and_pepper_for_line(i, amount_of_dots))

    image.update_image(output_data, '-saltedPeppered')


def salt_and_pepper_for_line(input_data, amount_of_dots) -> np.ndarray:
    output_data = []
    for i in input_data:
        p = random.randint(0, 1000)
        if p < amount_of_dots:
            output_data.append(0)
        elif 300 < p < (300 + amount_of_dots):
            output_data.append(255)
        else:
            output_data.append(i)

    return np.array(output_data)


def random_noise(image: MyImage, int):
    output_data = []
    for row in image.new_image:
        output_data.append(random_noise_for_line(row, int))

    image.update_image(output_data, '-randNoised')


def random_noise_for_line(input_data, int) -> np.ndarray:
    output_data = []
    for value in input_data:
        noise = random.randint(0, int)
        plus_or_minus = random.randint(1, 3)

        if plus_or_minus == 2:
            output_data.append((value + noise) % 255)
        else:
            output_data.append((value - noise) % 255)

    return np.array(output_data)


def linear_filter(image_in: MyImage, kernel):
    image = image_in.new_image
    mask = np.ones([kernel, kernel], dtype=int)
    output = np.zeros_like(image)

    image_padded = np.zeros((image.shape[0] + kernel - 1, image.shape[1] + kernel - 1))
    image_padded[1:-1, 1:-1] = image

    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            output[j, i] = np.mean((mask * image_padded[j: j+kernel, i: i+kernel]))

    image_in.update_image(np.array(output), '-linearFiltered')


def median_filter(image_in: MyImage, kernel):
    image = image_in.new_image
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + kernel - 1, image.shape[1] + kernel - 1))
    image_padded[1:-1, 1:-1] = image

    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            output[j, i] = np.median(image_padded[j: j+kernel, i: i+kernel])

    image_in.update_image(np.array(output), '-medianFiltered')


def cardiogram(dt):
    n = 1000
    m = 200
    x = [0]*n
    h = [0]*m
    y = [0]*(n+m)
    alpha = 10
    frequency = 4
    temp_n = n + m - 1

    for i in range(len(h)):
        h[i] = math.sin(2 * math.pi * frequency * (i*dt)) * math.exp(-alpha * (i * dt))

    x[200] = 120

    y = convolution(x, h)

    return y


# def fourier(input_data):
#     length = len(input_data)
#     real = 0.0
#     imagine = 0.0
#     for i in range(int(length/2)):
#         real = 0.0
#         imagine = 0.0
#         for j in range(length):
#             real += input_data[j] * math.cos((2.0 * math.pi * i * j) / length)
#             imagine += input_data[j] * math.sin((2.0 * math.pi * i * j) / length)
#         real /= length
#         imagine /= length
#
#     return real, imagine


# def fourier_amplitude(input_data):
#     length = len(input_data)
#     output_data = []
#     for i in range(int(length/2)):
#         real = 0.0
#         imagine = 0.0
#         for j in range(length):
#             real += input_data[j] * math.cos((2.0 * math.pi * i * j) / length)
#             imagine += input_data[j] * math.sin((2.0 * math.pi * i * j) / length)
#         real /= length
#         imagine /= length
#         y = math.sqrt(real*real + imagine*imagine)
#         output_data.append(y)
#
#     return output_data


def twodfour(data):
    return np.fft.fft2(data)

"""
@numba.jit(nopython=True)
def fourier(input_data):
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


# @numba.jit(nopython=True)
def fourier_two_dimension(image, dt=1):
    data = image
    m_rows = data.shape[0]
    n_cols = data.shape[1]
    for i in range(m_rows):
        stroka = []
        for j in range(n_cols):
            stroka.append(data[i][j])
        four_stroki = fourier(stroka)
        for j in range(len(four_stroki)-1, int(len(four_stroki)/2), -1):
            four_stroki.pop(j)
        rev = four_stroki[::-1]
        four_stroki = rev + four_stroki
        four_stroki.pop(0)
        four_stroki.pop(480)

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
        four_stolbca = fourier(stolbec)
        #print('1')
        for i in range(len(four_stolbca)-1, int(len(four_stolbca)/2), -1):
            four_stolbca.pop(i)
        #print('1')
        rev = four_stolbca[::-1]
        four_stolbca = rev + four_stolbca
        #print('1')
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
    for row in range(m_rows):
        for col in range(n_cols):
            data[row][col] = 255 - data[row][col]

    return data
"""

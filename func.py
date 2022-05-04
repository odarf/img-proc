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


def read_bin(file, format, width, height) -> MyImage:
    d = {'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'l': 4, 'L': 4,
         'q': 8, 'Q': 8, 'f': 4, 'd': 8}
    with open(file, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(0, len(data), d[format]):
            elem = struct.unpack(format, data[i:i + d[format]])
            figures.append(elem[0])

        data = np.reshape(figures, (height, width))
        return MyImage()


def dat_image_binary_read(folder, name, extension, width, height, format) -> MyImage:
    d = {'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'l': 4, 'L': 4,
         'q': 8, 'Q': 8, 'f': 4, 'd': 8}
    file = folder + name + extension
    with open(file, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(0, len(data), d[format]):
            elem = struct.unpack(format, data[i:i + d[format]])
            figures.append(elem[0])

        data = np.reshape(figures, (height, width))
    # return MyImage(folder, name, extension, data, 'f')
    return MyImage(folder, name, data, np.uint8)


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
    Image.fromarray(np.array(data, copy=True).astype(np.uint8)).save(saveas)
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
    # output = normalization(output, 2)
    return output


# def grayscale(arr):
#     print('Started making grayscale')
#     gs_data = []
#     if len(np.array(arr).shape) == 2:
#         x_max = np.max(arr)
#         x_min = np.min(arr)
#         dx = x_max - x_min
#         for i in arr:
#             calculated_x = int((i[0] - x_min) * 255) / dx
#             i = (calculated_x, calculated_x, calculated_x)
#             gs_data.append(i)
#         return gs_data
#     if len(np.array(arr).shape) == 3:
#         x_max = np.amax(arr)[0]
#         x_min = np.amin(arr)[0]
#         dx = x_max - x_min
#         for i in arr:
#             temp = []
#             for j in i:
#                 calculated_x = int((i[0] - x_min) * 255) / dx
#                 i = [calculated_x, calculated_x, calculated_x]
#                 temp.append(i)
#             gs_data.append(temp)
#         return gs_data


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
    pixel_matrix = array_to_image.load()
    # pixel_matrix = image

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
        pix_bilinear_rows = image_bilinear_rows.load()
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
            i = int(image[x, y])
            hist[i] += 1

    return np.array(hist), '-histogram'


def cdf_calc(histogram: np.ndarray):
    value = [0] * len(histogram)
    for i in range(len(histogram)):
        for j in range(i + 1):
            value[i] += histogram[j]

    return np.array(value), '-cdf'


def eq(image: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    output_data = image
    cdf_min = cdf[cdf != 0].min()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output_data[x, y] = round((cdf[output_data[x, y]] - cdf_min) * 255.0 /
                                      (image.shape[0] * image.shape[1] - 1)
                                      )

    return output_data


def equalize_img(image: MyImage):
    hist = histogram_img(image.new_image, image.colors())[0]

    cdf = cdf_calc(hist)[0]

    eq_img = eq(image.new_image, cdf)
    image.update_image(eq_img, '-cdf-equalized')


def diff(input_data, dx=1):
    return np.diff(input_data) / dx


# @numba.jit(nopython=True)
def convolution(x, h):
    n = len(x)
    m = len(h)
    output_data = [0] * (n + m)
    temp_n = n + m - 1
    for i in range(temp_n):
        # (if_test_is_false, if_test_is_true)[test]
        jmn = (0, i - (m - 1))[i >= m - 1]
        jmx = (n - 1, i)[i < n - 1]
        for j in range(jmn, jmx + 1):
            output_data[i] += x[j] * h[i - j]

    return output_data


# @numba.jit(nopython=True)
def convolution_image(image, control_signal, m):
    height = image.shape[0]
    width = image.shape[1]
    output = []
    for i in range(height):
        temp = convolution(image[i], control_signal)
        output.append(temp[m:(width + m)])
    return output


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
        buff.append(np.diff(row) / dt)

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
            output[j, i] = np.mean((mask * image_padded[j: j + kernel, i: i + kernel]))

    image_in.update_image(np.array(output), '-linearFiltered')


def median_filter(image_in: MyImage, kernel):
    image = image_in.new_image
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + kernel - 1, image.shape[1] + kernel - 1))
    image_padded[1:-1, 1:-1] = image

    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            output[j, i] = np.median(image_padded[j: j + kernel, i: i + kernel])

    image_in.update_image(np.array(output), '-medianFiltered')


def cardiogram(dt):
    n = 1000
    m = 200
    x = [0] * n
    h = [0] * m
    y = [0] * (n + m)
    alpha = 10
    frequency = 4
    temp_n = n + m - 1

    for i in range(len(h)):
        h[i] = math.sin(2 * math.pi * frequency * (i * dt)) * math.exp(-alpha * (i * dt))

    x[200] = 120

    y = convolution(x, h)

    return y


def two_d_four_vstroen(data, dt) -> MyFourier:
    return MyFourier(np.fft.fft2(data), 0, 0, 0, dt)


# @numba.jit(nopython=True)
def fourier_ampl(input_data):
    length = len(input_data)
    arg = 2.0 * math.pi / length
    # fourier_data = np.zeros(length, dtype=float)
    fourier_data = []
    for i in range(int(length)):
        real = 0.0
        complex = 0.0
        for j in range(length):
            real += float(input_data[j] * math.cos(arg * i * j))
            complex += float(input_data[j] * math.sin(arg * i * j))
        real /= length
        complex /= length
        y = np.sqrt(float(real * real) + float(complex * complex))
        fourier_data.append(y)
    return fourier_data


@numba.jit(nopython=True)
def fourier_complex(input_data):
    length = len(input_data)
    arg = 2.0 * math.pi / length
    fourier_data = []
    for i in range(int(length)):
        real = 0.0
        complex = 0.0
        for j in range(length):
            real += float(input_data[j] * math.cos(arg * i * j))
            complex += float(input_data[j] * math.sin(arg * i * j))
        real /= length
        complex /= length
        fourier_data.append(real + complex)
    return fourier_data


# @numba.jit(nopython=False)
def fourier_two_dimension(image: np.array, mode=1):
    data = np.zeros((len(image), len(image[0])), dtype=float)

    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i, j] = image[i, j]

    n = len(data[0])
    m = len(data)
    for i in range(m):
        print('Текущая строка -', i)
        stroka = [0] * n
        for j in range(n):
            stroka[j] = data[i, j]

        four_stroki = fourier_complex(stroka)

        for j in range(n):
            data[i, j] = four_stroki[j]

    for col in range(n):
        print('Текущий столбец -', col)
        stolbec = [0] * m
        for j in range(m):
            stolbec[j] = data[j, col]

        four_stolbca = fourier_complex(stolbec)

        for i in range(m):
            data[i, col] = four_stolbca[i]

    print('end')

    return data


def fourier_two_dimension_ampl(image: np.array, mode=1):
    data = np.zeros((len(image), len(image[0])), dtype=float)

    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i, j] = image[i, j]

    n = len(data[0])
    m = len(data)
    for i in range(m):
        print('Текущая строка -', i)
        stroka = [0] * n
        for j in range(n):
            stroka[j] = data[i, j]

        four_stroki = fourier_ampl(stroka)
        for j in range(len(four_stroki) - 1, int(len(four_stroki) / 2), -1):
            four_stroki.pop(j)
        rev = four_stroki[::-1]
        four_stroki = rev + four_stroki

        for j in range(n):
            data[i, j] = four_stroki[j]

    for col in range(n):
        print('Текущий столбец -', col)
        stolbec = [0] * m
        for j in range(m):
            stolbec[j] = data[j, col]

        four_stolbca = fourier_ampl(stolbec)
        for j in range(len(four_stolbca) - 1, int(len(four_stolbca) / 2), -1):
            four_stolbca.pop(j)
        rev = four_stolbca[::-1]
        four_stolbca = rev + four_stolbca

        for i in range(m):
            data[i, col] = four_stolbca[i]

    print('end')

    return data


def four_inv_two_dimensional(image: np.array):
    data = image
    rows = data.shape[0]
    cols = data.shape[1]

    for col in range(cols):
        print('Текущий столбец -', col)
        stolbec = []
        for row in range(rows):
            stolbec.append(data[row][col])
        four_stolbca = four_inverse_one_dimensional(stolbec)

        for i in range(len(four_stolbca)):
            data[i][col] = four_stolbca[i]

    for i in range(rows):
        print('Текущая строка -', i)
        stroka = []
        for j in range(cols):
            stroka.append(data[i, j])
        four_stroki = four_inverse_one_dimensional(stroka)

        for j in range(len(four_stroki)):
            data[i, j] = four_stroki[j]

    return data


def four_inverse_one_dimensional(input_data):
    data = input_data
    length = len(input_data)
    arg = 2.0 * math.pi / length
    output_data = np.zeros(len(input_data), dtype=float)
    im = [0] * length
    re = [0] * length
    for i in range(length):
        for j in range(length):
            re[i] += float(data[j] * math.cos(arg * i * j))
            im[i] += float(data[j] * math.sin(arg * i * j))
        output_data[i] = float(re[i] + im[i])

    return output_data


def resize_fourier(input_data: np.ndarray, coefficient):
    fourier_complex_spectrum = fourier_two_dimension(input_data)
    height_orig = input_data.shape[0]
    width_orig = input_data.shape[1]
    height = int(height_orig * coefficient)
    width = int(width_orig * coefficient)
    add_zeros_here = np.zeros((height, width), dtype=float)
    x = 0
    y = 0
    for i in range(height):
        for j in range(width):
            if (i < height / 2 or i > (height - height_orig / 2 - 1)) \
                    and (j < width_orig / 2 or j > (width - width_orig / 2 - 1)):
                add_zeros_here[i, j] = fourier_complex_spectrum[x, y]
                y += 1
        y = 0
        if i < height_orig / 2 or i > (height - height_orig / 2 - 1):
            x += 1

    output = four_inv_two_dimensional(add_zeros_here)

    return output


def substract_images(image1: np.ndarray, image2: np.ndarray):
    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1]:
        print('Can\'t compare images with different size')
        return 0
    output = np.zeros((image1.shape[0], image1.shape[1]), dtype=float)
    wdata1 = np.zeros((image1.shape[0], image1.shape[1]), dtype=float)
    wdata2 = np.zeros((image1.shape[0], image1.shape[1]), dtype=float)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            wdata1[i, j] = float(image1[i, j])
            wdata2[i, j] = float(image2[i, j])

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            output[i, j] = wdata1[i, j] - wdata2[i, j]
    return output


def gradient(image: MyImage):
    mask = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])

    img = convolution_image(image.new_image, mask, 32)
    image.update_image(img, '-gradient')


def laplasian(m):
    height = m.shape[0]
    width = m.shape[1]

    new_image = []
    for row in range(1, height - 1):
        new_row = []
        for col in range(1, width - 1):
            new_row.append(
                int(m[row + 1][col]) + int(m[row - 1][col]) + int(m[row][col + 1]) + int(m[row][col - 1]) - int(
                    4 * m[row][col])
            )
        new_image.append(new_row)
    output = np.array(new_image).reshape(height - 2, width - 2)

    return output


def erosion(image, mask_x, mask_y):
    er_mask = np.full((mask_y, mask_x), 255)
    height = image.shape[0]
    width = image.shape[1]

    new_image = []
    for row in range(height):
        new_row = []
        for col in range(width):
            if (row < (mask_y // 2)) or (row >= height - (mask_y // 2)) \
                    or (col < (mask_x // 2)) or (col >= width - (mask_x // 2)):
                new_row.append(image[row][col])
            else:
                check_sum = 0
                for mask_row in range(mask_y):
                    for mask_col in range(mask_x):
                        if image[(row - (mask_y // 2) + mask_row), (col - (mask_x // 2) + mask_col)] != \
                                er_mask[mask_row][mask_col]:
                            check_sum += 1
                if check_sum == mask_y * mask_x:
                    new_row.append(image[row][col])
                else:
                    new_row.append(0)

        new_image.append(new_row)

    new_height = len(new_image)
    new_width = len(new_image[0])

    new_image = np.array(new_image).reshape(new_height, new_width)
    return new_image


def dilatation(image, mask_x, mask_y):
    er_mask = np.full((mask_y, mask_x), 255)
    height = image.shape[0]
    width = image.shape[1]

    new_image = []
    for row in range(height):
        new_row = []
        for col in range(width):
            if (row < (mask_y // 2)) or (row >= height - (mask_y // 2)) \
                    or (col < (mask_x // 2)) or (col >= width - (mask_x // 2)):
                new_row.append(image[row][col])
            else:
                check_sum = 0
                for mask_row in range(mask_y):
                    for mask_col in range(mask_x):
                        if image[(row - (mask_y // 2) + mask_row), (col - (mask_x // 2) + mask_col)] != \
                                er_mask[mask_row][mask_col]:
                            check_sum += 1
                if check_sum == mask_y * mask_x:
                    new_row.append(0)
                else:
                    new_row.append(255)

        new_image.append(new_row)

    new_height = len(new_image)
    new_width = len(new_image[0])

    new_image = np.array(new_image).reshape(new_height, new_width)

    return new_image


def data_read(folder, name, ext, dtype, width, height) -> np.ndarray:
    with open(folder + name + ext, mode='rb') as file:
        data = np.fromfile(file, dtype=dtype).reshape(height, width)
    return np.array(data)


def re_im(data: np.array, N):
    re = [0 for i in range(N)]
    im = [0 for i in range(N)]
    for i in range(N):
        for j in range(N):
            re[i] += (data[j] * np.cos((2.0 * np.pi * i * j) / N))
            im[i] += (data[j] * np.sin((2.0 * np.pi * i * j) / N))
    return re, im



def fix_blur(image: np.array, kernel: np.ndarray, alpha: float):
    width, height = len(image[0]), len(image)
    kernel_len = len(kernel[0])  # длина ядра смазывающей функции
    control_len = len(kernel)
    kernel = kernel[0].tolist()


    # добавляю нули в конец массива ядра
    for i in range(width - kernel_len):
        kernel.append(0)
        # kernel[0].append(0)

    # kernel_fourier = fourier_ampl(kernel)
    re_h, im_h = re_im(kernel, width)

    image_spectrum = []
    image_spectrum_re, image_spectrum_im = [], []
    module = []
    output_data = []
    step = 0
    # построчное преобразование фурье
    for row in range(height):
        temp = fourier_ampl(image[row])
        step += 1

        image_spectrum.append(temp)
        re, im = re_im(image[row], width)

        # комплексное деление строчки изображения и ядра функции
        # alpha = 0.005????
        r_re, r_im = [], []
        for i in range(width):
            # print('текущий индекс ', i)
            correction_factor = (re_h[i] ** 2 + im_h[i] ** 2) + alpha  # поправочный множитель
            r_re.append((re_h[i] * re[i] + im_h[i] * im[i]) / correction_factor)
            r_im.append((re_h[i] * im[i] - im_h[i] * re[i]) / correction_factor)

        # вычисляю модуль отношения комплексных величин
        module_temp = []
        for col in range(width):
            module_temp.append(r_re[col] + r_im[col])

        module.append(module_temp)
        output_data.append(four_inverse_one_dimensional(module[row]))

    return output_data


def otsu_threshold(image: MyImage, size) -> int:
    min = image.new_image.min()
    max = image.new_image.max()
    width, height = len(image.new_image[0]), len(image.new_image)

    hist_size = round(max - min + 1)
    hist = [0 for i in range(hist_size)]

    for col in range(width):
        for row in range(height):
            index = int(image.new_image[row, col] - min)
            hist[index] += 1

    m = 0  # сумма высот всех бинов, домноженных на положение их середины
    n = 0  # сумма высот всех бинов

    for i in range(0, int(max-min+1)):
        m += i * hist[i]
        n += hist[i]

    max_sigma = -1.0
    threshold = 0
    alpha1 = 0
    beta1 = 0

    for i in range(int(max - min)):
        alpha1 += i * hist[i]
        beta1 += hist[i]

        w1 = float(float(beta1) / n)
        a = float(float(alpha1 / beta1) - float((m-alpha1) / (n - beta1)))

        sigma = float(w1 * (1 - w1) * a * a)

        if sigma > max_sigma:
            max_sigma = sigma
            threshold = i

    threshold += min
    return threshold

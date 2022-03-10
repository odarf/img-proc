import struct

from PIL import Image, ImageDraw
import numpy as np


def read_image(file):
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

"""
def grayscale(arr):
    print('Started making grayscale')
    x_min = np.amin(arr)
    x_max = np.amax(arr)

    grayscaled_data = []
    dx = x_max - x_min
    for row in arr:
            # uint8.max = 255
            calculated_x = [NormTo255(int((x - x_min) * 255 / dx)) for x in row]
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
            draw.point((x, y), int(new_image_normalized[y][x]))
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


def negative(arr):
    print('Started negativization')
    data1 = np.array(arr)
    data = grayscale(data1)
    width, height = len(data[0]), len(data)
    #data = np.array(data).reshape(height, width)

    for row in range(height):
        for col in range(width):
            data[row][col] = 255 - data[row][col]

    print('Negativization done')
    return data


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

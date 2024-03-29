import imutils
import numpy as np
import cv2 as cv
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from func import *
from matplotlib import pyplot as plot
from MyImage import *
from MyFourier import *
# import matlab


def lab_1():
    image = read_image_gray('images/grace.jpg')
    width, height = image.size
    pix_mat = np.array(image).reshape(height, width)

    for row in range(height):
        for col in range(width):
            pix_mat[row][col] += 30

    save(pix_mat, '1grace+30.jpg')

    pix_mat = np.array(image).reshape(height, width)
    for row in range(height):
        for col in range(width):
            pix_mat[row][col] *= 1.3

    save(pix_mat, '2grace_x1_3.jpg')

    pix_mat = np.array(image).reshape(height, width)
    gs = grayscale(pix_mat)
    save(gs, '3grace_gs.jpg')

    # Лабораторная работа №3
    rs = resize(image, 1.3, 'bilinear', 'increase', width, height)
    save(rs, '1grace_bilinear_resized.jpg')
    rs = resize(image, 1.3, 'nearest', 'increase', width, height)
    save(rs, '1grace_nearest_resized.jpg')


def lab_2():
    width12, height12 = 1024, 1024  # c12
    width0, height0 = 2048, 2500  # u0

    c12_xcr_input = read_xcr('images/c12-85v.xcr', 5120, width12, height12)
    c12_xcr_rotated = np.rot90(c12_xcr_input)
    c12_xcr_grayscaled = grayscale(c12_xcr_rotated)
    # save(c12_xcr_grayscaled, 'c12_xcr_grayscaled.jpg')
    # new_image(c12_xcr_grayscaled, width12, height12).show()

    u0_xcr_input = read_xcr('images/u0.xcr', 5120, width0, height0)
    u0_xcr_rotated = np.rot90(np.array(u0_xcr_input))
    u0_xcr_grayscaled = grayscale(np.array(u0_xcr_rotated))
    # save(u0_xcr_grayscaled, 'u0_xcr_grayscaled.jpg')
    # new_image(u0_xcr_grayscaled, height0, width0).show()

    # Лабораторная работа №3
    rs = resize(c12_xcr_grayscaled, 0.6, 'bilinear', 'increase', width12, height12)
    save(rs, 'c12_xcr_bilinear_x06_resized.jpg')
    rs = resize(c12_xcr_grayscaled, 0.6, 'nearest', 'increase', width12, height12)
    save(rs, 'c12_xcr_nearest_x06_resized.jpg')

    rs = resize(u0_xcr_grayscaled, 0.6, 'bilinear', 'increase', width12, height12)
    save(rs, 'u0_xcr_test_x06_resized.jpg')
    # rs = resize(u0_xcr_grayscaled, 0.6, 'nearest', 'increase', width12, height12)
    # save(rs, 'u0_xcr_nearest_x06_resized.jpg')


def lab_4():
    grace = read_image_gray('images/grace.jpg')
    neg = negative(grace)
    save(neg, 'grace_negative.jpg')

    c12_xcr = read_xcr('images/c12-85v.xcr', 5120, 1024, 1024)
    # c12_xcr = c12_xcr.byteswap()
    width, height = len(c12_xcr[0]), len(c12_xcr)
    c12_xcr = grayscale(c12_xcr)
    # for i in range(width):
    #    for j in range(height):
    #        c12_xcr[i][j] = scale_to_255(c12_xcr[i][j])
    c12_xcr = np.rot90(c12_xcr)

    neg = negative(c12_xcr)
    save(c12_xcr, 'c12_xcr_negative123.jpg')

    u0_xcr = read_xcr('images/u0.xcr', 5120, 2048, 2500)
    u0_xcr = np.rot90(u0_xcr)
    neg = negative(u0_xcr)
    save(grayscale(neg), 'u0_xcr_negative.jpg')

    photo1 = read_image_gray('images/photo1.jpg')
    g = gamma_correction(np.array(photo1), 1, 0.4)
    save(grayscale(g), 'photo1_gamma.jpg')
    log = logarithmic_correction(np.array(photo1), 32)
    save(log, 'photo1_log.jpg')

    photo2 = read_image_gray('images/photo2.jpg')
    g = gamma_correction(np.array(photo2), 1, 0.4)
    save(grayscale(g), 'photo2_gamma.jpg')
    log = logarithmic_correction(np.array(photo2), 30)
    save(log, 'photo2_log.jpg')

    photo3 = read_image_gray('images/photo3.jpg')
    g = gamma_correction(np.array(photo3), 1, 0.4)
    save(grayscale(g), 'photo3_gamma.jpg')
    log = logarithmic_correction(np.array(photo3), 40)
    save(log, 'photo3_log.jpg')

    photo4 = read_image_gray('images/photo4.jpg')
    g = gamma_correction(np.array(photo4), 1, 0.4)
    save(grayscale(g), 'photo4_gamma.jpg')
    log = logarithmic_correction(np.array(photo4), 40)
    save(log, 'photo4_log.jpg')

    hollywood = read_image_gray('images/HollywoodLC.jpg')
    g = gamma_correction(np.array(hollywood), 1, 2)
    save(grayscale(g), 'hollywood_gamma.jpg')
    log = logarithmic_correction(np.array(hollywood), 25)
    save(log, 'hollywood_log.jpg')


def lab_5(image: MyImage):
    hist = histogram_img(image.new_image, image.colors())
    plot.plot(hist[0])
    plot.title(image.fname + 'Гистограмма')
    plot.savefig(image.dir + 'plots/' + image.fname + 'OrigHist')
    plot.show()
    plot.plot(cdf_calc(hist[0])[0])
    plot.title(image.fname + 'Функция распределения')
    plot.savefig(image.dir + 'plots/' + image.fname + 'OrigCdf')
    plot.show()

    equalize_img(image)

    image.save_image()

    img = MyImage.load_image('images/', image.new_fname, np.uint8)
    h = histogram_img(img.new_image, img.colors())
    plot.plot(h[0])
    plot.title(image.fname + 'Гистограмма после эквализации')
    plot.savefig(img.dir + 'plots/' + img.fname + 'EqHist')
    plot.show()
    plot.plot(cdf_calc(h[0])[0])
    plot.title(image.fname + 'Функция распределения после эквализации')
    plot.savefig(img.dir + 'plots/' + img.fname + 'EqCdf')
    plot.show()


# image.rotate90_ccw()
# negative(image)
#
# hist = histogram_img(image.new_image, image.colors())
# plot.plot(hist[0])
# plot.show()
# plot.plot(cdf_calc(hist[0])[0])
# plot.show()
#
# equalize_img(image)
# image.save_image()
#
# img = MyImage.load_image('images/', image.new_fname, np.uint8)
# h = histogram_img(img.new_image, img.colors())
# plot.plot(h[0])
# plot.show()
# plot.plot(cdf_calc(h[0])[0])
# plot.show()


def lab_6():
    c12 = MyImage.load_image('images/lab6/', 'c12-85v', np.uint8)
    u0 = MyImage.load_image('images/lab6/', 'u0', np.uint8)
    moire_frequences = moire_detector(c12, 32)
    is_fixed = moire_fixer(c12, moire_frequences, True, 0.17, 32)

    if is_fixed:
        c12.save_image()

    moire_frequences = moire_detector(u0, 32)
    is_fixed = moire_fixer(u0, moire_frequences, True, 0.17, 32)

    if is_fixed:
        u0.save_image()


def lab_7():
    image1 = MyImage.load_image('images/lab7/salt_pepper/', 'model', np.uint8)

    # --------------------------Соль-Перец---------------------------------
    salt_and_pepper(image1, 20)
    image1.save_image()

    image_lf = image1.copy_image()

    linear_filter(image_lf, 3)
    image_lf.save_image()

    image_mf = image1.copy_image()

    median_filter(image_mf, 3)
    image_mf.save_image()
    # ---------------------------------------------------------------------

    # -------------------------Случайный шум-------------------------------
    image2 = MyImage.load_image('images/lab7/random_noise/', 'model', np.uint8)

    random_noise(image2, 20)
    image2.save_image()

    image2_lf = image2.copy_image()

    linear_filter(image2_lf, 3)
    image2_lf.save_image()

    image2_mf = image2.copy_image()

    median_filter(image2_mf, 3)
    image2_mf.save_image()

    # ---------------------------------------------------------------------

    # -----------------------------СП+СШ-----------------------------------
    image3 = MyImage.load_image('images/lab7/sp_rn/', 'model', np.uint8)

    random_noise(image3, 20)
    salt_and_pepper(image3, 15)
    image3.save_image()

    image3_lf = image3.copy_image()

    linear_filter(image3_lf, 3)
    image3_lf.save_image()

    image3_md = image3.copy_image()

    median_filter(image3_md, 3)
    image3_md.save_image()
    # ---------------------------------------------------------------------


def lab_8():
    dt = 0.005
    cardio = cardiogram(dt)
    plot.plot(cardio)
    plot.title("Cardiogram")
    plot.savefig('images/lab8/Cardiogram')
    plot.show()

    four = MyFourier.fourier_one_dimension(cardio, dt)
    plot.plot(four.amplitude())
    plot.title("Amplitude")
    plot.savefig('images/lab8/CardiogramAmplitude')
    plot.show()

    four_back = four.four_inverse_one_dimensional()
    plot.plot(four_back)
    plot.title("Inverse fourier")
    plot.savefig('images/lab8/CardiogramInverseFourier')
    plot.show()

    image = MyImage.load_image('images/', 'grace', np.uint8)
    img_height = image.new_image.shape[0]
    img_width = image.new_image.shape[1]

    img = image
    img.update_image(power_grad(fourier_two_dimension_ampl(image.new_image), 5, 5), '-FOUR_AMPLITUDE_IMAGE')
    img.save_image()

    image_fourier_inverse_transform = four_inv_two_dimensional(image_fourier_transform)
    grayscaled = grayscale(image_fourier_inverse_transform)
    image.update_image(grayscaled, '-fourierAndBack')
    image.save_image()

    resize_f = resize_fourier(image.new_image, 1.3)
    image.update_image(resize_f, '-2dFourierResized')
    image.save_image()

    resize_f = MyImage.load_image('images/', 'grace-3-2dFourierResized', np.uint8)
    resize_near = resize(image.new_image, 1.3, 'nearest', 'increase', img_width, img_height)
    save(resize_near, 'images/lab8/resizeNear.jpg')
    resize_bilin = resize(image.new_image, 1.3, 'bilinear', 'increase', img_width, img_height)
    save(resize_bilin, 'images/lab8/resizeBilin.jpg')
    orig_res = MyImage.load_image('images/lab8/', 'grace-res', np.uint8)
    img_near = MyImage.load_image('images/lab8/', 'resizeNear', np.uint8)
    img_bilin = MyImage.load_image('images/lab8/', 'resizeBilin', np.uint8)
    # gs = grayscale(res)

    diff = MyImage('images/lab8/', 'diffImg', 0, np.uint8)
    buff = substract_images(orig_res.new_image, resize_f.new_image)
    buff = gamma_correction(buff, 1, 0.4)
    buff = grayscale(buff)
    diff.update_image(buff, '-fourier')
    diff.save_image()

    buff = substract_images(orig_res.new_image, img_near.new_image)
    buff = gamma_correction(buff, 1, 0.4)
    buff = grayscale(buff)
    diff.update_image(buff, '-nearest')
    diff.save_image()

    buff = substract_images(orig_res.new_image, img_bilin.new_image)
    buff = gamma_correction(buff, 1, 0.4)
    buff = grayscale(buff)
    diff.update_image(buff, '-bilinear')
    diff.save_image()


def lab_9():
    kernD76_f4 = data_read('images/lab9/', 'kernD76_f4', '.dat', "float32", 76, 1)
    blur307x221D = dat_image_binary_read('images/lab9/', 'blur307x221D', '.dat', 307, 221, format='f')
    blur307x221D_N = dat_image_binary_read('images/lab9/', 'blur307x221D_N', '.dat', 307, 221, format='f')

    blur307x221D_fix = fix_blur(blur307x221D.new_image, kernD76_f4, 0.0001)

    blur307x221D.update_image(grayscale(blur307x221D_fix), '-blurfixed')
    blur307x221D.save_image()
    blur307x221D_N_fix = fix_blur(blur307x221D_N.new_image, kernD76_f4, 15)
    blur307x221D_N.update_image(grayscale(blur307x221D_N_fix), '-blurfixed')
    blur307x221D_N.save_image()


def lab_9_beautify():
    image = MyImage.load_image('images/lab9/', 'blur307x221D_N-1-blurfixed', np.uint8)
    width, height = 307, 221
    # linear_filter(image, 3)
    median_filter(image, 3)
    hist = histogram_img(image.new_image, image.colors())[0]
    plot.plot(hist)
    plot.show()
    pix = image.new_image
    test_porog = otsu_threshold(image, (307 * 221))
    print(test_porog)
    for row in range(height):
        for col in range(width):
            if pix[row, col] < test_porog:
                pix[row, col] = 0
            else:
                pix[row, col] = 255

    image.update_image(pix, '-otsuTreshold')
    image.save_image()


def lab_10_template():
    image = MyImage.load_image('images/lab10/', 'model-noize-linearFiltered', np.uint8)
    pixels = image.new_image
    f = 32  # 6 - низких,
    m = 64  # 16 - низких,
    dx = 1 / image.new_image.shape[1]

    # пороговая фильтрация
    porog1 = 200
    print('porog1 = ', porog1)
    image.treshold(porog1)

    # control_mass = filters.lpw_filter(f, dx, m)[1]
    control_mass = filters.hpw_filter(f, dx, m)

    conv_img = convolution_image(pixels, control_mass, m)
    image.update_image(conv_img, '-convolved')

    # pix = conv_img - pixels  # lpw
    # image.update_image(pix, '-convolved-substracted')
    ## hist = histogram_img(image.new_image, image.colors())[0]
    ## plot.plot(hist)
    ## plot.show()

    porog2 = 60

    # porog2 = otsu_threshold(image, int(image.width * image.height))
    print('porog2 = ', porog2)

    # пороговая фильтрация для чётких контуров
    image.treshold(porog2)
    image.update_image(image.new_image, '-porogContur-' + 'f' + str(f) + '-' + 'm' + str(m) + '-' + str(porog1) + '-' + str(porog2))
    image.save_image()


def lab_11_gradient(image: MyImage, direction='default'):
    th1, th2 = 190, 250

    masks = [np.array([
            [0,  0, 0],
            [0, -1, 0],
            [0,  1, 0]
        ]), np.array([
            [0,  0, 0],
            [0, -1, 1],
            [0,  0, 0]
        ])]

    masks_vh = [np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]), np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])]

    masks_d = [np.array([
        [0,   1, 2],
        [-1,  0, 1],
        [-2, -1, 0]
    ]), np.array([
        [-2, -1, 0],
        [-1,  0, 1],
        [ 0,  1, 2]
    ])]

    image.treshold_inrange(th1, th2)

    if direction == 'default':
        image.add_mask(masks)
        image.update_image(image.new_image, '-gradient')
    elif direction == 'vh':
        image.add_mask(masks_vh)
        image.update_image(image.new_image, '-gradientSobel-VH')
    elif direction == 'd':
        image.add_mask(masks_d)
        image.update_image(image.new_image, '-gradientSobel-D')

    image.save_image()
    image.reset_image()


def lab_11_laplasian(image: MyImage):
    th1, th2 = 190, 250

    mask = [np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])]

    image.treshold_inrange(th1, th2)

    image.add_mask(mask)

    image.update_image(np.abs(image.new_image), '-laplas')

    image.save_image()
    image.reset_image()


def lab_12(image: MyImage):
    matrix_size = 3  # квадратная
    image.treshold(200)

    pix = image.new_image

    pix_dil = morphological_operator(pix, matrix_size, matrix_size, 'dilatation')  # обосновать выбор размера матрицы
    pix_eros = morphological_operator(pix, matrix_size, matrix_size, 'erosion')

    dilatation_result = pix_dil - pix
    image.update_image(dilatation_result, '-dilatated-' + str(matrix_size) + 'x' + str(matrix_size))
    image.save_image()

    erosion_result = np.abs(pix_eros - pix)
    image.update_image(erosion_result, '-eroseTemp')
    image.negative()
    image.update_image(image.new_image, '-erosed-' + str(matrix_size) + 'x' + str(matrix_size))
    image.save_image()
    image.reset_image()


def lab_mrt(image: MyImage):
    '''
    ОБРАБОТКУ НЕЛЬЗЯ ДЕЛАТЬ!
    ЛАПЛАСИАН НЕЛЬЗЯ
    '''
    # image = dat_image_binary_read('images/mrt/brain-h/', 'brain-h_x512', '.bin', width, height, format='h')
    width, height = image.width, image.height

    # image.update_image(image.new_image, '-test')

    image.save_image()
    # hist = histogram_img(image, image.colors())[0]
    plot.hist(image.new_image.flatten(), 256, [0, 256])
    plot.title('Исходное ' + image.fname)
    plot.savefig(image.dir + image.fname + 'Исходное ' + image.fname)
    # plot.plot(hist)
    plot.show()

    th = 10
    # src = cv2.imread('images/mrt/brain-h/brain-H_x512.jpg', 0)
    # mask = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)

    # th = int(otsu_threshold(image, (width * height)) / 2)
    # th = otsu(image)
    print('Порог для ' + image.fname + ' = ' + str(th))

    image.treshold(th)
    median_filter(image, 3)
    image.update_image(image.new_image, '-mask-' + str(th))
    mask = image.new_image
    image.save_image()
    image.reset_image()

    for row in range(height):
        for col in range(width):
            if mask[row, col] == 255:
                image.new_image[row, col] *= 1
            elif mask[row, col] == 0:
                image.new_image[row, col] = 0

    for row in range(height):
        for col in range(width):
            if mask[row, col] == 255:  # 3, 0.8
                # image.new_image[row, col] = (10 * np.log(image.new_image[row, col] + 1)).round()
                image.new_image[row, col] = (3 * (image.new_image[row, col] ** 0.8)).round()

    image.update_image(image.new_image, '-powerGraded')
    image.save_image()

    # equalize_img(image)
    imgtest = cv2.imread(image.dir + image.fname + '-3-powerGraded.jpg', 0)
    # imgtest = image.new_image
    plot.hist(imgtest.flatten(), 256, [0, 256])
    plot.title('Настроенное ' + image.fname)
    plot.savefig(image.dir + image.fname + 'Настроенное ' + image.fname)
    plot.show()
    equ = cv2.equalizeHist(imgtest)
    # res = np.hstack((imgtest, equ))
    cv2.imwrite(image.dir + image.fname + '-final-mask' + str(th) + '-.jpg', equ)


def stones():
    print('Size = 7px')
    image = MyImage.load_image('images/stones/', 'stones', np.uint8)
    # plot.hist(image.new_image.flatten(), 256, [0, 256])
    # plot.title('Исходное ' + image.fname)
    # plot.show()
    print(otsu(image))
    print(otsu_threshold(image, (image.height * image.width)))
    image.treshold(135)
    image.update_image(image.new_image, '-thed')
    m = image.new_image
    image.save_image()

    laplas = cv2.Laplacian(image.new_image, 5)
    # image.add_mask(mask)
    # image.update_image(foo, '-laplas')
    # image.save_image()

    # pix_eros1 = morphological_operator(m, 6, 6, 'erosion')
    # pix_eros2 = morphological_operator(m, 7, 7, 'erosion')

    kernel1 = np.ones((6, 6), dtype="uint8")
    kernel2 = np.ones((7, 7), dtype="uint8")
    kernel3 = np.ones((4, 4), dtype="uint8")
    pix_eros1 = cv2.erode(m, kernel1, iterations=1)
    image.update_image(pix_eros1, '-erosed-' + str(kernel1.shape[0]))
    image.save_image()
    pix_eros2 = cv2.erode(m, kernel2, iterations=1)
    pix_eros2 = cv2.dilate(pix_eros2, kernel3, iterations=1)

    image.update_image(pix_eros2, '-erosed-' + str(kernel2.shape[0]))
    image.save_image()

    erosion_result = sub(pix_eros1, pix_eros2)
    # erosion_result = np.subtract(pix_eros1, pix_eros2)

    image.update_image(erosion_result, '-erosed')
    image.save_image()

    # for row in range(2, image.height - 2):
    #     for col in range(2, image.width - 2):
    #         if image.new_image[row, col] == 255:
    #             if image.new_image[row, col - 1] == 0 and \
    #                     image.new_image[row, col + 1] == 0 and \
    #                     image.new_image[row, col - 2] == 0 and \
    #                     image.new_image[row, col + 2] == 0 and \
    #                     image.new_image[row - 1, col] == 0 and \
    #                     image.new_image[row - 1, col - 1] == 0 and \
    #                     image.new_image[row - 1, col + 1] == 0 and \
    #                     image.new_image[row - 1, col + 2] == 0 and \
    #                     image.new_image[row - 1, col - 2] == 0 and \
    #                     image.new_image[row + 1, col] == 0 and \
    #                     image.new_image[row + 1, col - 1] == 0 and \
    #                     image.new_image[row + 1, col + 1] == 0 and \
    #                     image.new_image[row + 1, col + 2] == 0 and \
    #                     image.new_image[row + 1, col - 2] == 0 and \
    #                     image.new_image[row + 2, col] == 0 and \
    #                     image.new_image[row + 2, col + 1] == 0 and \
    #                     image.new_image[row + 2, col - 1] == 0 and \
    #                     image.new_image[row + 2, col + 2] == 0 and \
    #                     image.new_image[row + 2, col - 2] == 0 and \
    #                     image.new_image[row - 2, col] == 0 and \
    #                     image.new_image[row - 2, col - 1] == 0 and \
    #                     image.new_image[row - 2, col + 1] == 0 and \
    #                     image.new_image[row - 2, col - 2] == 0 and \
    #                     image.new_image[row - 2, col + 2] == 0:
    #                 image.new_image[row, col] = 123
    count = 0
    circles = np.ones(shape=(len(image.new_image), len(image.new_image[0]), 3), dtype=np.uint8)

    for row in range(1, image.height, 1):
        for col in range(1, image.width, 1):
            if image.new_image[row, col] == 0:
                continue
            is_stone = image.new_image[row-1, col-1]
            is_stone += image.new_image[row-1, col]
            is_stone += image.new_image[row-1, col+1]

            is_stone += image.new_image[row, col-1]
            is_stone += image.new_image[row, col+1]

            is_stone += image.new_image[row+1, col-1]
            is_stone += image.new_image[row+1, col]
            is_stone += image.new_image[row+1, col+1]

            if is_stone == 0:
                count += 1
                cv2.circle(circles, (col, row), 6, (0, 0, 255), 2)

    print('TEST', count)
    count = 0

    image.original_image = np.abs(image.original_image - 50)
    image.update_image(np.abs(image.new_image), '-final')

    a = cv2.imread('images/stones/stones.jpg')
    b = cv2.addWeighted(a, 0.7, circles, 1, 0.0)

    cv2.imwrite('images/stones/rgb-final-' + str(count) + '.jpg', b)

    image.save_image()


def stones_2():
    image = MyImage.load_image('images/stones/', 'stones', np.uint8)
    image.treshold(150)
    image.update_image(image.new_image, '-thed')
    m = image.new_image

    kernel1 = np.ones((6, 1), 'uint8')  # x, 1 - vertical; 1, x - horizontal
    kernel2 = np.ones((7, 1), 'uint8')  # x, 1 - vertical; 1, x - horizontal
    # kernel3 = np.ones((1, 1), 'uint8')
    pix_eros1 = cv2.erode(m, kernel1, iterations=1)
    image.update_image(pix_eros1, '-erosed-' + str(kernel1.shape[0]))
    image.save_image()
    pix_eros2 = cv2.erode(m, kernel2, iterations=1)
    # pix_eros2 = cv2.dilate(pix_eros2, kernel3, iterations=1)

    image.update_image(pix_eros2, '-erosed-' + str(kernel2.shape[0]))
    image.save_image()

    # erosion_result = sub(pix_eros1, pix_eros2)
    erosion_result = np.subtract(pix_eros1, pix_eros2)

    image.update_image(erosion_result, '-erosed')
    image.save_image()

    for row in range(2, image.height-2):
        for col in range(2, image.width-2):
            if image.new_image[row, col] == 255:
                if image.new_image[row, col-1] == 0 and \
                        image.new_image[row, col+1] == 0 and \
                        image.new_image[row, col-2] == 0 and \
                        image.new_image[row, col+2] == 0 and \
                        image.new_image[row-1, col] == 0 and \
                        image.new_image[row-1, col-1] == 0 and \
                        image.new_image[row-1, col+1] == 0 and \
                        image.new_image[row-1, col+2] == 0 and \
                        image.new_image[row-1, col-2] == 0 and \
                        image.new_image[row+1, col] == 0 and \
                        image.new_image[row+1, col-1] == 0 and \
                        image.new_image[row+1, col+1] == 0 and \
                        image.new_image[row+1, col+2] == 0 and \
                        image.new_image[row+1, col-2] == 0 and \
                        image.new_image[row+2, col] == 0 and \
                        image.new_image[row+2, col+1] == 0 and \
                        image.new_image[row+2, col-1] == 0 and \
                        image.new_image[row+2, col+2] == 0 and \
                        image.new_image[row+2, col-2] == 0 and \
                        image.new_image[row-2, col] == 0 and \
                        image.new_image[row-2, col-1] == 0 and \
                        image.new_image[row-2, col+1] == 0 and \
                        image.new_image[row-2, col-2] == 0 and \
                        image.new_image[row-2, col+2] == 0:
                    image.new_image[row, col] = 123

    count = 0
    for row in range(image.height):
        for col in range(image.width):
            if image.new_image[row, col] == 123:
                count += 1
    print('Кол-во камней = ', count)

    for row in range(image.height):
        for col in range(image.width):
            if image.new_image[row, col] == 123:
                image.new_image[row, col] = 238
            else:
                image.new_image[row, col] = 0
    circles = np.ones(shape=(len(image.new_image), len(image.new_image[0]), 3), dtype=np.uint8)
    for row in range(image.height):
        for col in range(image.width):
            if image.new_image[row, col] == 238:
                cv2.circle(circles, (col, row), 6, (0, 0, 255), 2)

    image.original_image = np.abs(image.original_image - 50)
    image.update_image(np.abs(image.new_image), '-final')

    a = cv2.imread('images/stones/stones.jpg')
    b = cv2.addWeighted(a, 0.7, circles, 1, 0.0)

    cv2.imwrite('images/stones/rgb-final-vertical-' + str(count) + '.jpg', b)

    image.save_image()


SIZE = 7


def test():
    # считывается изображение
    my_image = MyImage.load_image('images/stones/', 'stones', np.uint8)
    buff_image = MyImage.load_image('images/stones/', 'stones', np.uint8)
    image = my_image.new_image
    # image = cv.imread("images/stones/stones.jpg")

    # приводим к шкале серости
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # бинаризируем
    _, binary = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    plot.imshow(binary, cmap="gray")
    plot.show()
    buff_image.update_image(binary, '-testBinary')
    buff_image.save_image()

    # Для разделения двух объектов на картинке
    # расчитываем расстояние от белого писеля до ближайшего чёрного
    # и отделяем пересекающиеся контуры друг от друга
    D = ndimage.distance_transform_edt(binary)
    localMax = peak_local_max(D, indices=False, min_distance=5, labels=binary)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=binary)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # массив с маской
    masks = np.zeros(gray_image.shape, dtype="uint8")
    masks = cv.cvtColor(masks, cv.COLOR_GRAY2RGB)

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print('Huh?', len(contours))
    count = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray_image.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        max_contour = max(cnts, key=cv.contourArea)
        area = cv.contourArea(max_contour)
        x, y, w, h = cv.boundingRect(max_contour)
        # проверка на размер
        #if (int(h) == 7 and int(h) > int(w)) or (int(w) == 7 and int(w) > int(h)):
        if int(h) == 7 and int(w) == 7:
            cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 1)
            cv.rectangle(masks, (x, y), (x+w, y+h), (255, 255, 255), -1)
            count += 1
            # print('Found stone: width =', w, 'height =', h, 'Total count:', count)
    # image = cv.drawContours(image, contours[1], -1, (0, 255, 0), 2)
    plot.imshow(image)
    plot.show()
    buff_image.update_image(image, '-testFinalAllSides-' + str(count))
    buff_image.save_image()
    plot.imshow(masks)
    plot.title('masks')
    plot.show()
    buff_image.update_image(masks, '-testMask')
    buff_image.save_image()

    print(count)


def test2():
    image = MyImage.load_image('images/stones/', 'stones', np.uint8)
    buf_image = MyImage.load_image('images/stones/', 'stones', np.uint8)
    erosion_level = SIZE - 1

    image.treshold(135)
    eroded = morphological_operator(image.new_image, erosion_level, erosion_level, 'erosion')
    image.update_image(eroded, '-eroded')
    image_row, image_col = image.new_image.shape
    pad_height = 1
    pad_width = 1

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image.new_image
    buf_image.update_image(padded_image, '-paddedStart')
    buf_image.save_image()
    stones = 0
    circles = np.ones(shape=(len(image.new_image), len(image.new_image[0]), 3), dtype=np.uint8)

    for i in range(1, image_row + 2, 1):
        for j in range(1, image_col + 2, 1):
            if padded_image[i][j] == 0:
                continue

            is_stone = padded_image[i - 1][j - 1]
            is_stone += padded_image[i - 1][j]
            is_stone += padded_image[i - 1][j + 1]

            is_stone += padded_image[i][j - 1]
            is_stone += padded_image[i][j + 1]

            is_stone += padded_image[i + 1][j - 1]
            is_stone += padded_image[i + 1][j]
            is_stone += padded_image[i + 1][j + 1]

            if is_stone == 0:
                stones += 1
                cv2.circle(circles, (i, j), 6, (0, 0, 255), 2)

    a = cv2.imread('images/stones/stones.jpg')
    b = cv2.addWeighted(a, 0.7, circles, 1, 0.0)

    cv2.imwrite('images/stones/rgb-test2-' + str(stones) + '.jpg', b)
    print(stones)


if __name__ == '__main__':
    # lab_1()  # ЛР №3 внутри
    # lab_2()  # ЛР №3 внутри
    # lab_4()

    # ---------- Лабораторная 5 ----------
    # images = [
    #     MyImage.load_image('images/', 'photo1', np.uint8),
    #     MyImage.load_image('images/', 'photo2', np.uint8),
    #     MyImage.load_image('images/', 'photo3', np.uint8),
    #     MyImage.load_image('images/', 'photo4', np.uint8),
    #     MyImage.load_image('images/', 'HollywoodLC', np.uint8),
    #     MyImage.load_image('images/', 'grace', np.uint8)
    # ]
    #
    # xcrs = [
    #     MyImage.load_binary('images/', 'u0', '>H', 2048, 2500, 5120),
    #     MyImage.load_binary('images/', 'c12-85v', '>H', 1024, 1024, 5120)
    # ]
    #
    # for image in images:
    #     lab_5(image)
    #
    # for image in xcrs:
    #     image.rotate90_ccw()
    #     negative(image)
    #
    #     lab_5(image)
    # ------------------------------------

    # lab_6()
    # lab_7()
    # lab_8()
    # lab_9()
    # lab_9_beautify()
    # lab_10_template()

    # ---------- Лабораторная 11 ----------
    # images = [
    #     MyImage.load_image('images/lab11/', 'model', np.uint8),
    #     MyImage.load_image('images/lab11/', 'model-noize', np.uint8),
    #     MyImage.load_image('images/lab11/', 'model-noize-linearFiltered', np.uint8),
    #     MyImage.load_image('images/lab11/', 'model-noize-medianFiltered', np.uint8)
    # ]
    # for imagimage_sd.modified_imagee in images:
    #     lab_11_gradient(image)
    #     lab_11_gradient(image, 'vh')
    #     lab_11_gradient(image, 'd')
    #     lab_11_laplasian(image)
    # -------------------------------------

    # ---------- Лабораторная 12 ----------
    # images = [
    #     MyImage.load_image('images/lab12/', 'model', np.uint8),
    #     MyImage.load_image('images/lab12/', 'model-noize', np.uint8),
    #     MyImage.load_image('images/lab12/', 'model-noize-linearFiltered', np.uint8),
    #     MyImage.load_image('images/lab12/', 'model-noize-medianFiltered', np.uint8),
    #     MyImage.load_image('images/lab12/', 'rock', np.uint8)
    # ]
    # for image in images:
    #     lab_12(image)
    # -------------------------------------

    # lab_12()
    # ---------- Творческое задание №1 ----------
    # images = [
    #          dat_image_binary_read('images/mrt/brain-h/', 'brain-H_x512', '.bin', 512, 512, format='h'),
    #          dat_image_binary_read('images/mrt/brain-v/', 'brain-V_x256', '.bin', 256, 256, format='h'),
    #          dat_image_binary_read('images/mrt/spine-h/', 'spine-H_x256', '.bin', 256, 256, format='h'),
    #          dat_image_binary_read('images/mrt/spine-v/', 'spine-V_x512', '.bin', 512, 512, format='h')
    #
    #     ]
    # for image in images:
    #     lab_mrt(image)
    # -------------------------------------------
    stones()  # тестовая версия по двум сторонам
    # stones_2()  # тестовая версия по одной стороне
    # test()  # финальная версия
    # test2()

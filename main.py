import numpy as np

import func
from func import *
from matplotlib import pyplot as plot
from MyImage import *
from MyFourier import *


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
    save(c12_xcr_grayscaled, 'c12_xcr_grayscaled.jpg')
    # new_image(c12_xcr_grayscaled, width12, height12).show()

    u0_xcr_input = read_xcr('images/u0.xcr', 5120, width0, height0)
    u0_xcr_rotated = np.rot90(np.array(u0_xcr_input))
    u0_xcr_grayscaled = grayscale(np.array(u0_xcr_rotated))
    save(u0_xcr_grayscaled, 'u0_xcr_grayscaled.jpg')
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
    plot.savefig(image.dir + 'plots/' + image.fname + 'OrigHist')
    plot.show()
    plot.plot(cdf_calc(hist[0])[0])
    plot.savefig(image.dir + 'plots/' + image.fname + 'OrigCdf')
    plot.show()

    equalize_img(image)

    # image.show_image()

    image.save_image()

    img = MyImage.load_image('images/', image.new_fname, np.uint8)
    # img.show_image()
    h = histogram_img(img.new_image, img.colors())
    plot.plot(h[0])
    plot.savefig(img.dir + 'plots/' + img.fname + 'EqHist')
    plot.show()
    plot.plot(cdf_calc(h[0])[0])
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
    image = MyImage.load_image('images/lab7/', 'model', np.uint8)

    salt_and_pepper(image, 20)
    image.save_image()

    image_lf = image.copy_image()
    #image_lf.fname += str('test_lf')

    linear_filter(image_lf, 3)
    image_lf.save_image()

    image_mf = image.copy_image()
    #image_mf.fname += str('test_mf')

    median_filter(image_mf, 3)
    image_mf.save_image()


def lab_8():
    dt = 0.005
    cardio = cardiogram(dt)
    plot.plot(cardio)
    plot.title("Cardiogram")
    plot.show()

    four = MyFourier.fourier_one_dimension(cardio, dt)
    plot.plot(four.amplitude())
    plot.title("Amplitude")
    plot.show()

    four_back = four.four_inverse_one_dimensional()
    plot.plot(four_back)
    plot.title("Inverse fourier")
    plot.show()
    image = MyImage.load_image('images/', 'test', np.uint8)
    power_grad(image.new_image, 2, 3)
    twod = four.fourier_two_dimension(four, image.new_image)
    image.update_image(np.array(twod), '-twoD')
    image.save_image()

    image = MyImage.load_image('images/', 'grace', np.uint8)
    twod2 = twodfour(image.new_image)
    twod2 = np.fft.ifft2(twod)
    image.update_image(np.array(twod2), '-twoD_vstroen_back')
    image.save_image()



if __name__ == '__main__':
    # lab_1()  # ЛР №3 внутри
    # lab_2()  # ЛР №3 внутри
    # lab_4()

    # Лабораторная 5
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
    # percent = 100 / len(images)
    # completed = percent
    # for image in images:
    #     lab_5(image)
    #     print(str(completed) + '% completed')
    #     completed += percent
    #
    # percent = 100 / len(xcrs)
    # completed = percent
    # for image in xcrs:
    #     image.rotate90_ccw()
    #     negative(image)
    #
    #     lab_5(image)
    #     print(str(completed) + '% completed')
    #     completed += percent

    # lab_6()
    # lab_7()
    lab_8()


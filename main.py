# --coding:utf-8--
import string

import cv2
import numpy as np
import time
import threading
import sys
import torch

NUM_DIGITS = 8
PIXEL_SAMPLE_SIZE = 3  # Square calculation , must be odd number.


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])


def binary_decode(bin_input):
    result = 0
    for i in range(NUM_DIGITS - 1, -1, -1):
        result += bin_input[i] * (10 ** (NUM_DIGITS - i - 1))
    return int(str(result), 2)


def rgb_2_binary(cv2_img_object):
    start_time = time.time()
    print('rgb_2_binary encoding start')
    result = np.zeros((cv2_img_object.shape[0], cv2_img_object.shape[1], cv2_img_object.shape[2], NUM_DIGITS),
                      dtype="uint8")
    for y in range(0, cv2_img_object.shape[0]):
        for x in range(0, cv2_img_object.shape[1]):
            for z in range(0, cv2_img_object.shape[2]):
                result[y][x][z] = binary_encode(cv2_img_object[y][x][z], NUM_DIGITS)
                # result[y][x][z] = bin(result[y][x][z])

    # TODO rgb_2_binary use too much time to process(about 2 min), To be optimized.
    # cv2_img_object = cv2_img_object[:, :, :, np.newaxis]
    # print(cv2_img_object[0][0][0])
    # exit(0)
    # for x in np.nditer(cv2_img_object, op_flags=['readwrite']):
    #     x[...] = binary_encode(x, NUM_DIGITS)
    # return cv2_img_object

    print('rgb_2_binary encoding completed, use time:', start_time - time.time(), 's')
    return result


def binary_2_rgb(rgb_array):
    start_time = time.time()
    print('binary_2_rgb encoding start')
    result = np.zeros((rgb_array.shape[0], rgb_array.shape[1], rgb_array.shape[2]), dtype='uint8')
    for y in range(0, rgb_array.shape[0]):
        for x in range(0, rgb_array.shape[1]):
            for z in range(0, rgb_array.shape[2]):
                result[y][x][z] = binary_decode(rgb_array[y][x][z])
    print('binary_2_rgb encoding completed, use time:', -(start_time - time.time()), 's')
    return result


def fuzzy_process(image_object, scale_num, interpolation_function=cv2.INTER_CUBIC):
    # There's two solutions:
    # 1) Use the fuzzy_process function to production training data,
    # then the image scale is processing in fuzzy_process function.
    # 2) Fuzzy_process
    return cv2.resize(
        cv2.resize(image_object, (0, 0), fx=1 / scale_num, fy=1 / scale_num, interpolation=interpolation_function),
        (0, 0), fx=scale_num, fy=scale_num, interpolation=interpolation_function)


def pixel_sample_2_binary(image_object):
    image_object = rgb_2_binary(image_object)  # Convert the image data to binary.

    image_object.flatten()
    image_object.resize(image_object.shape[0], image_object.shape[1], image_object.shape[2] * image_object.shape[3])
    image_object = np.insert(image_object, image_object.shape[2], values=np.ones(image_object.shape[1]), axis=2)
    #
    # print
    # print(sys.getsizeof(image_object))
    return image_object


def pixel_sample_chunk(image_object, chunk_size=3, pixel_location=None):
    if pixel_location is None:
        pixel_location = [0, 0]

    if chunk_size % 2 != 0:  # Convert even to odd.
        chunk_size -= 1
        print('Adjust chunk_size to ', chunk_size)

    if chunk_size < 3:  # If variable less than three, assignment to three.
        chunk_size = 3
        print('Adjust chunk_size to ', chunk_size)

    result = np.zeros((PIXEL_SAMPLE_SIZE, PIXEL_SAMPLE_SIZE, 25), dtype='uint8')

    for y in range(chunk_size):
        for x in range(chunk_size):
            result[y][x] = image_object[y + pixel_location[1]][x + pixel_location[0]]

    return result


def chunk_2_img(chunk):
    chunk_size = [chunk.shape[0], chunk.shape[1]]

    for y in range(chunk.shape[0]):
        for x in range(chunk.shape[1]):
            chunk[y][x] = chunk[y][x][:-1]

    chunk.flatten()

    chunk.resize(chunk_size[0], chunk_size[1], 3, 8)

    for y in range(chunk[0]):
        for x in range(chunk[1]):
            for z in range(chunk[2]):
                chunk[y][x][z] = binary_decode(chunk[y][x][z])

    return chunk


img = cv2.imread('th.jpg')

cv2.imshow('src', chunk_2_img(pixel_sample_chunk(pixel_sample_2_binary(fuzzy_process(img, 4)), 500, [0, 0])))

# cv2.imshow('src', fuzzy_process(img, 4))
# cv2.imwrite('fuzzy_th.jpg', fuzzy_process(img, 4))
cv2.waitKey()

# print(pixel_sample_2_binary(img))

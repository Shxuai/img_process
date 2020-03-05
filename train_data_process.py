# --coding:utf-8--
import os
import string

import cv2
import numpy as np
import time
import threading
import sys
import torch

NUM_DIGITS = 8
PIXEL_SAMPLE_SIZE = 3  # Square calculation , must be odd number.
FUZZY_RATE = 4
CHUNK_SIZE = 500


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
                      dtype=np.uint8)
    for y in range(0, cv2_img_object.shape[0]):
        print("rgb_2_binary encoding complete: " + str(round(y / cv2_img_object.shape[0] * 100, 2)) + "%")
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
    result = np.zeros((rgb_array.shape[0], rgb_array.shape[1], rgb_array.shape[2]), dtype=np.uint8)
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
    print("Function pixel_sample_chunk start")
    if pixel_location is None:
        pixel_location = [0, 0]

    if chunk_size % 2 == 1:  # Convert even to odd.
        chunk_size -= 1
        print('Adjust chunk_size to ', chunk_size)

    if chunk_size < 3:  # If variable less than three, assignment to three.
        chunk_size = 3
        print('Adjust chunk_size to ', chunk_size)

    result = np.zeros((chunk_size, chunk_size, 25), dtype=np.uint8)

    start_y = int((chunk_size - 1) / 2 - pixel_location[0])
    start_x = int((chunk_size - 1) / 2 - pixel_location[1])

    if start_y < 0:
        start_y = 0
    if start_x < 0:
        start_x = 0

    end_y = (chunk_size - 1) / 2 + (image_object.shape[0] - pixel_location[0])
    end_x = (chunk_size - 1) / 2 + (image_object.shape[1] - pixel_location[1])

    if end_y > result.shape[0]:
        end_y = result.shape[0]
    if end_x > result.shape[1]:
        end_x = result.shape[1]

    for start_y in range(start_y, end_y):
        start_x = int((chunk_size - 1) / 2 - pixel_location[1])
        if start_x < 0:
            start_x = 0
        for start_x in range(start_x, end_x):
            result[start_y][start_x] = image_object[int(pixel_location[0] - (chunk_size - 1) / 2 + start_y)][
                int(pixel_location[1] - (chunk_size - 1) / 2 + start_x)]

    print("Function pixel_sample_chunk end")
    return result


def chunk_2_img(chunk):
    chunk_size = [chunk.shape[0], chunk.shape[1]]
    chunk_process = np.zeros((chunk.shape[0], chunk.shape[1], chunk.shape[2] - 1), dtype=np.uint8)
    chunk_result = np.zeros((chunk.shape[0], chunk.shape[1], 3), dtype=np.uint8)

    for y in range(chunk.shape[0]):
        for x in range(chunk.shape[1]):
            chunk_process[y][x] = chunk[y][x][:-1]

    chunk_process.flatten()

    chunk_process.resize(chunk_size[0], chunk_size[1], 3, 8)

    for y in range(chunk_process.shape[0]):
        for x in range(chunk_process.shape[1]):
            for z in range(chunk_process.shape[2]):
                chunk_result[y][x][z] = binary_decode(chunk_process[y][x][z])

    chunk_result.resize(chunk_size[0], chunk_size[1], 3)
    return chunk_result


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.jpg':
            list_name.append(file_path)
        elif os.path.splitext(file_path)[1] == '.JPG':
            list_name.append(file_path)
        elif os.path.splitext(file_path)[1] == '.png':
            list_name.append(file_path)
        elif os.path.splitext(file_path)[1] == '.PNG':
            list_name.append(file_path)


if __name__ == '__main__':

    img_path_list = []
    listdir("./resource/img_data/", img_path_list)

    for t in range(len(img_path_list)):
        if not os.path.exists(img_path_list[t] + ".dir/"):
            os.mkdir(img_path_list[t] + ".dir/")
        img = cv2.imread(img_path_list[t])
        img_bin = pixel_sample_2_binary(fuzzy_process(img, FUZZY_RATE))
        for y in range(img_bin.shape[0]):
            for x in range(img_bin.shape[1]):
                chunk = pixel_sample_chunk(img_bin, CHUNK_SIZE, [y, x])
                pixel = np.array(img[y][x]).astype(np.uint8)
                chunk.tofile(img_path_list[t] + ".dir/[" + str(y) + "," + str(x) + "].chunk")
                pixel.tofile(img_path_list[t] + ".dir/[" + str(y) + "," + str(x) + "].pixel")
                cv2.imshow('src', chunk_2_img(chunk))
                cv2.waitKey()

    test_img = chunk_2_img(pixel_sample_chunk(img_bin, CHUNK_SIZE, [0, 0]))

    cv2.imshow('src', test_img)

    # cv2.imshow('src', fuzzy_process(img, 4))
    # cv2.imwrite('fuzzy_th.jpg', fuzzy_process(img, 4))
    cv2.waitKey()

    # print(pixel_sample_2_binary(img))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import matplotlib.image as plt_img
from collections import Counter
# from flask import request
import cv2


def invert():
    image = cv2.imread("static/img/img_now.jpg")
    new_image = cv2.bitwise_not(image)
    cv2.imwrite("static/img/img_now.jpg", new_image)


def grayscale():
    image = cv2.imread("static/img/img_now.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/img/img_now.jpg", gray)


def brightness(value):
    img = cv2.imread("static/img/img_now.jpg")
    # Hệ màu HSV/HSB
    # H: Vùng màu
    # S: Độ bão hoà màu
    # V(B): Độ sáng
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    print("processing")
    print(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite("static/img/img_now.jpg", img)


def kmeans(k):
    # k : number of area
    image = cv2.imread("static/img/img_now.jpg")
    # convert to RGB
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    new_image = new_image.reshape((-1, 3))
    # convert to float
    new_image = np.float32(image)
    # xác định tiêu chí, số lượng cụm (K) và áp dụng công thức hàm kmeans ()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, label, center = cv2.kmeans(new_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # sau đó chuyển đổi trở lại thành uint8 và tạo hình ảnh gốc
    center = np.uint8(center)
    res = center[label.flatten()]
    new_image = res.reshape(image.shape)
    cv2.imwrite("static/img/img_now.jpg", new_image)


def edge_detection():
    image = cv2.imread("static/img/img_now.jpg")
    # Canny edge detection
    img = cv2.Canny(image, 30, 70)
    cv2.imwrite("static/img/img_now.jpg", img)


def threshold(low, high):
    image = cv2.imread("static/img/img_now.jpg")
    # cat nguong
    ret, th = cv2.threshold(image, low, high, cv2.THRESH_BINARY)
    # ret, th2 = cv2.threshold(image, low, high, cv2.THRESH_BINARY_INV)
    # ret, th3 = cv2.threshold(image, low, high, cv2.THRESH_TRUNC)
    # ret, th4 = cv2.threshold(image, low, high, cv2.THRESH_TOZERO)
    # ret, th5 = cv2.threshold(image, low, high, cv2.THRESH_TOZERO_INV)
    cv2.imwrite("static/img/img_now.jpg", th)


def grabcut(bit):
    # Đọc hình ảnh trong thang độ xám
    image = cv2.imread("static/img/img_now.jpg")
    # Lặp lại từng pixel và thay đổi giá trị pixel thành nhị phân bằng cách sử dụng np.binary_repr () và lưu trữ
    # trong danh sách
    lst = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lst.append(np.binary_repr(image[i][j], width=8))  # width = no. of bits

    if bit == 1:
        new_image = (np.array([int(i[7]) for i in lst], dtype=np.uint8) * 1).reshape(image.shape[0], image.shape[1])
    elif bit == 2:
        new_image = (np.array([int(i[6]) for i in lst], dtype=np.uint8) * 2).reshape(image.shape[0], image.shape[1])
    elif bit == 3:
        new_image = (np.array([int(i[5]) for i in lst], dtype=np.uint8) * 4).reshape(image.shape[0], image.shape[1])
    elif bit == 4:
        new_image = (np.array([int(i[4]) for i in lst], dtype=np.uint8) * 8).reshape(image.shape[0], image.shape[1])
    elif bit == 5:
        new_image = (np.array([int(i[3]) for i in lst], dtype=np.uint8) * 16).reshape(image.shape[0], image.shape[1])
    elif bit == 6:
        new_image = (np.array([int(i[2]) for i in lst], dtype=np.uint8) * 32).reshape(image.shape[0], image.shape[1])
    elif bit == 7:
        new_image = (np.array([int(i[1]) for i in lst], dtype=np.uint8) * 64).reshape(image.shape[0], image.shape[1])
    else:
        new_image = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(image.shape[0], image.shape[1])
    cv2.imwrite("static/img/img_now.jpg", new_image)


def sharpening():
    image = cv2.imread("static/img/img_now.jpg", flags=cv2.IMREAD_COLOR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    new_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    cv2.imwrite("static/img/img_now.jpg", new_image)


def smoothing(matrix):
    image = cv2.imread("static/img/img_now.jpg")
    # lọc trung vị nxn
    new_image = cv2.blur(image, (matrix, matrix))
    cv2.imwrite("static/img/img_now.jpg", new_image)


def zoomin():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def zoomout():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x / 2)):
        for j in range(0, int(y / 2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    new_arr = np.uint8(new_arr)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_left():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr + 100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr - 100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    h_img, w_img, _ = img.shape
    out = np.zeros((h_img - 2, w_img - 2), dtype=np.float)
    new_img = np.zeros((h_img - 2, w_img - 2, 3))
    if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]):
        array = img[:, :, 0]
        for h in range(h_img - 2):
            for w in range(w_img - 2):
                S = np.multiply(array[h:h + 3, w:w + 3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        for channel in range(3):
            new_img[:, :, channel] = out_
    else:
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img - 2):
                for w in range(w_img - 2):
                    S = np.multiply(array[h:h + 3, w:w + 3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
    new_img = np.uint8(new_img)
    return new_img


def blur():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True


def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :, 0].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)
        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()


def df(img):  # to make a histogram (count distribution frequency)
    values = [0] * 256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele * 255 / cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv2.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/img/img_now.jpg', image_equalized)

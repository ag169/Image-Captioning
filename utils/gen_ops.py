from PIL import Image
import numpy as np
import cv2


MAX = 1e10
MAX_RAD = 10
MAX_DEG = 500


# STRING OPS
def lowercase(string: str):
    return string.lower()


def uppercase(string: str):
    return string.upper()


# MATH FUNCTIONS
def round_down(a, n=10):
    a -= a % n
    return a


def round_up(a, n=10):
    a -= a % n
    return a + n


def round_nearest(a, n=10):
    rem = a % n
    half_n = 1.0 * n / 2

    a -= rem
    if rem >= half_n:
        a += n

    return a


def distance(a, b, order=2):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b, ord=order)


def cartesian2rad(pt: tuple, origin=(0, 0), deg=False, p=False):
    r = distance(pt, origin, order=2)
    if p:
        print(pt, origin)
    if pt[0] >= origin[0]:
        if pt[1] >= origin[1]:
            if p:
                print(1)
            theta = np.arctan(1.0 * (pt[1] - origin[1]) / (pt[0] - origin[0] + 0.00000001))
        else:
            if p:
                print(4)
            theta = 2 * np.pi - np.arctan(1.0 * (origin[1] - pt[1]) / (pt[0] - origin[0] + 0.00000001))
    else:
        if pt[1] >= origin[1]:
            if p:
                print(2)
            theta = np.pi - np.arctan(1.0 * (pt[1] - origin[1]) / (origin[0] - pt[0] + 0.00000001))
        else:
            if p:
                print(3)
            theta = np.pi + np.arctan(1.0 * (origin[1] - pt[1]) / (origin[0] - pt[0] + 0.00000001))
    if p:
        print('*'*20)

    if deg:
        theta = 180 * theta / np.pi

    return r, theta


# MATH CONVERSION FUNCTIONS
def rad2deg(angle_rad):
    return 180.0 * angle_rad / np.pi


def deg2rad(angle_deg):
    return np.pi * angle_deg / 180


# OBJECT CONVERSION FUNCTIONS
def pil2np(img):
    return np.asarray(img)


def np2pil(array):
    return Image.fromarray(array)


def int2tuple(integer: int, n=2):
    if n < 0:
        raise ValueError
    t = (integer, )
    for i in range(1, n):
        t = t + (integer, )
    return t


def tensor2np(tensor):
    return tensor.detach().cpu().numpy()


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.n = 0

    def reset(self):
        self.__init__()

    def update(self, value, n=1):
        self.sum += value * n
        self.n += n
        self.avg = self.sum / self.n


# IMAGE UTILS
def make_square(img, size=256, mode='RGB'):
    img.thumbnail((size, size))
    x, y = img.size
    new_im = Image.new(mode, (size, size), "black")
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def bounding_boxes(img, thresh=25):
    if type(img) != np.ndarray:
        img = pil2np(img)

    img = img > thresh

    ind = np.nonzero(img.any(axis=0))[0]
    w1 = ind[0]
    w2 = ind[-1]
    ind = np.nonzero(img.any(axis=1))[0]
    h1 = ind[0]
    h2 = ind[-1]

    return [w1, h1, w2, h2]


def max_rot_clk(img, thresh=25, bb=None, max_deg=45, op_int=False, margin=0.05):
    w, h = img.size
    if bb is None:
        [w1, h1, w2, h2] = bounding_boxes(img, thresh)
    else:
        [w1, h1, w2, h2] = bb

    if margin > 0:
        m_w = margin * w
        m_h = margin * h

        if w1 <= m_w or w2 >= w - m_w or h1 <= m_h or h2 >= h - m_h:
            return 0

    o = (float(w) / 2, float(h) / 2)

    deg = False
    cis1 = cartesian2rad((w2, h - h1), origin=o, deg=deg)
    cis2 = cartesian2rad((w1, h - h1), origin=o, deg=deg)
    cis3 = cartesian2rad((w1, h - h2), origin=o, deg=deg)
    cis4 = cartesian2rad((w2, h - h2), origin=o, deg=deg)

    # print(cis1)
    # print(cis2)
    # print(cis3)
    # print(cis4)

    alpha1 = 0
    if h < 2 * cis1[0]:
        alpha1 = np.arcsin(h / (2 * cis1[0])) - cis1[1]
    if alpha1 <= 0:
        alpha1 = MAX_RAD

    alpha2 = 0
    if w < 2 * cis2[0]:
        alpha2 = np.pi - np.arccos(w / (2 * cis2[0])) - cis2[1]
    if alpha2 <= 0:
        alpha2 = MAX_RAD

    alpha3 = 0
    if h < 2 * cis3[0]:
        alpha3 = np.pi + np.arcsin(h / (2 * cis3[0])) - cis3[1]
    if alpha3 <= 0:
        alpha3 = MAX_RAD

    alpha4 = 0
    if w < 2 * cis4[0]:
        alpha4 = 2 * np.pi - np.arccos(w / (2 * cis4[0])) - cis4[1]
    if alpha4 <= 0:
        alpha4 = MAX_RAD

    # print(rad2deg(alpha1))
    # print(rad2deg(alpha2))
    # print(rad2deg(alpha3))
    # print(rad2deg(alpha4))

    deg_min = rad2deg(min(alpha1, alpha2, alpha3, alpha4))
    deg_min = min(deg_min, max_deg)
    if op_int:
        deg_min = int(deg_min)
    return deg_min


def max_rot_aclk(img, thresh=25, bb=None, max_deg=45, op_int=False, margin=0.05):
    w, h = img.size
    if bb is None:
        [w1, h1, w2, h2] = bounding_boxes(img, thresh)
    else:
        [w1, h1, w2, h2] = bb

    if margin > 0:
        m_w = margin * w
        m_h = margin * h

        if w1 <= m_w or w2 >= w - m_w or h1 <= m_h or h2 >= h - m_h:
            return 0

    o = (float(w) / 2, float(h) / 2)

    deg = False
    cis1 = cartesian2rad((w2, h - h1), origin=o, deg=deg)
    cis2 = cartesian2rad((w1, h - h1), origin=o, deg=deg)
    cis3 = cartesian2rad((w1, h - h2), origin=o, deg=deg)
    cis4 = cartesian2rad((w2, h - h2), origin=o, deg=deg)

    # print(cis1)
    # print(cis2)
    # print(cis3)
    # print(cis4)

    alpha1 = 0
    if w < 2 * cis1[0]:
        alpha1 = cis1[1] - np.arccos(w / (2 * cis1[0]))
    if alpha1 <= 0:
        alpha1 = MAX_RAD

    alpha2 = 0
    if h < 2 * cis2[0]:
        alpha2 = cis2[1] + np.arcsin(h / (2 * cis2[0])) - np.pi
    if alpha2 <= 0:
        alpha2 = MAX_RAD

    alpha3 = 0
    if w < 2 * cis3[0]:
        alpha3 = cis3[1] - np.arccos(w / (2 * cis3[0])) - np.pi
    if alpha3 <= 0:
        alpha3 = MAX_RAD

    alpha4 = 0
    if h < 2 * cis4[0]:
        alpha4 = cis4[1] + np.arcsin(h / (2 * cis4[0])) - 2 * np.pi
    if alpha4 <= 0:
        alpha4 = MAX_RAD

    # print(rad2deg(alpha1))
    # print(rad2deg(alpha2))
    # print(rad2deg(alpha3))
    # print(rad2deg(alpha4))

    deg_min = rad2deg(min(alpha1, alpha2, alpha3, alpha4))
    deg_min = min(deg_min, max_deg)
    if op_int:
        deg_min = int(deg_min)
    return deg_min


# TEST FUNCTIONS
def test_bb():
    i_num = 951
    l_path = "E:\\Python Projects\\Datasets\\MSRA10K_Imgs_GT\\GT\\" + str(i_num) + ".png"
    i_path = "E:\\Python Projects\\Datasets\\MSRA10K_Imgs_GT\\Imgs\\" + str(i_num) + ".jpg"

    img = Image.open(l_path)
    c = bounding_boxes(img)
    print(c)

    img = Image.open(i_path)
    img = pil2np(img)

    cv2.rectangle(img, (c[0], c[1]), (c[2], c[3]), (255, 0, 0), 2)

    img = np2pil(img)
    img.show()


def test_max_rot():
    i_num = 76753
    l_path = "E:\\Python Projects\\Datasets\\MSRA10K_Imgs_GT\\GT\\" + str(i_num) + ".png"
    i_path = "E:\\Python Projects\\Datasets\\MSRA10K_Imgs_GT\\Imgs\\" + str(i_num) + ".jpg"

    img = Image.open(l_path)
    c = bounding_boxes(img)
    print(c)

    deg1 = max_rot_clk(img, bb=c, op_int=True, margin=0)
    deg2 = max_rot_aclk(img, bb=c, op_int=True, margin=0)
    print('Deg clk: ', deg1)
    print('Deg aclk: ', deg2)

    img = Image.open(i_path)
    img = pil2np(img)

    cv2.rectangle(img, (c[0], c[1]), (c[2], c[3]), (255, 0, 0), 2)

    img = np2pil(img)
    img1 = img.rotate(deg1)
    img2 = img.rotate(-deg2)

    img1.show()
    img2.show()


if __name__ == '__main__':
    # test_bb()
    test_max_rot()


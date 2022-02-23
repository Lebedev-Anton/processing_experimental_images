import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

path = '/home/anton/projects/processing_experimental_images/test_images'
image_name = 'Lab3_0002.JPG'
# image = import_image(os.path.join(path, image_name))
full_path = os.path.join(path, image_name)

image_name = 'template.JPG'
# image = import_image(os.path.join(path, image_name))
template_path = os.path.join(path, image_name)


def read_image_from_path(path, color_space):
    if color_space == 'COLOR':
        space = cv2.IMREAD_COLOR
    else:
        space = cv2.IMREAD_GRAYSCALE
    return cv2.imread(path, space)


def resize_img(img, percent_resize=20):
    h, w = img.shape[:2]
    resize_height = int(h * percent_resize / 100)
    resize_width = int(w * percent_resize / 100)
    resized_image = cv2.resize(img, (resize_width, resize_height), cv2.INTER_AREA)
    return resized_image


def set_binary_color(img, threshold_value, setting_color):
    _, binary_color_img = cv2.threshold(img, threshold_value, setting_color,
                                  cv2.THRESH_BINARY_INV)
    return binary_color_img


def find_shapes_by_pattern(base_image, template_image, ORIGINAL):
    w, h = template_image.shape[::-1]
    res = cv2.matchTemplate(base_image, template_image, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    rectangle_coords = []
    unique_coord_min = [0, 0, 0, 0]
    unique_coord_max = [0, 0, 0, 0]
    for pt in zip(*loc[::-1]):
        pt_1_min = unique_coord_min[0]
        pt_1_h_min = unique_coord_min[1]
        pt_0_min = unique_coord_min[2]
        pt_0_w_min = unique_coord_min[3]

        pt_1_max = unique_coord_max[0]
        pt_1_h_max = unique_coord_max[1]
        pt_0_max = unique_coord_max[2]
        pt_0_w_max = unique_coord_max[3]

        if not (pt_1_min < pt[1] < pt_1_max
        or pt_1_h_min < pt[1] + h < pt_1_h_max
        or pt_0_min < pt[0] < pt_0_max
        or pt_0_w_min < pt[0] + w < pt_0_w_max):
            unique_coord_min = [pt[1] - 20, pt[1] + h - 20, pt[0] - 20, pt[0] + w - 20]
            unique_coord_max = [pt[1] + 20, pt[1] + h + 20, pt[0] + 20, pt[0] + w + 20]
            rectangle_coords.append([pt[1], pt[1] + h, pt[0], pt[0] + w])

    coord_point = []
    for coord in rectangle_coords:
        pt = [coord[2], coord[0]]
        crop_image = base_image[pt[1]: pt[1] + h, pt[0]: pt[0] + w]
        crop_original_image = ORIGINAL[pt[1]: pt[1] + h, pt[0]: pt[0] + w]
        _, x, y = find_contours(crop_image, crop_original_image)
        cv2.circle(ORIGINAL, (x + pt[0], y + pt[1]), 1, (0, 255, 255), 3)
        # ORIGINAL = cv2.rectangle(ORIGINAL, pt, (pt[0] + w, pt[1] + h), (255, 228, 0), 5)
        coord_point.append((x + pt[0], y + pt[1]))
    return ORIGINAL, coord_point


def find_contours(img, crop_original_image):
    img = cv2.medianBlur(img, 11)
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(img,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=20, maxRadius=100)
    a, b = 0, 0
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(crop_original_image, (a, b), r, (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(crop_original_image, (a, b), 1, (0, 0, 255), 3)
            # cv2.imshow("Detected Circle", crop_original_image)
            # cv2.waitKey(0)
    return img, a, b


def find_nearest_points(x, y, coord_points):
    nearest_points = []
    for point in coord_points:
        distance = ((x - point[0])**2 + (y - point[1])**2)**0.5
        real_point = get_real_point(point)
        nearest_points.append([point, distance, real_point])
    nearest_points.sort(key=lambda i: i[1])
    print(nearest_points)
    return nearest_points


def get_real_point(point):
    return None

def calc_point_coord(x, y, nearest_points):
    pass


if __name__ == '__main__':
    ORIGINAL = read_image_from_path(full_path, 'COLOR')
    img = read_image_from_path(full_path, 'GRAY')
    img = set_binary_color(img, 60, 255)
    # img = resize_img(img, 30)
    # cv2.imshow('gray_pic', img)
    # cv2.waitKey()
    template_orr = read_image_from_path(template_path, 'COLOR')
    template_img = read_image_from_path(template_path, 'GRAY')
    template_img = set_binary_color(template_img, 75, 255)
    # template_img = resize_img(template_img, 85)
    # cv2.imshow('gray_pic', template_img)
    # cv2.waitKey()

    n_img, coord_points = find_shapes_by_pattern(img, template_img, ORIGINAL)
    print(coord_points)
    # n_img = find_contours(template_img)
    n_img = resize_img(n_img, 30)
    # cv2.imshow('gray_pic', n_img)
    # cv2.waitKey()


    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            find_nearest_points(x, y, coord_points)
            xy = "%d,%d" % (x, y)
            cv2.circle(n_img, (x, y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(n_img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0,0,0), thickness = 1)
            cv2.imshow("image", n_img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    while(1):
        cv2.imshow("image", n_img)
        if cv2.waitKey(0)&0xFF==27:
            break
    cv2.destroyAllWindows()

# img = cv2.imread(full)
# img_rgb = img.copy()
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread(template_im, 0)
# w, h = template.shape[::-1]
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.55
#
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     # print(pt)
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
# print('eagf')
# cv2.imshow('contours', img_rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()

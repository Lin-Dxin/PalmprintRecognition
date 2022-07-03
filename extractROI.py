import math
import cv2
import numpy as np

PI = 3.14159265
debug = 1

img_path = 'Data/TestingSet/new_valdataset/false1/first_img.jpg'
src = cv2.imread(img_path, 0)
cv2.imshow("src",src)
if src is None:
    print('图片没读到')
    exit(0)
if debug:
    cv2.imshow("src", src)

# 2.1 crop image
crop_img = src[0:284, 40:200]
if debug:
    cv2.imshow("crop_img", crop_img)

# 2.2 low-pass Gaussion Filter
before_Gaussion = np.zeros((284, 160), dtype=np.int64)
roi = crop_img[7:crop_img.shape[0] - 7, 7:crop_img.shape[1] - 7]
before_Gaussion[7:crop_img.shape[0] - 7, 7:crop_img.shape[1] - 7] = roi.copy()
Gaussion_img = cv2.GaussianBlur(np.uint8(before_Gaussion), (15, 15), 2, 2)
if debug:
    cv2.imshow("Gaussion_img", Gaussion_img)

# 2.3 thresholding
ret, Binary_img = cv2.threshold(Gaussion_img, 20, 1, 0)
if debug:
    Binary_img_for_show = cv2.normalize(Binary_img, None, 0, 255, 32)
    cv2.imshow("Binary_img", Binary_img_for_show)

# 2.4 Find Reference Points
# (a)Find External Reference Point
Out_top = (0, 0)
Out_bottom = (0, 0)
for row in range(Binary_img.shape[0]):
    is_get = 0
    for col in range(Binary_img.shape[1]):
        if Binary_img[row][col] == 1:
            Out_top = (col, row)
            is_get = 1
            break
    if is_get:
        break
for row in range(Binary_img.shape[0] - 1, -1, -1):
    is_get = 0
    for col in range(Binary_img.shape[1]):
        if Binary_img[row][col] == 1:
            Out_bottom = (col, row)
            is_get = 1
            break
    if is_get:
        break
if debug:
    print("Out_top(x,y):{}".format(Out_top))
    print("Out_bottom(x,y):{}".format(Out_bottom))

# (b)Find Internal Reference Point
In_top = (0, 0)
In_bottom = (0, 0)
gap_x = 0
for col in range(Binary_img.shape[1]):
    gap_width = 0
    for row in range(Binary_img.shape[0]):
        if Binary_img[row][col] == 0:
            gap_width += 1
    if gap_width < 200:
        gap_x = col
        break
In_top = (gap_x, 0)
In_bottom = (gap_x, 0)
center_y = Binary_img.shape[0] // 2
for row in range(center_y, -1, -1):
    if Binary_img[row][gap_x] == 1:
        In_top = (gap_x, row)
        break
for row in range(center_y, Binary_img.shape[0]):
    if Binary_img[row][gap_x] == 1:
        In_bottom = (gap_x, row)
        break
if debug:
    print('In_top(x,y):{}'.format(In_top))
    print('In_bottom(x,y):{}'.format(In_bottom))

# 2.5.1 Find Countours
Out_top_j = Out_bottom_j = In_top_j = In_bottom_j = 0
reference_point_num = 0
contours, hierarchy = cv2.findContours(Binary_img, 0, 1)
Contours = np.zeros(Binary_img.shape, np.int64)
for j in range(len(contours[0])):
    if contours[0][j][0][0] == Out_top[0] and contours[0][j][0][1] == Out_top[1]:
        Out_top_j = j
        reference_point_num += 1
    if contours[0][j][0][0] == Out_bottom[0] and contours[0][j][0][1] == Out_bottom[1]:
        Out_bottom_j = j
        reference_point_num += 1
    if contours[0][j][0][0] == In_top[0] and contours[0][j][0][1] == In_top[1]:
        In_top_j = j
        reference_point_num += 1
    if contours[0][j][0][0] == In_bottom[0] and contours[0][j][0][1] == In_bottom[1]:
        In_bottom_j = j
        reference_point_num += 1
if reference_point_num != 4:
    print('not four')
    exit(0)
for j in range(Out_top_j, In_top_j + 1):
    P = (contours[0][j][0][0], contours[0][j][0][1])
    Contours[P[1]][P[0]] = 255
for j in range(In_bottom_j, Out_bottom_j + 1):
    P = (contours[0][j][0][0], contours[0][j][0][1])
    Contours[P[1]][P[0]] = 255

# 2.5.2 Key Point Positioning
Top_x = Bottom_x = 0.0
Top_y_vector = []
Bottom_y_vector = []
for j in range(Out_top_j, In_top_j + 1):
    if contours[0][j][0][0] > Top_x:
        Top_x = contours[0][j][0][0]
for j in range(In_bottom_j, Out_bottom_j + 1):
    if contours[0][j][0][0] > Bottom_x:
        Bottom_x = contours[0][j][0][0]
for j in range(Out_top_j, In_top_j + 1):
    if contours[0][j][0][0] == Top_x:
        Top_y_vector.append(contours[0][j][0][1])
for j in range(In_bottom_j, Out_bottom_j + 1):
    if contours[0][j][0][0] == Bottom_x:
        Bottom_y_vector.append(contours[0][j][0][1])

top_sum = sum(Top_y_vector)
bottom_sum = sum(Bottom_y_vector)
Top_y = top_sum / float(len(Top_y_vector))
Bottom_y = bottom_sum / float(len(Bottom_y_vector))

print('Top:({},{})'.format(Top_x, Top_y))
print('Bottom:({},{})'.format(Bottom_x, Bottom_y))

# 2.6 Build a Coordinate System on the Oridinal Image
Top = (Top_x + 40, Top_y)
Bottom = (Bottom_x + 40, Bottom_y)
Origin_X = (Top[0] + Bottom[0]) / 2.0
Origin_Y = (Top[1] + Bottom[1]) / 2.0
Origin = (Origin_X, Origin_Y)
Slope_y_axis = (Top_y - Bottom_y) / (Top_x - Bottom_x)
Slope_x_axis = -1 / Slope_y_axis

angle = -1 * math.atan(1 / Slope_y_axis) * (180 / PI)
rotated_sz = (src.shape[1], src.shape[0])
center = (Origin_X, Origin_Y)
rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
Rotated_img = cv2.warpAffine(src, rot_mat, rotated_sz, 1, 0)
dst = Rotated_img.copy()
Uleft = (int(Origin_X + 50), int(Origin_Y - 128 / 2))
dst = dst[Uleft[1]:Uleft[1] + 128, Uleft[0]:Uleft[0] + 128]
if debug:
    cv2.imshow("dst", dst)

if debug:
    cv2.waitKey(0)
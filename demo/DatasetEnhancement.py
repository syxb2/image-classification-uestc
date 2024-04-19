# 本项目主要用于提交和展示你们完善的数据集增强代码
# 请把各个效果（翻转、旋转、位移等）操作效果都展示一遍）

# ln[1]
# 相关库导入
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as save
import random


# ln[2]
# 任务1：完成矩阵乘法代码，
# 注意，此处不使用Numpy的矩阵乘法库
def OneMultplt(A, B, i=0, j=0):
    out = 0
    for n in range(len(A[0])):
        out += A[i][n] * B[n][j]
    return out


def DotMatrix(A, B):
    if len(A[0]) == len(B):
        res = [[0] * len(B[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                res[i][j] = OneMultplt(A, B, i, j)
        return res

    else:
        return "输入矩阵有误！"


# ln[3]
# 任务2：根据课件完善不同操作的transform矩阵
class Img:
    def __init__(self, image, rows, cols, center=[0, 0]):
        self.src = image
        self.rows = rows
        self.cols = cols
        self.center = center

    def Move(self, delta_x, delta_y):  # 平移
        self.transform = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])
        img.Process()
        img2 = Image.fromarray(img.dst)
        plt.imshow(img2)
        plt.show()

    def Zoom(self, factor):  # 缩放
        self.transform = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])
        img.Process()
        img2 = Image.fromarray(img.dst)
        plt.imshow(img2)
        plt.show()

    def Horizontal(self):  # 水平镜像
        #'''水平镜像
        # 镜像的这两个函数，因为原始图像读进来后是height×width×3,和我们本身思路width×height×3相反
        # 所以造成了此处水平镜像和垂直镜像的tranform做了对调，同学们可以考虑如何把它们调换回来？'''
        self.transform = np.array([[1, 0, 0], [0, -1, self.cols - 1], [0, 0, 1]])
        img.Process()
        img2 = Image.fromarray(img.dst)
        plt.imshow(img2)
        plt.show()

    # 垂直镜像，transform用的是PPT上所述的水平镜像的transfrom，理由如上所述。
    def Vertically(self):  # 垂直镜像
        self.transform = np.array([[-1, 0, self.rows - 1], [0, 1, 0], [0, 0, 1]])
        img.Process()
        img2 = Image.fromarray(img.dst)
        plt.imshow(img2)
        plt.show()

    def Rotate(self, beta):  # 旋转
        self.transform = np.array(
            [
                [math.cos(beta), -math.sin(beta), 0],
                [math.sin(beta), math.cos(beta), 0],
                [0, 0, 1],
            ]
        )
        img.Process()
        img2 = Image.fromarray(img.dst)
        plt.imshow(img2)
        plt.show()

    def Process(self):
        # 初始化定义目标图像，具有3通道RBG值，一定要注意dst和src的通道值是否对应.
        self.dst = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)

        # 提供for循环，遍历图像中的每个像素点，然后使用矩阵乘法，找到变换后的坐标位置
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([[i - self.center[0]], [j - self.center[1]], [1]])
                [[x], [y], [z]] = DotMatrix(self.transform, src_pos)
                x = int(x) + self.center[0]
                y = int(y) + self.center[1]
                if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                    self.dst[i][j] = 255
                else:

                    self.dst[i][j] = self.src[x][y]


# ln[4]
# 上面的定义完成后，利用此处开始的主程序代码测试图像操作效果
infer_path = r"1.jpeg"
imgv = Image.open(infer_path)
plt.imshow(imgv)
plt.show()


imgv = np.array(imgv)
print("My pic shape is %d", imgv.shape)
rows = len(imgv)
cols = len(imgv[0])
img = Img(imgv, rows, cols, [0, 0])

a = random.randint(0, int(rows / 2))
b = random.randint(0, int(cols / 2))
img.Move(a, b)

c = random.random()
img.Zoom(c)

img.Vertically()

img.Horizontal()

d = random.uniform(0, 45)
img.Rotate(math.radians(d))

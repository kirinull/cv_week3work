import cv2
from matplotlib import pyplot as plt
import numpy as np

# 计算互相关
def CCORR(img, temp, normalize=True):
    w, h = temp.shape[::-1]
    W, H = img.shape[::-1]
    img = np.array(img, dtype='float')
    temp = np.array(temp, dtype='float')
    res = np.zeros((W - w + 1, H - h + 1))
    
    # 利用循环计算互相关的值
    for i in range(W - w + 1):
        for j in range(H - h + 1):
            res[i, j] = np.sum(temp * img[j:j + h, i:i + w])
            if normalize:
                res[i, j] /= (np.sqrt(np.sum(temp ** 2)) * np.sqrt(np.sum(img[j:j + h, i:i + w] ** 2))) 
    return res
    

# 构建单目标匹配类
class temp_match_single():
    def __init__(self, img, temp):
        self.img = img
        self.temp = temp
        
    def match(self):
        # 输入目标图像
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 输入模板图像
        temp = cv2.cvtColor(self.temp, cv2.COLOR_BGR2GRAY)
        w, h = temp.shape[::-1]
        
        # 计算互相关
        res = CCORR(img, temp)
        
        # 找到互相关值最大的位置
        max_val = np.max(res)
        loc = np.where(res == max_val)

        # 从数组中提取单个元素（使用[0]索引或.item()方法）
        top_left = [int(loc[0][0]), int(loc[1][0])]  # 推荐这种方式
        # 或者
        # top_left = [loc[0].item(), loc[1].item()]

        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # 将其框出
        cv2.rectangle(self.img, top_left, bottom_right, (255,255,255), 1)
        
        # 显示结果
        cv2.imshow('Target Found', self.img[:, :, ::-1])
        

img = cv2.imread('lena_tri.jpg')
template = cv2.imread('lena_small.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
test = temp_match_single(img_rgb, template_rgb)
test.match()

cv2.waitKey(0)
cv2.destroyAllWindows()
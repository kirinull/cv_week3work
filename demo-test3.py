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
        scales = np.linspace(0.5, 1.5, 10)  # 例如从0.5倍到1.5倍
        best_score = -np.inf
        best_loc = None
        best_scale = 1.0

        for scale in scales:
            resized_template = cv2.resize(temp, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            res = CCORR(img, resized_template)
            max_val = np.max(res)
            loc = np.where(res == max_val)
            if max_val > best_score:
                best_score = max_val
                best_loc = loc
                best_scale = scale
        print(f"Best Scale: {best_scale}, Best Score: {best_score}")
        # # 计算互相关
        # res = CCORR(img, resized_template)
        
        # # 找到互相关值最大的位置
        # max_val = np.max(res)
        # loc = np.where(res == max_val)

        # 从数组中提取单个元素（使用[0]索引或.item()方法）
        top_left = [int(best_loc[0][0]), int(best_loc[1][0])]  # 推荐这种方式
        # 或者
        # top_left = [loc[0].item(), loc[1].item()]

        bottom_right = (top_left[0] + (int)(w*best_scale), top_left[1] + (int)(h*best_scale))
        
        # 将其框出
        cv2.rectangle(self.img, top_left, bottom_right, (255,255,255), 1)
        
        # 显示结果
        img_rgb = cv2.cvtColor(self.img[:, :, ::-1], cv2.COLOR_BGR2RGB)
        cv2.imshow('Target Found', img_rgb)
        

img = cv2.imread('lena_tri.jpg')
# img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
template = cv2.imread('lena_smalllarge.png')
# template_bgr = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)


test = temp_match_single(img, template)
test.match()

cv2.waitKey(0)
cv2.destroyAllWindows()
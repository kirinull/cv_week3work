import numpy as np
import cv2

def main():
    # 读取图像并转换为灰度图
    gray = cv2.imread("lena_tri.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 显示原始图像
    cv2.imshow("Original Image", gray)

    # 处理图像
    size = 9
    k = np.ones((size, size), np.float32) / (size * size)
    img_aver = cv2.filter2D(gray, -1, k)

    # 显示处理后图像
    cv2.imshow("Average Image", img_aver)

    # 保存处理后图像
    cv2.imwrite("lenaface_aver.jpg", img_aver)

    cv2.waitKey(0)    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# import numpy as np
# import cv2

# def template_match_demo():
#     # 读取原图和模板图像
#     img = cv2.imread("lena_tri.jpg", cv2.IMREAD_GRAYSCALE)
#     template = cv2.imread("lena_small.png", cv2.IMREAD_GRAYSCALE)
#     w, h = template.shape[::-1]

#     # 模板匹配
#     res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#     # 绘制匹配结果
#     top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

#     # 显示结果
#     cv2.imshow("Template Matching Result", img_color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     template_match_demo()
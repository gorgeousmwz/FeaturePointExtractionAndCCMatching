import cv2
import numpy as np
from matplotlib import pyplot as plt
from Moravec import moravec
from Forstner import forster

#图片读取
def readImage(imgPath):
    '''
        读取图片,并返回图像数组\n
        Parameters:
            imgPath:图片路径
    '''
    img=cv2.imread(imgPath)
    cv2.imshow('original image',img)
    cv2.waitKey(0)
    return img

def drawFeaturePoint(img,FP_map):
        '''
            根据特征点在原图上标注
            Parameters:
                img:输入图像
                FP_map:特征点图
        '''
        h=img.shape[0]
        w=img.shape[1]
        for r in range(h):
            for c in range(w):
                if FP_map[r,c]==255:#如果是特征点
                    cv2.circle(img, (c,r), 1, [0, 0, 255], 2, cv2.LINE_AA)#标注
        return img



if __name__=='__main__':
    while True:
        image=readImage('data\\panLeft.bmp')
        print('--------------------------------------------点特征提取--------------------------------------------')
        print('0.退出程序')
        print('1.Moravec算法')
        print('2.Forstner算法')
        decision=input('请输入您想要使用的方法:(输入序号)')
        if int(decision)==0:
            print('程序退出成功')
            break
        elif int(decision)==1:
            moravec_test=moravec(image,T=1200)#6000
            FP_map=moravec_test.execute()
            print('特征点数目:{}'.format(np.count_nonzero(FP_map)))
            result=drawFeaturePoint(image,FP_map)
            cv2.imwrite('Result\\PointFeatureExtraction\\Moravec\\tenniscourt\\result.jpg',result)
        elif int(decision)==2:
            forstner_test=forster(image)
            FP_map=forstner_test.execute()
            result=drawFeaturePoint(image,FP_map)
            cv2.imwrite('Result\\PointFeatureExtraction\\Forstner\\tenniscourt\\result.jpg',result)
        else:
            print('请输入正确的序号')

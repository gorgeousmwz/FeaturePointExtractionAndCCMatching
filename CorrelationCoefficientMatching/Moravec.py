import cv2
import numpy as np
from matplotlib import pyplot as plt

class moravec:
    '''
        Moravec点特征提取算法
    '''
    def __init__(self,img,window_size=5,T=-1,noMax_size=25):
        '''
            Parameters:
                img:输入彩色图像\n
                window_size:计算兴趣值窗口大小(default:5)\n
                T:确定候选点的兴趣值阈值(default:-1,即取IV非零均值为阈值)\n
                noMax_size:非极大值抑制窗口大小(default:25)\n
        '''
        self.img=img.astype(int)
        self.h=self.img.shape[0]
        self.w=self.img.shape[1]
        self.win_size=window_size
        self.T=T
        self.noMax_size=noMax_size
        self.IV_map=np.zeros([self.h, self.w], np.int32)#兴趣值图
        self.CP_map=np.zeros([self.h, self.w], np.int32)#候选点图
        self.FP_map=np.zeros([self.h, self.w], np.int32)#特征点图

    def getInterestValue(self):
        '''
            计算每个像素点的兴趣值
        '''
        k=int(self.win_size/2)
        for r in range(k,self.h-k):#遍历像素
            for c in range(k,self.w-k):#去除外围像素
                V1=V2=V3=V4=0
                for i in range(-k,k):#遍历窗口
                    #计算四个方向相邻像素灰度差的平方和
                    V1=V1+(self.img[r+i,c]-self.img[r+i+1,c])**2#纵向
                    V2=V2+(self.img[r+i,c+i]-self.img[r+i+1,c+i+1])**2#正对角线
                    V3=V3+(self.img[r,c+i]-self.img[r,c+i+1])**2#横向
                    V4=V4+(self.img[r+i,c-i]-self.img[r+i+1,c-i-1])**2#反对角线
                IV=min(V1,V2,V3,V4)#取最小值为该点的兴趣值
                self.IV_map[r,c]=IV#记录各点的兴趣值
        cv2.imwrite('Result\\PointFeatureExtraction\\Moravec\\tenniscourt\\IV_map.jpg',self.IV_map)

    def getCondidatePoint(self):
        '''
            根据阈值筛选得到候选点,如果T=-1则默认将非零均值作为阈值
        '''
        T=self.T
        if self.T==-1:
            T=np.sum(self.IV_map)/np.count_nonzero(self.IV_map)
        self.CP_map=np.where(self.IV_map<T,0,self.IV_map)#兴趣值大于阈值的点作为候选点
        cv2.imwrite('Result\\PointFeatureExtraction\\Moravec\\tenniscourt\\CP_map.jpg',self.CP_map)

    def getFeaturePoint(self):
        '''
            在候选点中选取极值点作为特征点
        '''
        k=int(self.noMax_size/2)
        for r in range(k,self.h-k,self.noMax_size):#按抑制窗口覆盖图片进行筛选
            for c in range(k,self.w-k,self.noMax_size):
                a1=r-k if r-k>0 else 0
                a2=r+k if r+k<self.h else self.h
                b1=c-k if c-k>0 else 0
                b2=c+k if c+k<self.w else self.w
                area=self.CP_map[a1:a2,b1:b2]#抑制区域
                fp_positions=np.where(area==np.max(area))#每个区域内最大值为特征点
                for i in range(len(fp_positions[0])):
                    if self.CP_map[fp_positions[0][i]+a1,fp_positions[1][i]+b1]!=0:
                        self.FP_map[fp_positions[0][i]+a1,fp_positions[1][i]+b1]=255#在特征图中标注特征点
        cv2.imwrite('Result\\PointFeatureExtraction\\Moravec\\tenniscourt\\FP_map.jpg',self.FP_map)

    def execute(self):
        '''
            执行Moravec算法,返回标注图像
        '''
        self.getInterestValue()#计算兴趣值
        self.getCondidatePoint()#得到候选点
        self.getFeaturePoint()#得到特征点
        return self.FP_map


        


        


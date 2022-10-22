import cv2
import numpy as np
from matplotlib import pyplot as plt
from Moravec import moravec

class forster:
    '''
        Forster点特征提取算子
    '''
    def __init__(self,img,window_size=5,noMax_size=7,threshold_for_preselect=20,threshold_for_q=0.5,f=0.5,c=5):
        '''
            Parameters:
                img:输入彩色图像\n
                window_size:计算协方差矩阵窗口大小(default:5)\n
                noMax_size:非极大值抑制窗口大小(default:7)\n
                threshold_for_preselect:进行最初筛选的差分阈值\n
                threshold_for_q:q值阈值(default:0.5)\n
                f、c:确定w权值阈值的系数(default:f=0.5,c=5)\n
        '''
        self.img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(int)
        self.h=self.img.shape[0]
        self.w=self.img.shape[1]
        self.win_size=window_size
        self.noMax_size=noMax_size
        self.Tpre=threshold_for_preselect
        self.Tq=threshold_for_q
        self.f=f
        self.c=c
        self.PR_map=np.zeros([self.h, self.w], np.int32)#初选点标志图
        self.q_map=np.zeros([self.h, self.w], np.float)#q值图
        self.w_map=np.zeros([self.h, self.w], np.float)#w值图
        self.CP_map=np.zeros([self.h, self.w], np.int32)#候选点标志图
        self.FP_map=np.zeros([self.h, self.w], np.int32)#特征点标志图

    def getPrimaryPoint(self):
        '''
            利用像素点的差分算子提取出初选点（提高计算速度）
            差分算子中位数大于阈值则为初选点
        '''
        k=int(self.win_size/2)
        for r in range(k,self.h-k):#遍历像素
            for c in range(k,self.w-k):#去除外围像素
                #差分算子
                dg1=dg2=dg3=dg4=0              
                dg1=abs(self.img[r,c]-self.img[r+1,c])
                dg2=abs(self.img[r,c]-self.img[r,c+1])
                dg3=abs(self.img[r,c]-self.img[r-1,c])
                dg4=abs(self.img[r,c]-self.img[r,c-1])
                a=np.median([dg1,dg2,dg3,dg4])
                if np.median([dg1,dg2,dg3,dg4])>=self.Tpre:#标记初选点
                    self.PR_map[r,c]=255
    
    def getGraycovarianceMatrix(self,r,c,k):
        '''
            计算灰度协方差矩阵\n
            Parameters:
                r,c:像素点行列号\n
                k:窗口半径\n
        '''
        gu2=guv=gv2=0
        N=np.zeros((2,2),dtype=np.float)
        for i in range(-k,k):#遍历窗口
            for j in range(-k,k):
                #计算灰度协方差矩阵
                gu2+=(self.img[r+i+1,c+j+1]-self.img[r+i,c+j])**2
                gv2+=(self.img[r+i+1,c+j]-self.img[r+i,c+j+1])**2
                guv+=(self.img[r+i+1,c+j+1]-self.img[r+i,c+j])*(self.img[r+i+1,c+j]-self.img[r+i,c+j+1])
        N[0,0]=gu2
        N[0,1]=guv
        N[1,0]=guv
        N[1,1]=gv2
        return N

    def getQandW(self):
        '''
            根据每个像素窗口内的灰度协方差矩阵计算兴趣值q、w
        '''
        k=int(self.win_size/2)
        for r in range(k,self.h-k):#遍历像素
            for c in range(k,self.w-k):#去除外围像素
                N=self.getGraycovarianceMatrix(r,c,k)#获取灰度协方差矩阵
                #计算q和w
                q=w=0
                if np.trace(N)!=0:
                    q = 4*np.linalg.det(N)/((np.trace(N))**2)
                    w = np.linalg.det(N)/np.trace(N)
                self.q_map[r,c]=q
                self.w_map[r,c]=w

    def getCondidatePoint(self):
        '''
            根据q、w和阈值确定待选点
        '''
        k=int(self.win_size/2)
        for r in range(k,self.h-k):#遍历像素
            for c in range(k,self.w-k):#去除外围像素
                if self.PR_map[r,c]!=0:#在初选点中寻找
                    if self.q_map[r,c]>=self.Tq and \
                    self.w_map[r,c]>=self.f*np.mean(self.w_map) and \
                    self.w_map[r,c]>=self.c*np.median(self.w_map):
                        self.CP_map[r,c]=255#标志候选点

    def getFeaturePoint(self):
        '''
            将候选点经过非最大值抑制之后得到特征点
        '''
        k=int(self.noMax_size/2)
        for r in range(k,self.h-k,self.noMax_size):#按抑制窗口覆盖图片进行筛选
            for c in range(k,self.w-k,self.noMax_size):
                a1=r-k if r-k>0 else 0
                a2=r+k if r+k<self.h else self.h
                b1=c-k if c-k>0 else 0
                b2=c+k if c+k<self.w else self.w
                area=self.w_map[a1:a2,b1:b2]#抑制区域
                fp_positions=np.where(area==np.max(area))#每个区域内最大值为特征点
                for i in range(len(fp_positions[0])):
                    if self.CP_map[fp_positions[0][i]+a1,fp_positions[1][i]+b1]!=0:
                        self.FP_map[fp_positions[0][i]+a1,fp_positions[1][i]+b1]=255#在特征图中标注特征点

    def execute(self):
        '''
            执行Forstner算法,返回标注图像
        '''
        self.getPrimaryPoint()#筛选初选点
        self.getQandW()#计算兴趣值
        self.getCondidatePoint()#得到候选点
        self.getFeaturePoint()#得到特征点
        return self.FP_map


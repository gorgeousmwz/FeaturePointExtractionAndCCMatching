import cv2
import numpy as np
from Moravec import moravec
from math import sqrt

#图片读取
def readImage(imgPath,showName):
    '''
        读取图片,并返回图像数组\n
        Parameters:
            imgPath:图片路径
            showName:展示名称
        Return:
            img:读取图像
            h:图像高度
            w:图像宽度
    '''
    img=cv2.imread(imgPath,0)
    h=img.shape[0]
    w=img.shape[1]
    cv2.imshow(showName,img)
    cv2.waitKey(0)
    return img,h,w

def drawFeaturePoint(img,FP_map):
        '''
            根据特征点在原图上标注\n
            Parameters:
                img:输入图像
                FP_map:特征点图
            Return:
                img:标注后图像
        '''
        h=img.shape[0]
        w=img.shape[1]
        for r in range(h):
            for c in range(w):
                if FP_map[r,c]==255:#如果是特征点
                    cv2.circle(img, (c,r), 1, [0, 0, 255], 2, cv2.LINE_AA)#标注
        return img

def getCC(leftImageWin,rightImageWin):
    '''
        计算得到两个窗口间图像的相关系数\n
        Parameters:
            leftImageWin:左图计算窗口
            rightImageWin:右图计算窗口
        Return:
            CC:两个计算窗口的相关系数
    '''
    if leftImageWin.shape!=rightImageWin.shape:#如果左右窗口大小不相等
        return 0.0#相关系数为0
    #窗口大小
    h,c=leftImageWin.shape[0],leftImageWin.shape[1]
    #计算窗口灰度平均值
    lg_avg=np.average(leftImageWin)
    rg_avg=np.average(rightImageWin)
    #计算相关系数累加量
    sum1=sum2=sum3=0
    for i in range(h):
        for j in range(c):#遍历窗口
            sum1+=(leftImageWin[i,j]-lg_avg)*(rightImageWin[i,j]-rg_avg)
            sum2+=(leftImageWin[i,j]-lg_avg)**2
            sum3+=(rightImageWin[i,j]-rg_avg)**2
    #计算相关系数
    CC=sum1/(sqrt(sum2*sum3))
    return CC

def CCMatching(leftImage,rightImage,FP_map,winSize=7,winSearch=21,threshold=0.75):
    '''
        相关系数匹配\n
        Parameters:
            leftImage:左图
            rightImage:右图
            FP_map:左图像特征点图\n
            winSize:计算相关系数窗口大小(default:7)\n
            winSearch:搜索区大小(default:21)
            threshold:判断是否为同名点的阈值(default:0.75)
        Return:
            FP:特征点位置数组([x1,x2,x3...],[y1,y2,y3...])
            MP:与特征点对应匹配点位置数组[[x1,y1],[x2,y2],[x3,y3]...]
    '''
    leftImage=leftImage.astype(int)
    rightImage=rightImage.astype(int)
    k=int(winSize/2)
    s=int(winSearch/2)
    #左右图像坐标偏差
    dx=176
    dy=8
    #图像高宽
    l_h,l_w=leftImage.shape
    r_h,r_w=rightImage.shape
    #得到特征点的坐标
    FP=np.nonzero(FP_map)
    #匹配点图
    MP=np.tile(np.array([-1,-1]),(len(FP[0]),1))
    for p in range(len(FP[0])):
        #左图特征点坐标
        rl=FP[0][p]
        cl=FP[1][p]
        #右图特征点坐标
        rr=rl-dy
        cr=cl-dx
        #确定搜索区(防止越界)
        a1=rr-s if rr-s>=k else k
        a2=rr+s if rr+s<=r_h-k else r_h-k
        b1=cr-s if cr-s>=k else k
        b2=cr+s if cr+s<=r_w-k else r_w-k
        leftImageWin=leftImage[rl-k:rl+k,cl-k:cl+k]#左图计算窗口
        CC_map=np.zeros((winSearch,winSearch),np.float32)#搜索区内的相关系数图
        #遍历搜索区
        for i in range(a1,a2):
            for j in range(b1,b2):
                rightImageWin=rightImage[i-k:i+k,j-k:j+k]#右图计算窗口
                CC=getCC(leftImageWin,rightImageWin)#计算相关系数
                CC_map[i-a1,j-b1]=CC
        if np.max(CC_map)>=threshold:#如果搜索区最大相关系数大于阈值
            #得到搜索区相关系数最大的位置
            pos=np.where(CC_map==np.max(CC_map))
            #记录特征点的相应匹配点
            MP[p,0]=pos[0][0]+a1
            MP[p,1]=pos[1][0]+b1
    return FP,MP

def drawMatchLine(leftImage,rightImage,FP,MP,l_h,l_w,r_h,r_w):
    '''
        将匹配点之间连线\n
        Parameters:
            leftImage:左图
            rightImage:右图
            FP:特征点位置信息
            MP:匹配点匹配关系及位置
            l_h、l_w:左图大小
            r_h、r_w:右图大小
        Return:
            concatImage:标注好匹配关系的拼接图像
            dx:平均水平视差
            dy:平均上下视差
            ddx:水平视差方差
            ddy:上下视差方差
            num1:未经筛选的成功匹配点数
            num2:经过筛选后成功匹配点数
    '''     
    #图片拼接
    concatImage=np.zeros((l_h,l_w+r_w,3),np.uint8)
    concatImage[:,0:l_w]=leftImage
    concatImage[0:r_h,l_w:l_w+r_w]=rightImage
    #视差
    dx=0#左右视差
    dy=0#上下视差
    num1=0#匹配个数
    #匹配点连线
    for p in range(len(FP[0])):#遍历每个特征点
        #特征点位置
        fp_r=FP[0][p]
        fp_c=FP[1][p]
        #相应匹配点位置
        mp_r=MP[p,0]
        mp_c=MP[p,1]     
        if mp_r!=-1 and mp_c!=-1:#如果有匹配点
            #视差计算
            dx+=mp_c-fp_c
            dy+=mp_r-fp_r
            num1+=1
    #计算平均视差
    dx0=dx/num1
    dy0=dy/num1
    #匹配连线，计算筛选后平均视差和方差
    dx=dy=0
    ddx=ddy=0
    num2=0
    for p in range(len(FP[0])):#遍历每个特征点
        #特征点位置
        fp_r=FP[0][p]
        fp_c=FP[1][p]
        #相应匹配点位置
        mp_r=MP[p,0]
        mp_c=MP[p,1]     
        if mp_r!=-1 and mp_c!=-1:#如果有匹配点
            if abs(mp_c-fp_c-dx0)<=5 and abs(mp_r-fp_r-dy0)<=5:#视差与视差均值之差大于5像素的筛除
                #画点连线
                cv2.circle(concatImage, (fp_c,fp_r), 1, [0, 0, 255], 2, cv2.LINE_AA)#标注特征点
                cv2.circle(concatImage, (mp_c+l_w,mp_r), 1, [0, 0, 255], 2, cv2.LINE_AA)#标注匹配点
                cv2.line(concatImage,(fp_c,fp_r),(mp_c+l_w,mp_r),[0,255,0],1,8)#连线
                #视差计算
                dx+=mp_c-fp_c
                dy+=mp_r-fp_r
                #方差计算
                ddx+=(mp_c-fp_c-dx0)**2
                ddy+=(mp_r-fp_r-dy0)**2
                num2+=1
    #最终平均视差
    dx=dx/num2
    dy=dy/num2
    #最终方差计算
    ddx=ddx/(num2-1)
    ddy=ddy/(num2-1)
    return concatImage,dx,dy,ddx,ddy,num1,num2



if __name__=='__main__':

    print('------------------Read Image------------------')
    l_img,l_h,l_w=readImage('data\\panLeft.bmp','original left image')
    r_img,r_h,r_w=readImage('data\\panRight.bmp','original right img')
    cv2.waitKey(0)
    print('left image size:({},{})'.format(l_h,l_w))
    print('right image size:({},{})'.format(r_h,r_w))

    print('------------------Feature Point Extraction------------------')
    moravec_test=moravec(l_img,T=1000)
    l_FP=moravec_test.execute()#moravec算子获取特征点
    l_colorimg=cv2.imread('data\\panLeft.bmp')
    l_colorimg=drawFeaturePoint(l_colorimg,l_FP)#标注特征点
    cv2.imwrite('Result\\CorrectlationCoefficientMatching\\featurepoint.jpg',l_colorimg)
    print('特征点个数:{}'.format(np.count_nonzero(l_FP)))
    print('Feature Point Extraction Finish')

    print('------------------Correlation Coefficient Matching-----------------')
    FP,MP=CCMatching(l_img,r_img,l_FP,threshold=0.85)#相关系数匹配
    l_colorimg=cv2.imread('data\\panLeft.bmp')
    r_colorimg=cv2.imread('data\\panRight.bmp')
    result,dx,dy,ddx,ddy,num1,num2=drawMatchLine(l_colorimg,r_colorimg,FP,MP,l_h,l_w,r_h,r_w)#连线标注
    cv2.imwrite('Result\\CorrectlationCoefficientMatching\\result.jpg',result)
    print('为筛选匹配点个数:{}'.format(num1))
    print('筛选后匹配点个数:{}'.format(num2))
    print('左右平均视差:{}'.format(dx))
    print('上下平均视差:{}'.format(dy))
    print('左右视差方差:{}'.format(ddx))
    print('上下视差方差:{}'.format(ddy))
    print('Correlation Coefficient Matching Finish')

Project Name: 点特征提取及相关系数法匹配
Writer: 马文卓 2020302131249
Time: 2022-10-20

1.实习任务
本次实习的任务主要包含两大模块：点特征提取、相关系数法匹配
·点特征提取：利用Moravec或Forstner算法提取影像的特征点，并在影像上显示出来。同时研究兴趣阈值、抑制窗口对结果的影响，以及随机提取和均匀提取的实施分析
·相关系数法匹配：利用相关系数匹配算法，对两张影像进行匹配，得到同名点，并进行可视化。同时研究匹配窗口、相关系数阈值等对结果的影响

2.实验数据
本次实习当中所用到的数据均存放于data文件夹中。包含三张图片：
·panLeft.bmp：灰度影像，942x1023
·panRight.bmp：灰度影像，887x805
·chess.bmp：彩色影像，467x638，主要用于检测点特征提取算法的效果

3.编程环境
本次实习当中编程环境如下：
·编程语言：Python
·环境：Python 3.9.7
·第三方库：opencv、numpy、math（其余Moravec、Forstner、匹配算法均为自己编写）

4.文件组织结构
·点特征提取及相关系数匹配：项目总文件夹
    ·data：实验数据文件夹
    ·PointFeatureExtraction：点特征提取文件夹
        ·Moravec(py)：Moravec算法
        ·Forstner(py)：Forstner算法
        ·main(py)：点特征提取主程序
    ·CorrelationCoefficientMatching：相关系数匹配文件夹
        ·Moravec(py)：匹配算法中所用到的Moravec特征提取算法
        ·main(py)：相关系数法匹配的主程序
    ·Result：实验结果文件夹
        ·PointFeatureExtraction：点特征提取结果文件夹
            ·Moravec：Moravec算法结果文件夹
                ·chess：棋盘数据结果文件夹
                ·tenniscourt：网球场数据结果文件夹
            ·Forstner：Forstner算法结果文件夹
                ·chess：棋盘数据结果文件夹
                ·tenniscourt：网球场数据结果文件夹
        ·CorrelationCoefficientMatching：相关系数匹配结果文件夹
    ·实验报告(docx、pdf)：实验报告
    ·ReadMe(txt)：项目概述
    ·illustration：流程图文件夹
5.运行程序
如果需要执行程序，则在python（任意版本）环境下，安装opencv库即可运行
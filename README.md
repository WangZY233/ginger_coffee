## 项目说明
本项目是基于ginger机器人端到端全流程的强化学习真机部署环境
视频展示：【机器人咖啡师来啦！】 https://www.bilibili.com/video/BV1AT421Y7zC/?share_source=copy_web

### 环境依赖
    python == 3.8
    ros noetic
    open3d == 0.13.0
    numpy == 1.23.4
    torch == 1.9.1+cu111

### 文件说明

geometry3d.py  
    主要点云3d处理函数，包括获取包围盒、获取深度信息、点云坐标转换等

inference.py
    主要是强化学习模型推理文件

RLPlanning.py
    主要是ginger机器人运动文件，包括机器人运动的做咖啡全流程运动轨迹和关键点，强化学习控制机器人运动，也是主要运行的文件

PlanningTest.py
    用于真机测试

cup_loc.py
    获取按钮，大拇指，食指，杯子位置文件，将这些信息使用ros话题通信将信息发送到话题中


### 编译

```
# 方法1: 直接在本文件夹下编译
cd grasp_coffee
catkin_make

# 方法2: 创建新文件夹然后编译
mkdir ~/ginger_coffee_ws
cp -r ./src ~/ginger_coffee_ws/
cd ~/ginger_coffee_ws
catkin_make
```

### 运行

#### 运行做咖啡流程需要三步 
    1. 开启位置识别节点
        rosrun RLPlanningNode cup_loc.py

    2. 运行RLPlanning.py 
        rosrun RLPlanningNode RLPlanning.py

    3. 发送运动开始信号
        rosrun RLPlanningNode start.py

    PS:每一次运行的时候都需要重新开启位置识别节点

### 机器人真实部署环境

    咖啡机按钮和机器人的距离大约在48-52cm之间

    咖啡机需要尽量正对机器人

    关于左右方向：咖啡机需要相对于机器人需要偏左4-5cm,也就是机器人的中心点大概和左边按钮对齐之间的距离在2-6cm之间

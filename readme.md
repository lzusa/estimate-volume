## 文件结构

config.py // 配置文件

get_temp.py // 主程序入口，获取对应图像的匹配温度

estimate_volume.py // 计算体积

ellipse.py // 椭圆拟合方法

legendre.py // 勒让德拟合方法

edge_extend.py // 预处理相关文件

circle_acceptable.py // 预处理相关文件

background_interference.py // 预处理相关文件

specular_gaps.py // 预处理相关文件

## 目录结构

在默认的config的配置中，文件目录结构、文件名应为以下形式：

```
│  background_interference.py
│  circle_acceptable.py
│  config.py
│  edge_extend.py
│  ellipse.py
│  estimate_volume.py
│  get_temp.py
│  legendre.py
│  specular_gaps.py
│
└─ESL Data
    └─Cu30 //样品名称
        │  Cu30.csv //样品名称.csv
        │  Cu30_像素比与质量.txt //样品名称_像素比与质量.csv
        │  Image_20250425163432543.bmp //普通图像存放于此处
        │  Image_20250425163432562.bmp
        │  Image_20250425163432602.bmp
        │
        ├─Cu30_low_temp //样品名称_low_temp，存放存在缺损的图像
        │   Image_20250425163443118.bmp
        │   Image_20250425163443206.bmp
        │   Image_20250425163443420.bmp
        │
        └─room temperature //如需计算常温下的密度，常温图像放在此处
            Image_20250425163046637.bmp
            Image_20250425163046648.bmp
            Image_20250425163046655.bmp
```

例如对于样品Cu30：

在Cu30文件夹下：
存放样品对应的csv文件和像素比文件，其中像素比与质量.txt的格式应为：

```
像素比	0.00342094
质量 	0.12318
```
同时在这个路径下应存放待处理的图像，这个路径在config.py中可以通过ROOM_TEMP_DIR进行调整

在Cu30_low_temp文件夹下：
存放挑选出来的带有缺损的图像，请注意，**带有缺损的图像应同时存在与此文件夹与上一层文件夹中**

在room temperature文件夹下：
存放常温状态中的图像，这个路径在config.py中可以通过ROOM_TEMP_DIR进行调整

## 使用方法

直接运行get_temp.py即可
```
python get_temp.py
```


## 配置与配置文件

在config.py中，需重点关注以下参数：

ROOM_TEMP_TIME_MODE，若为True，将只处理常温图像，若为False，处理其他图像

LOW_TEMP，定义需要特殊处理无球冠图像的最高温度，低于这个值才会进行无球冠图像处理，若为空，不进行无球冠图像处理

VOLUME_METHOD，定义除了无球冠图像之外的其他图像的处理方法

LOW_TEMP_DEFECT_METHOD，定义无球冠图像的处理方法





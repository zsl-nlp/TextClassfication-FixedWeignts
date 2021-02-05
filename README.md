# 1.代码结构

	.
	├── README.md

	└── run.py


# 2.数据

	使用数据为清华文本分类数据集，点击http://thuctc.thunlp.org/ 下载。

	将下载的文件解压，根据需要放在指定文件夹下。

# 3.依赖库

	bert4keras==0.9.8

	h5py==2.10.0 
	 
	Keras==2.3.1
	 
	tensorflow-gpu==1.14.0
	 
	tqdm==4.54.1


# 4.运行

## 4.1 训练

	将run.py文件第17行修改为  do_train = True 

	执行 run run.py  开始训练

## 4.2 预测

	将run.py文件第17行修改为 do_train = False ，并根据预测语句修改第223行。

	执行 run run.py 开始训练


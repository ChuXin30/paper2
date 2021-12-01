# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_precision():
	x_cnn = [0.96, 0.50 ,0.75 ,0.78,0.76,0.44 , 0.56]
	x_text_cnn = [0.96, 0.24 , 0.50 , 0.66,  0.61,0.28 , 0.29]
	x_LSTM = [0.97,  0.31 , 0.64 , 0.36 ,   0.49, 0.18 ,  0.32]
	x_XGboost=[0.89, 0.22 , 0.60 , 0.19 ,    0.42, 0.20  ,  0.00 ]
	y = ['normal','adduser','ftp','java','meterpreter','ssh','webshell']
	plt.plot(y, x_cnn, 'ro-',label='CNN')
	plt.plot(y,x_text_cnn , 'g*:', ms=10,label='Text_CNN')
	plt.plot(y,x_LSTM , 'b^-', ls='--', ms=10,label='LSTM')
	plt.plot(y,x_XGboost , 'yx-', ms=10,label='XGboost')
	plt.legend(loc="lower left")
	plt.ylabel('precision')
	plt.show()

def draw_recall():
	x_cnn = [0.98,  0.37 ,0.77 ,0.85,0.55 ,0.60 , 0.35]
	x_text_cnn = [0.96, 0.17 , 0.70  , 0.80,  0.42 , 0.45 , 0.17]
	x_LSTM = [ 0.90 ,  0.31 ,  0.74  , 0.74  ,   0.53,  0.30 ,  0.35]
	x_XGboost=[0.99,  0.06 ,  0.06 , 0.05 ,     0.12 ,  0.10  ,  0.00 ]
	y = ['normal','adduser','ftp','java','meterpreter','ssh','webshell']
	plt.plot(y, x_cnn, 'ro-',label='CNN')
	plt.plot(y,x_text_cnn , 'g*:', ms=10,label='Text_CNN')
	plt.plot(y,x_LSTM , 'b^-', ls='--', ms=10,label='LSTM')
	plt.plot(y,x_XGboost , 'yx-', ms=10,label='XGboost')
	plt.legend(loc="lower left")
	#plt.title('recall')
	plt.ylabel('recall value')

	plt.show()

def draw_top_word():
	p = [0.9633,  0.96532 , 0.96372, 0.96650 ,0.95949, 0.96034 ]
	r = [0.98355, 0.99198 , 0.99394 , 0.99394 , 0.99299,  0.99091 ]
	f = [ 0.97332 ,  0.97847 , 0.97860 ,  0.98003, 0.97595  ,  0.97539 ]
	y = ['top1','top2','top3','top4','top5','top6']
	plt.plot(y, p, 'ro-',label='Precision')
	plt.plot(y,r , 'g*:', ms=10,label='Recall')
	plt.plot(y,f , 'b^-', ls='--', ms=10,label='F-measure')
#	plt.legend(loc="lower left")
	plt.legend(bbox_to_anchor=(0.95, 1.08), ncol=3, borderaxespad=0,fontsize='large')
	plt.ylim(0.94, 1.01)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	#plt.ylabel('f1-score')

	plt.show()

def draw_method_cmp():
	max_lst_of_all = {}  # 一个字典，value是四季最大阵风的风速值，key是年份
	max_lst_of_all['Precision'] = [0.97070,  0.99946, 0.94505, 0.96432,0.90975, 0.95904,0.96650]
	max_lst_of_all['Recall'] =    [0.63663,  0.77003, 0.98159, 0.98076, 0.99495, 0.97898,0.99394]
	max_lst_of_all['F-measure'] = [ 0.76895, 0.86987, 0.96297, 0.97247 ,0.95044, 0.96891,0.98003]


	fig = plt.figure()
	count = 1
	color = ['lightskyblue', 'lime', 'blue','red' ,  'white', 'white', 'white']  # 指定bar的颜色
	hatch = ["/", "X", "+", "*","o", "//", "."]
	labels = ['PCA', 'LogClustering', 'DeepLog', 'LogAnomaly','our method w/o CNN', 'our method w/o LSTM',
			  'our method']  # legend标签列表，上面的color即是颜色列表
	for key in max_lst_of_all.keys():
		print(max_lst_of_all[key])
		x = np.arange(count - 0.50, count + 0.50, 0.12)  # 一年有四季，此行指定四季对应的bar的位置，比如2010年：2009.7,2009.9,2010.1,2010.3
		y = max_lst_of_all[key]  # 此行决定了bar的高度(风速值）
		# bar_width = 0.2
		print(x)
		print(y)
		print(count)
		for x1, y1, c1 ,h1,l1 in zip(x, y, color , hatch,labels):  # 遍历以上三者，每一次生成一条bar
			if(count == 1):
				plt.bar(x1, y1, width=0.12, color=c1 , edgecolor='black', hatch=h1 , label = l1 )
			else:
				plt.bar(x1-0.001, y1, width=0.12, color=c1 , edgecolor='black', hatch=h1 )
			plt.text(x1-0.06,y1+0.001,"%.2f"%y1,fontsize=12)
		count += 1
	# 我试过这里不能直接生成legend，解决方法就是自己定义，创建legend
	# 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
	print(len(hatch))
	patches = [mpatches.Patch(hatch=hatch[i] ,color=color[i], edgecolor='black',label="{:s}".format(labels[i])) for i in range(len(hatch))]
	ax = plt.gca()
	#box = ax.get_position()
	#ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
	# 下面一行中bbox_to_anchor指定了legend的位置
	#ax.legend(handles=patches, bbox_to_anchor=(0.8, 1.15), ncol=3)  # 生成legend
	plt.legend(bbox_to_anchor=(0.9, 1.18), ncol=4,fontsize='large')
	plt.xticks([0.0,0.86 ,1.86,2.86],['','Precision','Recall','F-measure'])
	plt.ylim(0.6, 1.05)
	plt.xlim(0.4, 3.3)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.show()

def draw_method_cmp_hdfs():
	max_lst_of_all = {}  # 一个字典，value是四季最大阵风的风速值，key是年份
	max_lst_of_all['Precision'] = [0.97070,  0.99946, 0.94505, 0.96432,0.96650]
	max_lst_of_all['Recall'] =    [0.63663,  0.77003, 0.98159, 0.98076,0.99394]
	max_lst_of_all['F-measure'] = [ 0.76895, 0.86987, 0.96297, 0.97247 ,0.98003]


	fig = plt.figure()
	count = 1
	color = ['lightskyblue', 'lime', 'blue','red' ,   'white']  # 指定bar的颜色
	hatch = ["/", "X", "+", "*", "."]
	labels = ['PCA', 'LogCluster', 'DeepLog', 'LogAnomaly', 'our method']  # legend标签列表，上面的color即是颜色列表
	for key in max_lst_of_all.keys():
		print(max_lst_of_all[key])
		x = np.arange(count - 0.50, count + 0.5, 0.18)  # 一年有四季，此行指定四季对应的bar的位置，比如2010年：2009.7,2009.9,2010.1,2010.3
		y = max_lst_of_all[key]  # 此行决定了bar的高度(风速值）
		# bar_width = 0.2
		print(x)
		print(y)
		print(count)
		for x1, y1, c1 ,h1,l1 in zip(x, y, color , hatch,labels):  # 遍历以上三者，每一次生成一条bar
			if(count == 1):
				plt.bar(x1, y1, width=0.18, color=c1 , edgecolor='black', hatch=h1 , label = l1 )
			else:
				plt.bar(x1-0.001, y1, width=0.18, color=c1 , edgecolor='black', hatch=h1 )
			plt.text(x1-0.1,y1+0.001,"%.2f"%y1,fontsize=10)
		count += 1
	# 我试过这里不能直接生成legend，解决方法就是自己定义，创建legend
	# 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
	print(len(hatch))
	patches = [mpatches.Patch(hatch=hatch[i] ,color=color[i], edgecolor='black',label="{:s}".format(labels[i])) for i in range(len(hatch))]
	ax = plt.gca()
	#box = ax.get_position()
	#ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
	# 下面一行中bbox_to_anchor指定了legend的位置
	#ax.legend(handles=patches, bbox_to_anchor=(0.8, 1.15), ncol=3)  # 生成legend
	plt.legend(bbox_to_anchor=(1.005, 1.2), ncol=3,fontsize=11)
	plt.xticks([0.0,0.86 ,1.86,2.86],['','Precision','Recall','F-measure'])
	plt.ylim(0.6, 1.05)
	plt.xlim(0.3, 3.4)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.show()

def draw_method_cmp_bgl():
	max_lst_of_all = {}  # 一个字典，value是四季最大阵风的风速值，key是年份
	max_lst_of_all['Precision'] = [ 0.51,0.76504, 0.88445, 0.90819,0.93987]
	max_lst_of_all['Recall'] =    [ 0.60, 0.98608, 0.98675, 0.98916,0.96900]
	max_lst_of_all['F-measure'] = [ 0.55, 0.86161, 0.93280, 0.94694 ,0.95421]


	fig = plt.figure()
	count = 1
	color = ['lightskyblue', 'lime', 'blue','red' ,   'white']  # 指定bar的颜色
	hatch = ["/", "X", "+", "*", "."]
	labels = ['PCA', 'LogCluster', 'DeepLog', 'LogAnomaly', 'our method']  # legend标签列表，上面的color即是颜色列表
	for key in max_lst_of_all.keys():
		print(max_lst_of_all[key])
		x = np.arange(count - 0.50, count + 0.5, 0.18)  # 一年有四季，此行指定四季对应的bar的位置，比如2010年：2009.7,2009.9,2010.1,2010.3
		y = max_lst_of_all[key]  # 此行决定了bar的高度(风速值）
		# bar_width = 0.2
		print(x)
		print(y)
		print(count)
		for x1, y1, c1 ,h1,l1 in zip(x, y, color , hatch,labels):  # 遍历以上三者，每一次生成一条bar
			if(count == 1):
				plt.bar(x1, y1, width=0.18, color=c1 , edgecolor='black', hatch=h1 , label = l1 )
			else:
				plt.bar(x1-0.001, y1, width=0.18, color=c1 , edgecolor='black', hatch=h1 )
			plt.text(x1-0.1,y1+0.001,"%.2f"%y1,fontsize=10)
		count += 1
	# 我试过这里不能直接生成legend，解决方法就是自己定义，创建legend
	# 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
	print(len(hatch))
	patches = [mpatches.Patch(hatch=hatch[i] ,color=color[i], edgecolor='black',label="{:s}".format(labels[i])) for i in range(len(hatch))]
	ax = plt.gca()
	#box = ax.get_position()
	#ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
	# 下面一行中bbox_to_anchor指定了legend的位置
	#ax.legend(handles=patches, bbox_to_anchor=(0.8, 1.15), ncol=3)  # 生成legend
	plt.legend(bbox_to_anchor=(1.005, 1.2), ncol=3,fontsize=11)
	plt.xticks([0.0,0.86 ,1.86,2.86],['','Precision','Recall','F-measure'])
	plt.ylim(0.5, 1.05)
	plt.xlim(0.3, 3.4)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.show()

def draw_layer():
	p = [0.96384, 0.96019, 0.97475, 0.97112]
	r = [0.94655, 0.99133, 0.98367, 0.96449]
	f = [0.95511, 0.97551, 0.97919, 0.96779]
	y = ['1', '2', '3', '4']
	plt.plot(y, p, 'ro-', label='Precision')
	plt.plot(y, r, 'g*:', ms=10, label='Recall')
	plt.plot(y, f, 'b^-', ls='--', ms=10, label='F-measure')
	#	plt.legend(loc="lower left")
	# plt.legend(bbox_to_anchor=(0.8, 1.08), ncol=3, borderaxespad=0)
	# plt.ylabel('f1-score')
	plt.ylim(0.6, 1.05)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	# plt.title("layer")
	plt.show()

def draw_units():
	p = [0.97329, 0.96875, 0.97204, 0.96539]
	r = [0.96306, 0.99038, 0.96835, 0.98907]
	f = [0.96815, 0.97944, 0.97019, 0.97709]
	y = ['32', '64', '128', '256']
	plt.plot(y, p, 'ro-', label='Precision')
	plt.plot(y, r, 'g*:', ms=10, label='Recall')
	plt.plot(y, f, 'b^-', ls='--', ms=10, label='F-measure')
	#	plt.legend(loc="lower left")
	# plt.legend(bbox_to_anchor=(0.8, 1.1), ncol=3, borderaxespad=0)
	# plt.ylabel('f1-score')
	plt.ylim(0.6, 1.05)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	# plt.title("unit")
	plt.show()

def draw_window():
	p = [0.96800, 0.96602, 0.95883, 0.97328, 0.96254]
	r = [0.93788, 0.94726, 0.98474, 0.98414, 0.99204]
	f = [0.95270, 0.95622, 0.97161, 0.97868, 0.97707]
	y = ['8', '9', '10', '11','12']
	plt.plot(y, p, 'ro-', label='Precision')
	plt.plot(y, r, 'g*:', ms=10, label='Recall')
	plt.plot(y, f, 'b^-', ls='--', ms=10, label='F-measure')
	#	plt.legend(loc="lower left")
	# plt.legend(bbox_to_anchor=(0.8, 1.1), ncol=3, borderaxespad=0)
	# plt.ylabel('f1-score')
	plt.ylim(0.6, 1.05)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	# plt.title("unit")
	plt.show()

def draw_candiate():
	p = [0.94651, 0.94966, 0.96595, 0.97199, 0.97475]
	r = [0.99620, 0.99376, 0.99234, 0.98919, 0.98367]
	f = [0.97072, 0.97121, 0.97897, 0.98051, 0.97919]
	y = ['6', '7', '8', '9','10']
	plt.plot(y, p, 'ro-', label='Precision')
	plt.plot(y, r, 'g*:', ms=10, label='Recall')
	plt.plot(y, f, 'b^-', ls='--', ms=10, label='F-measure')
	#	plt.legend(loc="lower left")
	# plt.legend(bbox_to_anchor=(0.8, 1.1), ncol=3, borderaxespad=0)
	# plt.ylabel('f1-score')
	plt.ylim(0.6, 1.05)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	# plt.title("unit")
	plt.show()

def draw_kernel_dim():
	p = [0.97053, 0.96695, 0.97475, 0.97144]
	r = [0.97595, 0.98533, 0.98367, 0.97565]
	f = [0.97323, 0.97606, 0.97919, 0.97354]
	y = ['50', '100', '150', '200']
	plt.plot(y, p, 'ro-', label='Precision')
	plt.plot(y, r, 'g*:', ms=10, label='Recall')
	plt.plot(y, f, 'b^-', ls='--', ms=10, label='F-measure')
	#	plt.legend(loc="lower left")
	# plt.legend(bbox_to_anchor=(0.8, 1.08), ncol=3, borderaxespad=0)
	# plt.ylabel('f1-score')
	plt.ylim(0.6, 1.05)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	# plt.title("layer")
	plt.show()

def draw_kernel():
	p = [0.97202, 0.97252, 0.97475, 0.97111]
	r = [0.96158, 0.94168, 0.98367, 0.82635]
	f = [0.96677, 0.95685, 0.97919, 0.89290]
	y = ['(2,3)', '(3,4)', '(2,3,4)', '(3,4,5)']
	plt.plot(y, p, 'ro-', label='Precision')
	plt.plot(y, r, 'g*:', ms=10, label='Recall')
	plt.plot(y, f, 'b^-', ls='--', ms=10, label='F-measure')
	#	plt.legend(loc="lower left")
	plt.legend(bbox_to_anchor=(0.8, 1.08), ncol=3, borderaxespad=0)
	# plt.ylabel('f1-score')
	# plt.ylim(0.6, 1.05)
	# plt.xticks(fontsize=20)
	# plt.yticks(fontsize=20)
	# plt.title("layer")
	plt.show()


if __name__ == "__main__":

	#draw_precision()
	#draw_recall()
	# draw_top_word()
	# draw_method_cmp()
	# draw_method_cmp_hdfs()
	draw_method_cmp_bgl()
	# draw_layer()
	# draw_units()
	#draw_window()
	# draw_candiate()
	# draw_kernel_dim()
	# draw_kernel()
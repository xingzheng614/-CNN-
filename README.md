# CNN模型实现中文新闻分类

# 1数据预处理
import os
import paddle
import paddle.fluid as fluid
import numpy as np
from multiprocessing import cpu_count

# 定义公共变量
data_root = './'  # 数据集所在目录
data_file = 'news_classify_data.txt'  # 原始样本文件名称
test_file = 'test_list.txt'  # 测试集文件名称
train_file = 'train_list.txt'  # 训练集文件名称
dict_file = 'dict_txt.txt'  # 编码字典文件

# 完整路径
data_file_path = data_root + data_file
test_file_path = data_root + test_file
train_file_path = data_root + train_file
dict_file_path = data_root + dict_file


def create_dict():
    dict_set = set()

    with open(data_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # 遍历所有数据
        for line in lines:
            title = line.split('_!_')[-1].replace('\n', '')

            for w in title:  # 取出每一个字
                dict_set.add(w)

        dict_list = []
        i = 1  # 计数器 确定每个字对应的值
        for s in dict_set:
            dict_list.append([s, i])
            i += 1

        dict_txt = dict(dict_list)  # 将列表直接转成字典
        end_dict = {'<unk>': i}
        dict_txt.update(end_dict)

        # 将字典对象保存带文件中
        with open(dict_file_path, 'w', encoding='utf-8') as f:
            f.write(str(dict_txt))
        print('编码字典生成完毕')


# 队一行标题进行编码
def line_encoding(title, dict_txt, label):
    new_line = ""
    for w in title:
        if w in dict_txt:
            code = str(dict_txt[w])
        else:
            code = str(dict_txt['<unk>'])
        new_line = new_line + code + ','

    new_line = new_line[:-1]  # 去电最后一个多余的逗号
    new_line = new_line + '\t' + label + '\n'
    return new_line


# 清空测试集和训练集文件 将编码后结果写入文件
def create_data_list():
    with open(test_file_path, 'w') as f:
        pass
    with open(train_file_path, 'w') as f:
        pass

    # 在原始文件中，督导所有行
    with open(data_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 所有行的数据

    # 读取字典文件中的内容（只有一行）
    with open(dict_file_path, 'r', encoding='utf-8') as f:
        dict_txt = eval(f.readlines()[0])  # 将字符串使用eval执行成字典

    # 遍历每一行。取出辩题对应的编码
    i = 0
    for line in lines:
        words = line.replace('\n', '').split('_!_')
        label = words[1]
        title = words[3]
        new_line = line_encoding(title, dict_txt, label)

        if i % 10 == 0:
            with open(train_file_path, 'a', encoding='utf-8') as f:
                f.write(new_line)
        else:
            with open(train_file_path, 'a', encoding='utf-8') as f:
                f.write(new_line)
        i += 1
    print('测试集,训练集生成完毕')


create_dict()  # 生成字典
create_data_list()  # 编码写入测试集，训练集


##########模型大件，训练，评估############
#获取字典长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])

    return len(line.keys())

# data_mapper:将传入的一行样本转为增兴列表并返回
def data_mapper(sample):
    data, label = sample
    val = [int(w) for w in data.split(',')]
    return val, int(label)


# 训练集的reader
def train_reader(train_file_path):
    def reader():
        with open(train_file_path, 'r') as f:
            lines = f.readlines()
            np.random.shuffle(lines)  # 打乱样本数据，随机化

            for line in lines:
                data, label = line.split('\t')
                yield data, label  # 元祖

    return paddle.reader.xmap_readers(data_mapper,
                                      reader,
                                      cpu_count(),
                                      1204)


# 测试集reader
def test_reader(test_file_path):
    def reader():
        with open(test_file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data, label = line.split('\t')
                yield data, label  # 元祖

    return paddle.reader.xmap_readers(data_mapper,
                                      reader,
                                      cpu_count(),
                                      1204)


# 搭建网络
def Cnn_net(data, dict_dim, class_dim=10, emb_dim=128, hid_dim=128, hid_dim2=98):
    '''
     搭建TextCNN模型
    :param data:    原始数据
    :param dict_dim: 词典大小
    :param class_dim:　分类数量
    :param emb_dim: 词嵌入计算参数
    :param hid_dim: 第一组卷积运算卷积核数量
    :param hid_dim2:第二组卷积运算卷积核数量
    :return: 计算结果
    '''
    # 生成词向量
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # 并列两组卷积池化
    conv1 = fluid.nets.sequence_conv_pool(input=emb,#输入数据
                                          num_filters=hid_dim,#卷积核数量
                                          filter_size=3,#卷积核大小
                                          act='tanh',#双曲正切激活函数
                                          pool_type='sqrt')#池化类型:开方

    conv2 = fluid.nets.sequence_conv_pool(input=emb,
                                          num_filters=hid_dim2,
                                          filter_size=4,
                                          act='tanh',
                                          pool_type='sqrt')
    # 输出层
    output = fluid.layers.fc(input=[conv1, conv2],  # 前面两组卷积层输出值共同作为输入
                            size=class_dim,  # 分类数量
                            act='softmax')
    return output

# 模型保存路径
model_save_dir = 'model/news_classify/'

# 定义变量 words(lod tensor) label
words = fluid.layers.data(name='words',shape=[1],dtype='int64',lod_level=1)
label = fluid.layers.data(name='label',shape=[1],dtype='int64')

# 获取字典长度
dict_dim = get_dict_len(dict_file_path)
# 构建网络
model = Cnn_net(words,dict_dim)

# 损失函数（交叉熵）#fluid.layers.cross_eropy
cost = fluid.layers.cross_entropy(input=model, #预测值
                                  label=label) #真实值
# 求损失值均值 fluid.layers.mean
avg_cost = fluid.layers.mean(cost)

# 准确率 fluid.layers.accuracy
acc = fluid.layers.accuracy(input=model,#预测值
                            label=label)#真实值

# 克隆default_main_program,用于模型评估 clone  for_test = True
test_program = fluid.default_main_program().clone(for_test=True)

# 优化器  自适应梯度下降优化器 fluid.optimizer
optimizer = fluid.optimizer.AdadeltaOptimizer(learning_rate=0.001)

# 优化 收敛损失值 minisize
optimizer.minimize(avg_cost)

# 定义在什么设备上运行CUDAplace
place = fluid.CUDAPlace(0)

# 执行器 Executer
exe = fluid.Executor(place)

# 初始化
exe.run(fluid.default_startup_program())

# 从训练集读取器拿到一个批次的数据
tr_reader = train_reader(train_file_path)#已经是随机读取器
batch_train_reader = paddle.batch(tr_reader,batch_size=128) # 随机批量读取器

# 从测试集读取器拿到一个批次的数据
ts_reader = test_reader(test_file_path) #原始读取器
batch_test_reader = paddle.batch(ts_reader,batch_size=128)

# 定义feeder参数喂入器
feeder = fluid.DataFeeder(place=place,
                          feed_list=[words,label])

# 开始训练
for pass_id in range(5):
    for batch_id,data in enumerate(batch_train_reader()):
        train_cost,train_acc = exe.run(program=fluid.default_main_program(),
                                       feed=feeder.feed(data),
                                       fetch_list=[avg_cost,acc])

        if batch_id % 100 == 0:
            print(f'pass_id:{pass_id},batch_id:{batch_id},cost:{train_cost[0]},acc:{train_acc[0]}')


    #模型评估
    test_costs_list = []
    test_accs_list = []
    for batch_id,data in enumerate(batch_test_reader()):
        test_cost,test_acc = exe.run(program=test_program,
                                     feed=feeder.feed(data),
                                     fetch_list=[avg_cost,acc])
        test_costs_list.append(test_cost[0]) #记录损失值
        test_accs_list.append(test_acc[0])#记录准确率

    #计算测试集下的平均损失值和准确率
    test_cost_avg = (sum(test_costs_list) / len(test_costs_list))
    test_acc_avg = (sum(test_accs_list) / len(test_accs_list))

    print('pass_id:{},test_cost:{},test_acc:{}'.format(pass_id,test_cost_avg,test_acc_avg))

#训练结束，保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

fluid.io.save_inference_model(model_save_dir, #模型保存路径
                              feeded_var_names=[words.name],#预测是喂入的参数
                              target_vars=[model],#预测结果
                              executor=exe)

###模型加载预测#################
model_save_dir = 'model/news_classify/'

def get_data(sentence):
    #对待预测的文本进行编码
    with open(dict_file_path,'r',encoding='utf-8') as f:
        dict_txt = eval(f.readlines()[0])

    keys = dict_txt.keys()
    ret = []
    for s in sentence:
        if not s in keys:
            s = '<unk>'
        ret.append(int(dict_txt[s]))
    return ret

#执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

#执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

#加载模型

infer_program,feeded_var_names,target_var = fluid.io.load_inference_model(dirname=model_save_dir,executor=exe)

#生成一批数据
texts = []
data1 = get_data('相声届的老前辈你都认识哪几位，郭德纲真的只能被称为小学生！') #文化 0
data2 = get_data('周杰伦陪昆凌戛纳走红毯 任素汐再被前夫证实出轨|说好的六点见') #娱乐 1
data3 = get_data('每天读一遍，坚持15天，英语口语可与外国人交流的水平！') #教育 6
data4 = get_data('两分钟告诉你 什么叫做空 做空如何赚钱？') #财经 3
data5 = get_data('国外男子花百万打造“天使翅膀”，按下按钮后才是霸气的开始') #科技 7
data6 = get_data('沈阳新冠疫情得到有效控制')

texts.append(data1)
texts.append(data2)
texts.append(data3)
texts.append(data4)
texts.append(data5)
texts.append(data6)

#获取每个句子的词数量
base_shape = [[len(c) for c in texts]]

#生成Lod Tensor
tensor_words = fluid.create_lod_tensor(texts,base_shape,place)
#执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]:tensor_words},
                 fetch_list=target_var)

names = ['国际','文化','娱乐','体育','财经','汽车','教育','科技','房产','证券']

for i in range(len(texts)):
    lab = np.argsort(result)[0][i][-1]
    print('预测结果为:{},名称为:{},概率为:{}'.format(lab,names[lab],result[0][i][lab]))


import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm
import os

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'


do_train = True
maxlen = 510
batch_size = 16

def load_train_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    labels = os.listdir(filename)
    file_num = 0
    for label in labels:
        labelpath = os.path.join(filename,label)
        all_file = os.listdir(labelpath)
        ##抽取百分之70的数据作为训练集
        split_num = int(len(all_file)*(1-0.3))
        file_num += split_num
        print(label,split_num,file_num)
        for file in tqdm(all_file[:split_num]):
            file_name = os.path.join(labelpath,file)
            with open(file_name,'r',encoding='utf-8') as f:
                fr = f.read()
                text, label_id =  "".join(fr.split()), labels.index(label)
            if len(text)!=0:
                D.append((text, int(label_id)))

    return D
def load_test_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    labels = os.listdir(filename)
    file_num = 0
    for label in labels:
        labelpath = os.path.join(filename,label)
        all_file = os.listdir(labelpath)
        ##抽取百分之30的数据作为测试集
        split_num = int(len(all_file)*(1-0.3))
        file_n = len(all_file)-split_num
        file_num += file_n
        print(label,file_n,file_num)
        for file in tqdm(all_file[split_num:]):
            file_name = os.path.join(labelpath,file)
            with open(file_name,'r',encoding='utf-8') as f:
                fr = f.read()
                text, label_id =  "".join(fr.split()), labels.index(label)
            if len(text)!=0:
                D.append((text, int(label_id)))

    return D
    

filename = '../data/THUCNews'
labels = os.listdir(filename)
filename = '../data/THUCNews'
train_data = load_train_data(filename)
test_data = load_test_data(filename)

#建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=len(labels),
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(model, 'Embedding-Token', 0.3)



def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('model/best_model.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


print('训练集有{}，测试集有{}'.format(len(train_data),len(test_data)))
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(test_data,batch_size)

if do_train:
    evaluator = Evaluator()

    #初训练
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=1,
        callbacks=[evaluator]
    )
    
    #固定权重训练
    for layer in model.layers[:-1]:
        layer.trainable = False
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=1,
        callbacks=[evaluator]
    )
    

else:
    model.load_weights('model/best_model.weights')
    text = '山东金矿被困井下工人喝到小米粥'    
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    y_pred =  model.predict([[token_ids], [segment_ids]])[0].argmax()
    print('预测类别为：{}'.format(labels[y_pred]))

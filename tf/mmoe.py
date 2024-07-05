# coding:utf-8

import numpy as np
import os
import argparse
import tensorflow as tf
import log_util
import params_conf
from date_helper import DateHelper
import data_consumer
from mmoe import MMoE
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import VarianceScaling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input

from tensorflow.keras.backend import expand_dims, repeat_elements, sum


class MMoE(layers.Layer):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units 隐藏单元
        :param num_experts: Number of experts 专家个数,可以有共享专家,也可以有每个任务独立的专家
        :param num_tasks: Number of tasks  任务个数,和tower个数一致
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights. 专家的权重是否添加偏置
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights. 门控的权重是否添加偏置
        :param expert_activation: Activation function of the expert weights.  专家激活函数
        :param gate_activation: Activation function of the gate weights.  门控激活函数
        :param expert_bias_initializer: Initializer for the expert bias. 专家偏置初始化
        :param gate_bias_initializer: Initializer for the gate bias. 门控偏置初始化
        :param expert_bias_regularizer: Regularizer for the expert bias. 专家正则化
        :param gate_bias_regularizer: Regularizer for the gate bias.  门控正则化
        :param expert_bias_constraint: Constraint for the expert bias. 专家偏置
        :param gate_bias_constraint: Constraint for the gate bias.  门控偏置
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class  附属参数若干
        """
        super(MMoE, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        # self.expert_activation = activations.get(expert_activation)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []
        # 在初始化的过程中,先构建好网络结构
        for i in range(self.num_experts):
            # 有几个专家, 这里就添加几个dense层, dense层的输入为上面传入, 当前层的输出维度为units的值, 隐藏单元个数
            self.expert_layers.append(layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))

        # 门控网络, 门控网络的个数等于任务数目 , 但是取值数据的维度等于专家个数 , mmoe 对每个任务都要融合各个专家的意见。
        # 有几个任务,
        for i in range(self.num_tasks):
            # num_tasks个门控,num_experts维数据
            self.gate_layers.append(layers.Dense(self.num_experts, activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))

    def call(self, inputs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        # assert input_shape is not None and len(input_shape) >= 2

        # 三个输出的网络
        expert_outputs, gate_outputs, final_outputs = [], [], []

        # 专家网络
        # 有几个专家循环几次
        for expert_layer in self.expert_layers:
            # 注意这里是当前专家的变化
            # 输入的元素元素应该是整体embeding contact之后的一堆浮点数维度数据。
            # (batch_size, embedding_size,1)
            expert_output = expand_dims(expert_layer(inputs), axis=2)
            # nums_expert * ( batch_size, unit, 1)
            expert_outputs.append(expert_output)

        # 同 batch 的数据,既然是沿着第一个维度对接,那根本就不用看第二个维度,那个axis的维度数目相加
        #  nums_expert * ( batch_size, unit, 1) -> 这里 contact 之后,列表里 num_experts 个 tensor 在最后一个维度concat到一起,
        # 则最后维度变成了 ( batch_size, unit, nums_expert ),只有最后一个维度的维度值改变了。
        expert_outputs = tf.concat(expert_outputs, 2)

        # 门控网络, 每个门对每个专家均有一个分布函数.
        for gate_layer in self.gate_layers:
            # 对于当前门控,[ batch_size,num_units ] ->  [ nums_expert,batch_size,num_units ]
            # 有多少个任务,就有多少个gate
            # num_task * (batch_size,num_experts),这里对每个专家只有一个数值,和专家的输出维度unit相乘需要拓展维度
            gate_outputs.append(gate_layer(inputs))

        # 这里每个门控对所有的专家进行加权求和
        for gate_output in gate_outputs:
            # 对当前gate,忽略 num_task维度,为 (batch_size, 1, num_experts)
            expanded_gate_output = expand_dims(gate_output, axis=1)
            # 每个专家的输出和gate的数据维度相乘
            # ( batch_size, unit, nums_expert ) *  (batch_size, 1 * units, num_experts),因此 1*units
            # If x has shape (s1, s2, s3) and axis is 1, the output will have shape (s1, s2 * rep, s3).
            # 这里的本质是 门控和专家的输出相乘维度不对,如上面所说,门控维度1和需要拓展到各个专家的输出维度 unit,方便相乘。
            # "*"算子在tensorflow中表示element-wise product，即哈达马积,即两个向量按元素一个一个相乘，组成一个新的向量，结果向量与原向量尺寸相同。
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, self.units, axis=1)

            # 上面输出的维度是 (batch_size, unit, nums_expert ),对第二维nums_expert求和则该维度就变成一个数值 -> (batch_size,unit)
            # 这里对各个专家的结果聚合之后,返回的是一个综合专家对应的输出单元unit维度.
            # 最终有多个门控,上面多个塔,这里返回的是 num_tasks * batch * units 这个维度。
            final_outputs.append(sum(weighted_expert_output, axis=2))

        # 返回的矩阵维度 num_tasks * batch * units
        # 返回多个门控,每个门控有综合多个专家返回的维度 units
        # 这里 final_outputs返回的是个list,元素个数等于 门控个数也等于任务个数
        return final_outputs


def init_args():
    parser = argparse.ArgumentParser(description='dnn_demo')
    parser.add_argument("--mode", default="train")
    parser.add_argument("--train_data_dir")
    parser.add_argument("--model_output_dir")
    parser.add_argument("--cur_date")
    parser.add_argument("--log", default="../log/tensorboard")
    parser.add_argument('--use_gpu', default=False, type=bool)
    args = parser.parse_args()
    return args


def get_feature_column_map():
    key_hash_size_map = {
        "adid": 10000,
        "site_id": 10000,
        "site_domain": 10000,
        "site_category": 10000,
        "app_id": 10000,
        "app_domain": 10000,
        "app_category": 1000,
        "device_id": 1000,
        "device_ip": 10000,
        "device_type": 10,
        "device_conn_type": 10,
    }

    feature_column_map = dict()
    for key, value in key_hash_size_map.items():
        feature_column_map.update({key: tf.feature_column.categorical_column_with_hash_bucket(
            key, hash_bucket_size=value, dtype=tf.string)})

    return feature_column_map


def build_embeding():
    feature_map = get_feature_column_map()
    feature_inputs_list = []

    def get_field_emb(categorical_col_key, emb_size=16, input_shape=(1,)):
        # print(categorical_col_key)
        embed_col = tf.feature_column.embedding_column(feature_map[categorical_col_key], emb_size)
        # 层名字不可以相同,不然会报错
        dense_feature_layer = tf.keras.layers.DenseFeatures(embed_col, name=categorical_col_key + "_emb2dense")
        feature_layer_inputs = dict()

        # input和DenseFeatures必须要用dict来存和联合使用，深坑啊！！
        feature_layer_inputs[categorical_col_key] = tf.keras.Input(shape=(1,), dtype=tf.dtypes.string,
                                                                   name=categorical_col_key)
        # 保存供 model input 使用.
        feature_inputs_list.append(feature_layer_inputs[categorical_col_key])
        return dense_feature_layer(feature_layer_inputs)

    embeding_map = {}
    for key, value in feature_map.items():
        # print("key:" + key)
        embeding_map.update({key: get_field_emb(key)})

    return embeding_map, feature_inputs_list


def build_dnn_net(net, params_conf, name="ctr"):
    # 可以在下面接入残差网络
    for i, dnn_hidden_size in enumerate(params_conf.DNN_HIDDEN_SIZES):  # DNN_HIDDEN_SIZES = [512, 128, 64]
        net = tf.keras.layers.Dense(dnn_hidden_size, activation="relu", name="overall_dense_%s_%s" % (i, name))(net)
    return net


def build_model(emb_map, inputs_list):
    # 需要特殊处理和交叉的特征,以及需要短接残差的特征,可以单独拿出来
    define_list = []
    adid_emb = emb_map["adid"]
    device_id_emd = emb_map["device_id"]
    ad_x_device = tf.multiply(adid_emb, device_id_emd)

    define_list.append(ad_x_device)

    # 直接可以拼接的特征
    common_list = []
    for key, value in emb_map.items():
        common_list.append(value)

    # embeding contact
    net = tf.keras.layers.concatenate(define_list + common_list)

    # Set up MMoE layer
    # 返回的矩阵维度 num_tasks * batch * units
    # 返回多个门控,每个门控有综合多个专家返回的维度 units
    # 这里 final_outputs返回的是个list,元素个数等于 门控个数也等于任务个数
    mmoe_layers = MMoE(units=4, num_experts=8, num_tasks=2)(net)

    output_layers = []

    # Build tower layer from MMoE layer
    # 对每个 mmoe layer, 后面均接着 2层dense 到输出,
    # list,元素个数等于 门控个数也等于任务个数
    for index, task_layer in enumerate(mmoe_layers):
        # 对当前task, batch * units 维度的数据, 介入隐藏层
        tower_layer = layers.Dense(units=8, activation='relu', kernel_initializer=VarianceScaling())(task_layer)
        # 这里unit为1,当前任务为2分类
        output_layer = layers.Dense(units=1, name="task_%s" % (index), activation='sigmoid',
                                    kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    # 这里定义了模型骨架,input 为模型输入参数,而output_layers 是一个列表,列表里返回了2个任务各自的logit
    # 其实分别返回了每个task的logit,logit这里为分类数目维度的数组,2维过softmax

    model = Model(inputs=[inputs_list], outputs=output_layers)

    return model


def train():
    output_root_dir = "{}/{}/{}".format(params_conf.BASE_DIR, args.model_output_dir, args.cur_date)
    os.mkdir(output_root_dir)
    model_full_output_dir = os.path.join(output_root_dir, "model_savedmodel")
    # print info log
    log_util.info("model_output_dir: %s" % model_full_output_dir)

    # 重置keras的状态
    tf.keras.backend.clear_session()
    log_util.info("start train...")
    train_date_list = DateHelper.get_date_range(DateHelper.get_date(-1, args.cur_date),
                                                DateHelper.get_date(0, args.cur_date))
    train_date_list.reverse()
    print("train_date_list:" + ",".join(train_date_list))

    # load data from tf.data,兼容csv 和 tf_record
    train_set, test_set = data_consumer.get_dataset(args.train_data_dir, train_date_list,
                                                    get_feature_column_map().values())
    # train_x, train_y = train_set

    log_util.info("get train data finish ...")

    emb_map, feature_inputs_list = build_embeding()
    log_util.info("build embeding finish...")

    # 构建模型
    model = build_model(emb_map, feature_inputs_list)
    log_util.info("build model finish...")

    def my_sparse_categorical_crossentropy(y_true, y_pred):
        return tf.keras.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    opt = tf.keras.optimizers.Adam(params_conf.LEARNING_RATE)

    # 注意这里设定了2个损失分别对应[ctr_pred, ctcvr_pred] 这两个任务
    # loss_weights=[1.0, 1.0]这种方式可以固定的调整2个任务的loss权重。
    model.compile(
        optimizer=opt,
        loss={'task_0': 'binary_crossentropy', 'task_1': 'binary_crossentropy'},
        loss_weights=[1.0, 1.0],
        metrics=[
            tf.keras.metrics.AUC(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()]
    )
    model.summary()
    # tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True, dpi=150)

    print("start training")

    # 需要设置profile_batch=0，tensorboard页面才会一直保持更新
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.log,
        histogram_freq=1,
        write_graph=True,
        update_freq=params_conf.BATCH_SIZE * 200,
        embeddings_freq=1,
        profile_batch=0)

    # 定义衰减式学习率
    class LearningRateExponentialDecay:

        def __init__(self, initial_learning_rate, decay_epochs, decay_rate):
            self.initial_learning_rate = initial_learning_rate
            self.decay_epochs = decay_epochs
            self.decay_rate = decay_rate

        def __call__(self, epoch):
            dtype = type(self.initial_learning_rate)
            decay_epochs = np.array(self.decay_epochs).astype(dtype)
            decay_rate = np.array(self.decay_rate).astype(dtype)
            epoch = np.array(epoch).astype(dtype)
            p = epoch / decay_epochs
            lr = self.initial_learning_rate * np.power(decay_rate, p)
            return lr

    lr_schedule = LearningRateExponentialDecay(
        params_conf.INIT_LR, params_conf.LR_DECAY_EPOCHS, params_conf.LR_DECAY_RATE)

    # 该回调函数是学习率调度器

    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

    # 训练
    # 注意这里的train_set 可以使用for循环迭代,tf 2.0默认支持eager模式
    # 这里的train_set 包含两部分,第一部分是feature,第二部分是label ( click, click & conversion)
    # 注意这里是 feature,(click, click & conversion),第二项是tuple,不能是数组或列表[],不然报数据维度不对,坑死爹了。
    model.fit(
        train_set,
        # train_set["labels"],
        # validation_data=test_set,
        epochs=params_conf.NUM_EPOCHS,  # NUM_EPOCHS = 10
        steps_per_epoch=params_conf.STEPS_PER_EPHCH,
        # validation_steps=params_conf.VALIDATION_STEPS,
        #
        # callbacks=[tensorboard_callback, lr_schedule_callback]
    )

    # 模型保存
    tf.keras.models.save_model(model, model_full_output_dir)

    # tf.saved_model.save(model, model_full_output_dir)
    print("save saved_model success")


if __name__ == "__main__":
    print(tf.__version__)
    tf.compat.v1.disable_eager_execution()

    # run tensorboard:
    # tensorboard --port=8008 --host=localhost --logdir=../log
    args = init_args()
    if args.mode == "train":
        train()
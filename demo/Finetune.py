import paddle
import paddlehub as hub

import os

os.system("python3 foodClassificationPj/GenerateDataset.py")

paddle.enable_static()

# ---------------------------------------------------------- #

# 选择模型
# 此处代码为加载 Hub 提供的图像分类的预训练模型
module = hub.Module(name="resnet50_vd_imagenet_ssld")

# ---------------------------------------------------------- #
# 三、数据准备
# 此处的数据准备使用的是paddlehub提供的猫狗分类数据集，如果想要使用自定义的数据进行体验，需要自定义数据，请查看适配自定义数据
# 如果想加载自定义数据集完成迁移学习，详细参见自定义数据集

from paddlehub.dataset.base_cv_dataset import BaseCVDataset


class DemoDataset(BaseCVDataset):
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = (
            "/Users/baijiale/Documents/Code/py/aistudio/foodClassificationPj/data"
        )
        super(DemoDataset, self).__init__(
            base_path=self.dataset_dir,
            train_list_file="train_list.txt",
            validate_list_file="validate_list.txt",
            test_list_file="test_list.txt",
            # predict_file="predict_list.txt",
            label_list=[
                "1-Basset Hound",
                "2-Beagle",
                "3-Gray hound",
                "4-German Shepherd",
                "5-Schnauzer",
                "6-Springer Spaniel",
                "7-Labrador",
                "8-Coker",
                "9-Oldenglishsheepdog",
                "10-Shetlan",
            ],
        )


dataset = DemoDataset()

# ---------------------------------------------------------- #
# 四、生成Reader
# 接着生成一个图像分类的reader，reader负责将dataset的数据进行预处理，接着以特定格式组织并输入给模型进行训练。
# 当我们生成一个图像分类的reader时，需要指定输入图片的大小

data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),  # 预期图片经过reader处理后的图像宽度
    image_height=module.get_expected_image_height(),  # 预期图片经过reader处理后的图像高度
    images_mean=module.get_pretrained_images_mean(),  # 进行图片标准化处理时所减均值。默认为None
    images_std=module.get_pretrained_images_std(),  # 进行图片标准化处理时所除标准差。默认为None
    dataset=dataset,
)

# ---------------------------------------------------------- #
# 五、选择运行时配置
# 在进行Finetune前，我们可以设置一些运行时的配置，例如如下代码中的配置，表示：
# use_cuda：设置为False表示使用CPU进行训练。如果您本机支持GPU，且安装的是GPU版本的PaddlePaddle，我们建议您将这个选项设置为True；
# epoch：要求Finetune的任务只遍历1次训练集；
# batch_size：每次训练的时候，给模型输入的每批数据大小为32，模型训练时能够并行处理批数据，因此batch_size越大，训练的效率越高，但是同时带来了内存的负荷，过大的batch_size可能导致内存不足而无法训练，因此选择一个合适的batch_size是很重要的一步；
# log_interval：每隔10 step打印一次训练日志；
# eval_interval：每隔50 step在验证集上进行一次性能评估；
# checkpoint_dir：将训练的参数和数据保存到cv_finetune_turtorial_demo目录中；
# strategy：使用DefaultFinetuneStrategy策略进行finetune；
# 更多运行配置，请查看RunConfig
# 同时PaddleHub提供了许多优化策略，如AdamWeightDecayStrategy、ULMFiTStrategy、DefaultFinetuneStrategy等，详细信息参见策略

config = hub.RunConfig(
    use_cuda=False,  # 是否使用GPU训练，默认为False；
    num_epoch=3,  # Fine-tune的轮数；使用4轮，直到训练准确率达到90%多
    checkpoint_dir="cv_finetune_turtorial_demo",  # 模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
    batch_size=32,  # 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
    eval_interval=50,  # 模型评估的间隔，默认每100个step评估一次验证集；
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy(),  # Fine-tune优化策略；
)

# ---------------------------------------------------------- #
# 六、组建Finetune Task
# 有了合适的预训练模型和准备要迁移的数据集后，我们开始组建一个Task。
# 由于猫狗分类是一个二分类的任务，而我们下载的分类module是在ImageNet数据集上训练的千分类模型，所以我们需要对模型进行简单的微调，把模型改造为一个二分类模型：
# 获取module的上下文环境，包括输入和输出的变量，以及Paddle Program；
# 从输出变量中找到特征图提取层feature_map；
# 在feature_map后面接入一个全连接层，生成Task；

# 获取 module 的上下文信息包括输入、输出变量以及 paddle program
input_dict, output_dict, program = module.context(trainable=True)

# 待传入图片格式
img = input_dict["image"]

# 从预训练模型的输出变量中找到最后一层特征图，提取最后一层的feature_map
feature_map = output_dict["feature_map"]

# 待传入的变量名字列表
feed_list = [img.name]

task = hub.ImageClassifierTask(
    data_reader=data_reader,  # 提供数据的 Reader
    feed_list=feed_list,  # 待 feed 变量的名字列表
    feature=feature_map,  # 输入的特征矩阵
    num_classes=dataset.num_labels,  # 分类任务的类别数量，此处来自于数据集的 num_labels
    config=config,  # 运行配置
)

# ---------------------------------------------------------- #
# 七、开始Finetune
# 我们选择finetune_and_eval接口来进行模型训练，这个接口在finetune的过程中，会周期性的进行模型效果的评估，以便我们了解整个训练过程的性能变化。

run_states = (
    task.finetune_and_eval()
)  # 通过众多 finetune API 中的 finetune_and_eval 接口，可以一边训练网络，一边打印结果

# ---------------------------------------------------------- #
# 八、使用模型进行预测
# 当Finetune完成后，我们使用模型来进行预测，先通过以下命令来获取测试的图片
# 注意：填入测试图片路径后方可开始测试
# 预测代码如下：

import numpy as np

data = [
    "/home/aistudio/test-dog10/3-gray hound-test/2.jpg"
]  # 此处传入需要识别的照片地址
label_map = dataset.label_dict()
index = 0

# get classification result
run_states = task.predict(data=data)  # 进行预测
results = [
    run_state.run_results for run_state in run_states
]  # 得到用新模型预测test照片的结果

for batch_result in results:
    # get predict index
    batch_result = np.argmax(batch_result, axis=2)[0]
    for result in batch_result:
        index += 1
        result = label_map[result]
        print(
            "input %i is %s, and the predict result is %s"
            % (index, data[index - 1], result)
        )

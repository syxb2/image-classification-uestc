# 团队任务：使用finetune迁移训练（数据集增强对比）

# 一、简介
# 本任务为，将此前我们找到的准确率最高的预训练网络模型，使用新选择任务的数据集进行finetune训练，并测试模型情况。

# 二、准备工作
# 首先导入必要的python包

# In [1]
# !pip install paddlehub==1.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
import paddle
import paddlehub as hub
paddle.enable_static()

# 接下来我们要在PaddleHub中选择合适的预训练模型来Finetune。
# 请在此处选择前期任务准确率最高的模型
# PaddleHub 还有着许多的图像分类预训练模型，更多信息参见PaddleHub官方网站

# In [5]
# 选择模型
# 此处代码为加载Hub提供的图像分类的预训练模型
module = hub.Module(name=" ") 

# 三、数据准备
# 此处使用需要自定义数据集
#上传制作成功的小狗数据集后，使用类似的这段代码去做解压缩，注意路径
# !unzip -o /home/aistudio/dataset.zip
#新的数据集类定义

# In [4]
# 使用自定义数据集
dataset = 

# 四、生成Reader
# 接着生成一个图像分类的reader，reader负责将dataset的数据进行预处理，接着以特定格式组织并输入给模型进行训练。
# 当我们生成一个图像分类的reader时，需要指定输入图片的大小

# In [7]
data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(), #预期图片经过reader处理后的图像宽度
    image_height=module.get_expected_image_height(),#预期图片经过reader处理后的图像高度
    images_mean=module.get_pretrained_images_mean(),#进行图片标准化处理时所减均值。默认为None
    images_std=module.get_pretrained_images_std(), #进行图片标准化处理时所除标准差。默认为None
    dataset=dataset)

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

# In [6]
config = hub.RunConfig(
    use_cuda=True,                                            #是否使用GPU训练，默认为False；
    num_epoch=4,                                              #Fine-tune的轮数；使用4轮，直到训练准确率达到90%多
    checkpoint_dir="cv_finetune_turtorial_demo",              #模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
    batch_size=32,                                            #训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
    eval_interval=50,                                         #模型评估的间隔，默认每100个step评估一次验证集；
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy()) #Fine-tune优化策略；

# 六、组建Finetune Task
# 有了合适的预训练模型和准备要迁移的数据集后，我们开始组建一个Task。
# 我们下载的分类module是在ImageNet数据集上训练的千分类模型，所以我们需要对模型进行简单的微调，把模型改造为一个十分类模型：
# 获取module的上下文环境，包括输入和输出的变量，以及Paddle Program；
# 从输出变量中找到特征图提取层feature_map；
# 在feature_map后面接入一个全连接层，生成Task；

# In [7]
#获取module的上下文信息包括输入、输出变量以及paddle program
input_dict, output_dict, program = module.context(trainable=True) 
#待传入图片格式
img = input_dict["image"]  
#从预训练模型的输出变量中找到最后一层特征层，提取最后一层的feature_map
feature_map = output_dict["feature_map"]   
#待传入的变量名字列表
feed_list = [img.name]
task = hub.ImageClassifierTask(
    data_reader=data_reader,        #提供数据的Reader
    feed_list=feed_list,            #待feed变量的名字列表
    feature=feature_map,            #输入的特征矩阵
    num_classes=dataset.num_labels, #分类任务的类别数量，此处来自于数据集的num_labels
    config=config)                  #运行配置
# 如果想改变迁移任务组网，详细参见自定义迁移任务

# 七、开始Finetune
# 我们选择finetune_and_eval接口来进行模型训练，这个接口在finetune的过程中，会周期性的进行模型效果的评估，以便我们了解整个训练过程的性能变化。

# In [8]
run_states = task.finetune_and_eval() #通过众多finetune API中的finetune_and_eval接口，可以一边训练网络，一边打印结果

# 八、使用模型进行预测
# 当Finetune完成后，我们使用模型来进行预测，先通过以下命令来获取测试的图片
# 注意：填入测试图片路径后方可开始测试
# 预测代码示例如下：

# In [10]
import numpy as np
#此处传入需要识别的照片地址，可以是一个地址，也可以是个地址列表
#此处可以参考群内《paddlehub制作自定义数据集》最后一个示例
data = ["  "]        
label_map = dataset.label_dict()
index = 0
# get classification result
run_states = task.predict(data=data) #进行预测
results = [run_state.run_results for run_state in run_states] #得到用新模型预测test照片的结果
for batch_result in results:
    # get predict index
    batch_result = np.argmax(batch_result, axis=2)[0]
    for result in batch_result:
        index += 1
        result = label_map[result]
        print("input %i is %s, and the predict result is %s" % (index, data[index - 1], result))

# 剩下的主程序任务：
# 对新的模型识别结果进行混淆矩阵计算
# 对新的模型识别结果计算精确率、召回率
# 建议绘制图示
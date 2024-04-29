import paddle
import paddlehub as hub
import numpy as np
from paddlehub.dataset.base_cv_dataset import BaseCVDataset
from sklearn.metrics import confusion_matrix


import dataListGenerating
import finetune


def main() -> None:
    dataListGenerating.generate_data_list_txt(dataset="train")
    dataListGenerating.generate_data_list_txt(dataset="test")
    dataListGenerating.generate_data_list_txt(dataset="validate")

    dataListGenerating.generate_label_list_txt()

    data = []
    data = dataListGenerating.generate_data_list("validate")

    paddle.enable_static()
    module = hub.Module(name="resnet_v2_50_imagenet") 

    class DemoDataset(BaseCVDataset):
        def __init__(self):
            # 数据集存放位置
            self.dataset_dir = dataListGenerating.DATAPATH
            super(DemoDataset, self).__init__(
                base_path=self.dataset_dir,
                train_list_file="train_list.txt",
                validate_list_file="validate_list.txt",
                test_list_file="test_list.txt",
                # predict_file="predict_list.txt",
                label_list=["bread", "dessert", "egg", "meat", "noodles"],
            )

    dataset = DemoDataset()

    data_reader = hub.reader.ImageClassificationReader(
        image_width=module.get_expected_image_width(),  # 预期图片经过reader处理后的图像宽度
        image_height=module.get_expected_image_height(),  # 预期图片经过reader处理后的图像高度
        images_mean=module.get_pretrained_images_mean(),  # 进行图片标准化处理时所减均值。默认为None
        images_std=module.get_pretrained_images_std(),  # 进行图片标准化处理时所除标准差。默认为None
        dataset=dataset,
    )

    config = hub.RunConfig(
        use_cuda=False,  # 是否使用GPU训练，默认为False；
        num_epoch=4,  # Fine-tune的轮数；使用4轮，直到训练准确率达到90%多
        checkpoint_dir="cv_finetune_turtorial_demo",  # 模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
        batch_size=32,  # 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
        eval_interval=50,  # 模型评估的间隔，默认每100个step评估一次验证集；
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy(),
    )  # Fine-tune优化策略；

    # 获取module的上下文信息包括输入、输出变量以及paddle program
    input_dict, output_dict, program = module.context(trainable=True)

    # 待传入图片格式
    img = input_dict["image"]

    # 从预训练模型的输出变量中找到最后一层特征图，提取最后一层的feature_map
    feature_map = output_dict["feature_map"]

    # 待传入的变量名字列表
    feed_list = [img.name]

    task = hub.ImageClassifierTask(
        data_reader=data_reader,  # 提供数据的Reader
        feed_list=feed_list,  # 待feed变量的名字列表
        feature=feature_map,  # 输入的特征矩阵
        num_classes=dataset.num_labels,  # 分类任务的类别数量，此处来自于数据集的num_labels
        config=config,
    )  # 运行配置

    run_states = (
        task.finetune_and_eval()
    )  # 通过众多finetune API中的finetune_and_eval接口，可以一边训练网络，一边打印结果

    label_map = dataset.label_dict()
    index = 0

    # get classification result
    run_states = task.predict(data=data)  # 进行预测
    results = [
        run_state.run_results for run_state in run_states
    ]  # 得到用新模型预测test照片的结果

    truelable_lst = []
    prelable_lst = []
    for batch_result in results:
        # get predict index
        batch_result = np.argmax(batch_result, axis=2)[0]
        for result in batch_result:
            index += 1
            result = label_map[result]
            truelable_lst.append((data[index - 1].split("/")[5]).split("-")[1])
            prelable_lst.append(result)

    cm = confusion_matrix(y_true, y_pred)
    y_true = truelable_lst
    y_pred = prelable_lst

    # 结果评估
    finetune.plot_confusion_matrix(
        cm, ["bread", "dessert", "egg", "meat", "noodles"], "ConfusedMatrix"
    )
    finetune.result_evalution(truelable_lst, prelable_lst)

    return


# execute the code
if __name__ == "__main__":
    main()

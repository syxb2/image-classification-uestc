"""
README

使用说明:
1、运行前请先设置 DATAPATH 为相应的路径（48行）
2、将 DATAPATH 路径的文件设置为如下所示：
{
DATAPATH
├── test
│   ├── 1-bread
│   ├── 2-dessert
│   ├── 3-egg
│   ├── 4-meat
│   └── 5-noodles
├── train
│   ├── 1-bread
│   ├── 2-dessert
│   ├── 3-egg
│   ├── 4-meat
│   └── 5-noodles
└── validate
    ├── 1-bread
    ├── 2-dessert
    ├── 3-egg
    ├── 4-meat
    └── 5-noodles
}
3、各个 list 将生成在 DATAPATH 目录下

运行结果:
[INFO] DATAPATH = /Users/baijiale/Code/aistudio/data
[INFO] FOLDERNAMELIST = ['2-dessert', '5-noodles', '4-meat', '3-egg', '1-bread']
[INFO] train_count = 896
[INFO] train list generating finished
[INFO] test_count = 104
[INFO] test list generating finished
[INFO] validate_count = 108
[INFO] validate list generating finished
[INFO] label_count = 5
[INFO] label list generating finished
[INFO] data = ['/Users/baijiale/Code/aistudio/data/validate/2-dessert/8.jpg', '/Users/baijiale/Do......
[INFO] data_count = 108
[INFO] the list "data" generating finished
"""

import os

DATAPATH = "/Users/baijiale/Code/image_classification_semester2prj/data"
print("[INFO] DATAPATH = %s" % DATAPATH)


def init_folders_list(list) -> list:
    # init list "list"
    list = os.listdir(
        DATAPATH + "/train"
    )  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    if ".DS_Store" in list:
        list.remove(".DS_Store")
    if ".ipynb_checkpoints" in list:
        list.remove(".ipynb_checkpoints")

    print("[INFO] FOLDERNAMELIST = %s" % list)

    return list


FOLDERNAMELIST = []
FOLDERNAMELIST = init_folders_list(list=FOLDERNAMELIST)


def generate_data_list_txt(dataset) -> None:
    # init the pointer p which points to "train_list.txt"
    p = open(DATAPATH + "/" + dataset + "_list.txt", "w")

    # init the variable "count" which is used to count the number of values (count of pictures in "dataset")
    count = 0

    # generate "train_list.txt"
    for fld_name in FOLDERNAMELIST:
        image_names = os.listdir(DATAPATH + "/" + dataset + "/" + fld_name)
        for img_name in image_names:
            if img_name != ".DS_Store":
                p.write(
                    DATAPATH
                    + "/"
                    + dataset
                    + "/"
                    + fld_name
                    + "/"
                    + img_name
                    + " "
                    + (str(int(fld_name.split("-")[0]) - 1))
                    + "\n"
                )
                count = count + 1

    print("[INFO] %s_count = %d" % (dataset, count))
    print("\033[32m[INFO] %s list generating finished\033[0m" % dataset)

    p.close()

    return


def generate_label_list_txt() -> None:
    # init the pointer plabel which points to "validate_list.txt"
    plable = open(DATAPATH + "/label_list.txt", "w")

    # init the variable "label_count" which is used to count the number of values
    label_count = 0

    # generate label_list
    for fld_name in FOLDERNAMELIST:
        if fld_name != ".DS_Store":
            plable.write(fld_name + "-test" + "\n")  # .split("-")[1]
            label_count = label_count + 1

    print("[INFO] label_count = %d" % label_count)
    print("\033[32m[INFO] label list generating finished\033[0m")

    plable.close()

    return


def generate_data_list(dataset) -> list:
    imgpath_list = []
    data_count = 0
    for folder_name in FOLDERNAMELIST:
        img_paths = os.listdir(DATAPATH + "/" + dataset + "/" + folder_name)
        for img in img_paths:
            imgpath_list.append(
                DATAPATH + "/" + dataset + "/" + folder_name + "/" + img
            )
            data_count = data_count + 1
    # print("[INFO] data = %s" % imgpath_list)
    print("[INFO] data_count = %d" % data_count)
    print('\033[32m[INFO] the list "data" generating finished\033[m')

    return imgpath_list
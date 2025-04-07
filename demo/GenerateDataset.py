import os

path = "/Users/baijiale/Code/py/aistudio/data/test"
folders_name = os.listdir(
    path
)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
print(folders_name)
if ".DS_Store" in folders_name:
    folders_name.remove(".DS_Store")
if ".ipynb_checkpoints" in folders_name:
    folders_name.remove(".ipynb_checkpoints")

for file in folders_name:
    print(file)

a = open(
    "/Users/baijiale/Code/py/aistudio/data/train_list.txt",
    "w",
)
b = open(
    "/Users/baijiale/Code/py/aistudio/data/validate_list.txt",
    "w",
)

count = 0
val_count = 0
train_count = 0

for fld_name in folders_name:
    image_names = os.listdir(path + "/" + fld_name)
    for img_name in image_names:
        if img_name != ".DS_Store":
            if count % 8 == 0:
                b.write(
                    path
                    + "/"
                    + fld_name
                    + "/"
                    + img_name
                    + " "
                    + (str(int(fld_name.split("-")[0]) - 1))
                    + "\n"
                )
                val_count = val_count + 1
            else:
                a.write(
                    path
                    + "/"
                    + fld_name
                    + "/"
                    + img_name
                    + " "
                    + (str(int(fld_name.split("-")[0]) - 1))
                    + "\n"
                )
                train_count = train_count + 1
        count = count + 1

a.close()
b.close()
print("train_list生成完毕，train数据集共{}个数据".format(train_count))
print("val_list生成完毕，val数据集共{}个数据".format(val_count))
print("合计{}个数据".format(count))

# ---------------------------------------------------------- #

d = open(
    "/Users/baijiale/Code/py/aistudio/foodClassificationPj/data/label_list.txt",
    "w",
)

count = 0
label_count = 0

for name in folders_name:
    d.write(name + "\n")
    label_count = label_count + 1
    count = count + 1

d.close()
print("label_list生成完毕，label数据集共{}个数据".format(label_count))
print("合计{}个数据".format(count))

# ---------------------------------------------------------- #

c = open(
    "/Users/baijiale/Code/py/aistudio/foodClassificationPj/data/test_list.txt",
    "w",
)

count = 0
test_count = 0

for name in folders_name:
    image_names = os.listdir(path + "/" + name)
    for img_name in image_names:
        if img_name != ".DS_Store":
            c.write(
                path
                + "/"
                + name
                + "/"
                + img_name
                + " "
                + (str(int(name.split("-")[0]) - 1))
                + "\n"
            )
            test_count = test_count + 1
        count = count + 1

c.close()
print("test_list生成完毕，test数据集共{}个数据".format(test_count))
print("合计{}个数据".format(count))

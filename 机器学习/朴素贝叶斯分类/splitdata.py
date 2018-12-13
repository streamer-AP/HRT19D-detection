#将样本按照拆分为训练集和数据集
import random
test_data_ratio=0.3
with open("./spambase.data","r") as file_spambasedata:
    spambasedata=file_spambasedata.readlines()
train_data=[]
test_data=[]
for sample in spambasedata:
    if random.random()<test_data_ratio:
        test_data.append(sample)
    else:
        train_data.append(sample)
with open("./spambase_test.data","w") as file_spambasetestdata:
    file_spambasetestdata.writelines(test_data)
with open("./spambase_train.data","w") as file_spambasetraindata:
    file_spambasetraindata.writelines(train_data)


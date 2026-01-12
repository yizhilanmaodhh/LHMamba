##PyCharmCommunityEdition2023.1，Python3.11 环境。

##文件夹中包含模型文件夹models,以及模型训练文件train.py,模型加载，数据集读取，训练验证函数文件utils.py。

##在utils.py中可以修改数据集路径，在train.py中可以对超参数epoch，batch size等进行调整，以及模型的调用进行更换，然后进行模型训练

##模型文件夹models中包含模型文件LMHMamba.py以及其他的模型文件，LMHMamba.py文件中包含注意力机制模块，LMHMamba_Block1模块，BasicStage模块，Stem模块，LMHMamba模块以及模型注册代码。

##模型训练文件train.py中包含训练参数设置，main函数源码。

##utils.py文件中包含数据集路径，模型加载，训练集和验证集读取，训练函数train_one_epoch和验证函数evaluate，学习率调整函数create_lr_scheduler和样本数量统计函数sample


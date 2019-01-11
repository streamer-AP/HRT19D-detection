使用原有的tensorflow文本分类器复现。
两层结构。
第一层57个节点，第二层1个。
tensorflow， numpy， matplotlib.pyplot。
训练一百次。
损失率0.2以下，正确率约0.95。
	
每十个编码采用分组累加判断，当大于5时为1。
flag 10001001110101110100111011011100100101111110000011001001

============================================
笔记：
model = tf.keras.Sequential() 建立模型
model.add(layers.Dense(57, activation='relu')) 增加层，57个节点
model.compile(optimizer=tf.train.AdamOptimizer(),设置训练流程
              loss='binary_crossentropy',要在优化期间最小化的函数。
              metrics=['accuracy']) 用于监控训练。
tf.keras.Model.fit 采用三个重要参数：
epochs：以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）。
batch_size：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。
validation_data：在对模型进行原型设计时，您需要轻松监控该模型在某些验证数据上达到的效果。传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标。


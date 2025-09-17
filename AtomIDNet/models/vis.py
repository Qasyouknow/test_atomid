import torch
from torchviz import make_dot
import unet  # This imports the unet.py file

# 从您的脚本中加载模型
model = unet.get_model(input_dim=3, output_dim=1)

# 创建一个虚拟的输入张量
# 模型需要一个4维的张量：(batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224)

# 前向传播以获得输出
# 模型返回一个元组：(diameter, list_of_segmentation_outputs)
# 我们将使用第一个分割输出来进行可视化
diameter, seg_outputs = model(dummy_input)
y = seg_outputs[0]

# 生成网络结构图
# params=dict(model.named_parameters()) 将使用参数名称标记图形
dot = make_dot(y, params=dict(model.named_parameters()))

dot.format = 'pdf'
dot.render('unet_architecture')

import graphviz

# 创建一个新的有向图
dot = graphviz.Digraph('UNet_Architecture', comment='Simplified U-Net Architecture')
dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
dot.attr(rankdir='TB', splines='ortho') # 设置为从上到下的布局

# --- 编码器路径 ---
with dot.subgraph(name='cluster_encoder') as c:
    c.attr(label='Encoder Path (Downsampling)', style='rounded', color='gray')
    c.node('input', 'Input Image', shape='ellipse', fillcolor='lightgreen')
    c.node('e1', 'Residual Block\n(Features: 32)')
    c.node('e2', 'Residual Block\n(Features: 64)')
    c.node('e3', 'Residual Block\n(Features: 128)')
    c.node('e4', 'Residual Block\n(Features: 256)')
    c.edge('input', 'e1')
    c.edge('e1', 'e2', label='MaxPool')
    c.edge('e2', 'e3', label='MaxPool')
    c.edge('e3', 'e4', label='MaxPool')

# --- 瓶颈层 ---
dot.node('b', 'Bottleneck\nResidual Block\n(Features: 512)')
dot.edge('e4', 'b', label='MaxPool')

# --- 解码器路径 ---
with dot.subgraph(name='cluster_decoder') as c:
    c.attr(label='Decoder Path (Upsampling)', style='rounded', color='gray')
    c.node('d4', 'Fusion Block\n(Features: 256)')
    c.node('d3', 'Fusion Block\n(Features: 128)')
    c.node('d2', 'Fusion Block\n(Features: 64)')
    c.node('d1', 'Fusion Block\n(Features: 32)')
    c.node('output', 'Output Masks', shape='ellipse', fillcolor='lightgreen')
    c.edge('d4', 'd3', label='UpConv')
    c.edge('d3', 'd2', label='UpConv')
    c.edge('d2', 'd1', label='UpConv')
    c.edge('d1', 'output', label='Final Conv')

# --- 上采样与跳跃连接 ---
dot.edge('b', 'd4', label='UpConv')

# 注意：这里的跳跃连接是简化的经典U-Net形式，以保持图的清晰。
# 实际模型是UNet3+，连接方式更复杂。
dot.edge('e4', 'd4', style='dashed', color='grey', constraint='false', label='Skip + Attention')
dot.edge('e3', 'd3', style='dashed', color='grey', constraint='false', label='Skip + Attention')
dot.edge('e2', 'd2', style='dashed', color='grey', constraint='false', label='Skip + Attention')
dot.edge('e1', 'd1', style='dashed', color='grey', constraint='false', label='Skip + Attention')


# 保存图片
dot.render('simplified_unet_architecture', format='pdf', view=False)

print("Simplified architecture diagram saved as 'simplified_unet_architecture'")
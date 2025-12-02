## resources

[git入门](https://zhuanlan.zhihu.com/p/615581394)
[git命令](https://www.runoob.com/note/56524)


## file structure

```plain
/data               # raw data files 
/model              # model files
/preprocess         # preprocessing scripts
/utils              # utility scripts
/save               # saved models and results
/logs               # log files
/share              # shared important files from .gitignore
main.py             # main script to run the project
readme.md           # project documentation
.gitignore          # files to ignore in git
requirements.txt    # project dependencies
```

## todo list

### 数据预处理

- [x] 保留多模态信息的交互数据(`preprocess/raw_data_filter.ipynb`)
- [x] 过滤掉交互次数过少或过多的用户和商品，以提高用户-物品交互图的稠密度(`preprocess/raw_data_filter.ipynb`)
- [ ] 划分训练集、验证集和测试集
- [ ] 提取文本特征
- [ ] 提取图像特征
- [ ] CLIP式的多模态特征融合

#### 提取文本特征

- [ ] 使用预训练语言模型（如BERT）提取商品描述的文本特征$\mathbf{f}_{\text{text}}$

#### 提取图像特征

在`img_process.ipynb`中有将图片从url下载并返回原始向量的代码
- [ ] 查看原始向量是否为图片展开flatten后的像素值，若是则需要进行预处理，以便模型处理
- [ ] 使用预训练卷积神经网络（如ResNet、ViT）提取商品图片的图像特征$\mathbf{f}_{\text{image}}$

#### CLIP式多模态特征融合

训练轻量投影器MLP
$$
\begin{align*}
& \text{Text Projection:} \\
& \mathbf{z}_{\text{text}} = \text{MLP}_{\text{text}}(\mathbf{f}_{\text{text}}) \\
& \text{Image Projection:} \\
& \mathbf{z}_{\text{image}} = \text{MLP}_{\text{image}}(\mathbf{f}_{\text{image}}) \\
& \text{Contrastive Loss:} \\
& \mathcal{L}_{\text{contrastive}} = - \log \frac{\exp(\text{sim}(\mathbf{z}_{\text{text}}, \mathbf{z}_{\text{image}}) / \tau)}{\sum_{i=1}^{N} \exp(\text{sim}(\mathbf{z}_{\text{text}}, \mathbf{z}_{\text{image}_i}) / \tau)} \\
\end{align*}
$$
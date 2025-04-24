




# 余弦距离计算公式变量详解

## 公式定义:


$$cosinedistance=1-\cos{θ}=1-\frac{a⋅b}{∥a∥∥b∥}$$

其中：

$​​a$ 和 $b​$​：待比较的两个向量，可以是文本特征向量、图像像素向量等
。
$​​a⋅b​$​：向量点积（内积），计算方式为各维度分量乘积之和，即


$$a⋅b=\sum_{i=1}^{n}{{{a}_{i⋅}{b}_{i}}}$$


$∥a∥$ $∥b∥$ ：向量的模长（L2范数），计算公式为：

$$∥a∥=\sqrt{\sum_{i=1}^{n}{{{a}_{i}^{2}}}}$$


​​cos(θ)​​：向量夹角的余弦值，反映方向一致性，值域为 [−1,1]

## 关键点解析

​向量标准化​：若向量未归一化，模长会影响结果。归一化后公式简化为：
$cosinedistance=1−{a}_{norm}⋅{b}_{norm}$
其中${a}_{norm}=\frac{a}{∥a∥}$


## 关键代码解析


```
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
```

`np.asarray(a)`将输入数据 a 转换为NumPy数组（ndarray）。若 a 本身是NumPy数组且满足数据类型和存储顺序要求，则直接返回原数组（无复制开销）。


`np.linalg.norm(a, axis=1, keepdims=True)`计算数组的L2范数（默认）。此处指定 axis=1，表示对每行计算范数；keepdims=True 保持结果维度与输入一致，便于广播操作。


```
1.0 - np.dot(a, b.T)
```

功能​：计算矩阵乘法或向量内积，其中 b.T 表示对矩阵 b 进行转置。a 和 b 必须是 numpy 数组。a 的列数需等于 b.T 的行数（即 b 的列数）








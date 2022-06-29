# Adarnn自训练数据集

github地址：https://github.com/newsun-boki/adarnn_server

## I. 训练集的输入方式

+ 3_daily_count数据集共有386天，取预测输入为24天，预测第25天。每隔一天取一次输入模型，总共得到361个样本。每个样本包含的特征feature形状为(24，1)，而预测值label_reg为(1, )
+ 而adarnn原来的weather数据集为5年，每天24小时的6个特征(PM2.5, PM10, S02, NO2, CO, and O3)，取前一天的24小时的形状为(24,6)的样本预测下一天的第一个小时的第一个特征(1, )。
+ 与weather数据集相比，新加入的3_daily_count数据集，取样本时会重复利用同一个样本多次。



## II. 代码更改内容

由于源代码中个人感觉写的不是太规范，weather数据集中涉及大量对于时间的利用，如

```python
start_time = datetime.datetime.strptime(
            '2013-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime(
            '2016-06-30 23:00:00', '%Y-%m-%d %H:%M:%S')
num_day = (end_time - start_time).days
    
```



并且原模型有将每一天24个小时的数据当成一个样本输入的特点。所以数据处理部分大部分重写，取消了各种对时间的操作改用index。

新增[data_count.py](https://github.com/newsun-boki/adarnn_server/blob/main/dataset/data_count.py)文件。

### 3.1 TDC

+ 对于TDC中，原来的weather数据的片段划分是针对小时而不是对于天，原作者直接将处理好的数据用于TDC的划分。但对于3_daily_count数据由于处理过后时间上是重叠的，只能重新读取原始数据对原始数据进行划分。
+ 原TDC中划分所得的结果是类似`2013-03-01 00:00:00`的标准时间，但为了适配不同数据我将其改为最终结果为index。

### 3.2 可视化

关于可视化发现原来的文件中实际上包含`utils/visualize.py`,并且是通过网页端显示。通过vscode进行端口转发可以实现从服务器到自己的电脑查看。

### 3.3 参数

新增参数

```python
parser.add_argument('--dataset', type=str, default='weather')#weather or count
```

```python
if args.dataset == 'weather':
    train_loader_list, valid_loader, test_loader = data_process.load_weather_data_multi_domain(
        args.data_path, args.batch_size, args.station, args.num_domain, args.data_mode)
elif args.dataset == 'count':
    train_loader_list, valid_loader, test_loader = data_process.load_count_data_multi_domain(
        args.data_path, args.batch_size, args.num_domain, args.data_mode)
```



## III. 训练效果

![屏幕截图-2022-06-29-205107](https://cdn.jsdelivr.net/gh/newsun-boki/img-folder@main/mypage/屏幕截图-2022-06-29-205107.300uoe7qxs20.webp)

上图中灰色是原数据集，黄色是训练集，蓝色是验证集，绿色是测试集。

+ 关于训练经验：
  + 该模型对各种参数都(batch_size, lr, num_layer, hidden_size)非常敏感(真的玄)，个人感觉是网络层中没有Normlization的操作，且由于训练样本较小batch_size只能选取较小。
  + TDC的划分，尝试过不同的tdc划分，但效果都不好。特别是自己随意进行划分domain时效果更差。
  + 当把dw(表示损失函数中将不同域化为一致的系数)调确实有用，一定范围内增大效果回好一点。
+ 可以看到他可以预测出为0的点(节假日)，但可能取的验证集和测试集部分确实难以预测。
+ 尝试过对数据整体进行归一化，但结果是效果更差了(原weather数据集做过归一化)

## IV. 超参数

| 超参数名称    | 数值 | 备注                                                  |
| ------------- | ---- | ----------------------------------------------------- |
| batch_size    | 10   | 偏小和偏大都会导致结果很差，需要精确到10              |
| learning_rate | 5e-3 | 同样lr也需要精确的离谱，偏小或大0.001都会导致效果更差 |
| num_layer     | 2    | GRU的层数                                             |
| hidden_size   | 32   | GRU当中每层的隐含层                                   |
| dw            | 5.0  | 损失函数中关于将域归一化的系数，                      |
| loss_type     | adv  | 测定不同域之间距离的函数                              |
| epoch         | 200  | 200次基本收敛                                         |
| data_mode     | tdc  | 可以选择使用tdc划分域或者自己划分域                   |
| num_domain    | 2    | 划分为两段域时效果最好                                |


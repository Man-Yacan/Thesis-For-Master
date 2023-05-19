> ###### Introduction / 引言
>
> 大学期间用来打发无聊时间学的Python没想到竟然在写毕业论文的时候用处这么大，整个硕士论文所做研究，从前期的数据整理、数据分析，到最后的数据可视化我基本上都使用Python来完成，这篇博客就来分享下我毕业论文课题中所做数据分析相关的Python代码。
>
> 本博文所有相关的代码都上传在GitHub仓库：[Data-Analysis-for-Thesis](https://github.com/Man-Yacan/Data-Analysis-for-Thesis)，如果帮到你了，记得给我来个Star☺️，也可以顺便去参观下我的个人博客[亚灿网志](https://blog.manyacan.com/)。

## Python环境配置

### 环境安装

首先是在[Python官网](https://www.python.org/)下载你计算机对应的Python软件，然后安装。安装过程基本都是傻瓜式，不做过多叙述，一路回车即可。

![Python官网](https://image.manyacan.com/202303092145746.png-wm04)

### IDE选择

然后就是IDE（Integrated Development Environment）的配置，最常见的肯定是[VSCode](https://code.visualstudio.com/)，特点是足够轻量化且免费。但是我选择的是[PyCharm](https://www.jetbrains.com/pycharm/)，我选择它的理由是以前学Python的时候就用的它，因此快捷键什么的都比较熟悉。PyCharm正版是收费的，教育版可以申请一年的使用权限，破解版的自行百度即可。

[photos]

![VSCode](https://image.manyacan.com/202303092142909.png-wm04)

![PyCharm](https://image.manyacan.com/202303092143213.png-wm04)

[/photos]

> ##### Tips / 提示
>
> 关于IDE的选择大家没必要纠结，新手不太了解想简单上手的话就选择VSCode；用过PyCharm并且熟练怎么破解安装的那就选择PyCharm。

### 包的安装

1. 数据矩阵分析及处理：`Pandas`、`Numpy`、`Math`、`Scipy`；
2. 绘图可视化：`Matplotlib`、`Seaborn`；
3. 其他包：
   1. `hues`可以在控制台打印出彩色的提示信息，用法也比较简单，[官方手册](https://github.com/prashnts/hues)。

![hues的简单用法](https://image.manyacan.com/202303092203839.png-wm04)

## 数据整理及预处理

### 数据介绍

论文所做内容简单来说就是对不同样本点的多个指标进行相关性分析，具体的数据格式就是：

| 样本点 | 变量1 | 变量2 | 变量n |
| :----: | :---: | :---: | :---: |
|  D-H1  |  12   | 2424  |  ...  |
|  L-N2  |  324  |  232  |  ...  |
|  ...   |  ...  |  ...  |  ...  |

### 基本常量的定义

```python
IMG_TYPE = '.svg'  # 出图格式，我选择矢量图svg
DATA_PATH, EXPORT_PATH, DPI = './data/', './export/', 300  # 数据存储路径、出图写入路径、出图DPI
ROW_NUM, COL_NUM = 5, 3  # 对于多子图的图片，定义默认布图网格为5×3
```

### 文件读取

读取`csv`文件需要使用pandas的`pd.read_csv()`方法，具体的参数有：

- `index_col`：设置行索引为哪一列，可以使用序号或者列名称；
- `sep`：`csv`文件中的分隔符，默认常见的用法都可以自动识别，不需要设置；
- `header`：设置表头，参数为`None`就是没有表头，设置为`n`就是把第`n`行读取为表头；
- `names`：设置列名称，参数为`list`；
- `usecols`：仅读取文件内某几列。

> #### Quote / 参考
>
> 具体用法可以参考李庆辉所著《深入浅出Pandas——利用Python进行数据处理与分析》3.2章 读取CSV（PDF P89）。

### 数据表合并

首先遇到的第一个需求就是，所有样本点的列变量存储在不同的数据表中，比如，样本点的指标分为上覆水的指标与沉积物的指标两部分，分别存储在两个或者多个数据表中，那么如何将两个或者多个数据表进行合并呢？

```python
all_df_0 = pd.concat([gene_abundance, sediment_env_df, eems_df,
                     overlying_water_env_df, sediment_size_df], axis=1)  # 将五个数据表按照行索引合并
```

> #### Quote / 参考
>
> 具体用法可以参考李庆辉所著《深入浅出Pandas——利用Python进行数据处理与分析》7.2章 数据连接pd.concat（PDF P274）。

### 根据行索引为每个样本点设置分类

行索引也就是每个样本点的标记名，分别为“D-H1”、“L-N3”之类的，其中第一个字符的值为“D”、“L”、“W”，分别代表枯水期（Dry Season）、平水期（Level Season）、丰水期（Wet Season）；第二个字符为一个分隔符“-”；第三个字符的值为“N”、“P”、“H”，分别代表三条不同的河流，南淝河、派河、杭埠河；最后的数字代表是第几个样本点。例如：“D-H1”代表枯水期杭埠河第一个样本点、“L-N3”代表平水期南淝河第三个样本点。

那么问题来了，我想要为合并后的数据表新增两列“River”、“Period”，分别来反应这个样本点的属性，应该如何实现呢？

| 样本点 | 变量1 | 变量2 | 变量n |    River     |    Period    |
| :----: | :---: | :---: | :---: | :----------: | :----------: |
|  D-H1  |  12   | 2424  |  ...  | Hangbu River |  Dry Season  |
|  L-N2  |  324  |  232  |  ...  | Nanfei River | Level Season |
|  ...   |  ...  |  ...  |  ...  |     ...      |     ...      |

思路其实也很简单，就是使用`apply`函数分别对每一行（也就是每一个样本点）进行处理，获取该行的行索引，然后对行索引的字符进行判断即可：

```python
all_df['Period'] = all_df.apply(lambda x: 'Dry' if 'D' in x.name else ('Wet' if 'W' in x.name else 'Level'), axis=1) + ' Season'
all_df['River'] = all_df.apply(lambda x: 'Nanfei' if 'N' in x.name else ('Pai' if 'P' in x.name else 'Hangbu'), axis=1) + ' River'
```

> ##### Tips / 提示
>
> 这里使用了Python列表推导式相关的知识，具体的讲解可以看之前的博文：[「Python」列表推导式](https://blog.manyacan.com/archives/1952/)。

在对每一行的样本点添加`River`、`Period`变量后，会有一个问题，`River`、`Period`两列的数据都是`Object`字符串类型。这种数据类型有两个问题：

1. 如果数据矩阵有几十万行，那么这两列会占用很大的内存空间；
2. 对数据进行绘图过程中，我想把`River`变量按照`Nanfei River`、`Pai River`、`Hangbu River`的顺序排列，那么就很麻烦。因为字符串变量默认是按照首字母的顺序来进行排序的，默认排序是`Hangbu River`、`Nanfei River`、`Pai River`。

为了解决这两个问题，我们可以将这两列的数据由原来的`object`类型转换为`Category`类型，`Category`的好处就是，当数据量较大时，可以显著减小数据所占用的内存；第二还可以对数据类型进行排序。

具体的处理方法：

```python
# 对季节、河流两列进行排序，首先定义category类型顺序
river_order = CategoricalDtype(  # 河流的顺序定义为南淝河、派河、杭埠河
    ['Nanfei River', 'Pai River', 'Hangbu River'],
    ordered=True
)
period_order = CategoricalDtype(  # 时期的顺序定义为枯水期、平水期、丰水期
    ['Dry Season', 'Level Season', 'Wet Season'],
    ordered=True
)

# 将两列Object类型数据转换为category类型并排序
all_df['River'] = all_df['River'].astype(river_order)
all_df['Period'] = all_df['Period'].astype(period_order)
all_df = all_df.sort_values(by=['Period', 'River', 'SortNum'])
```

### 异常值处理

#### 缺失值的填充

Pandas中缺失值的填充所用方法时`pd.fillna()`，具体的参数可以填写：

```python
In [16]: pd.DataFrame.fillna
Out[16]: <function pandas.core.frame.DataFrame.fillna(self, value: 'object | ArrayLike | None' = None, method: 'FillnaOptions | None' = None, axis: 'Axis | None' = None, inplace: 'bool' = False, limit=None, downcast=None) -> 'DataFrame | None'>
```

- `value`：直接将缺失值填充为字符串或者数字；
- `method`：填充方式，`method='ffill'` 向前填充，`method='bfill'`向后填充，也就是说用前面的值来填充NA或用后面的值来填充NA。

另外，在使用读取`pd.read_csv()`读取`csv`文件的时候，也可以通过参数：

- `na_values=None`
- `keep_default_na=True`
- `na_filter=True`

的设置来对NA值进行过滤或者识别。

#### 删除缺失值

使用`pd.DataFrame.dropna()`方法完成缺失值的删除：

```python
In [17]: pd.DataFrame.dropna
Out[17]: <function pandas.core.frame.DataFrame.dropna(self, axis: 'Axis' = 0, how: 'str' = 'any', thresh=None, subset: 'IndexLabel' = None, inplace: 'bool' = False)>
```

通过参数`how`的属性值来设置：

- `any`：当每一行有一个缺失值时就删除这一行；
- `all`：当一行所有的数据都时缺失值时再删除这一行。

#### 重复值的删除

使用`pd.DataFrame.drop_duplicates()`方法完成缺失值的删除：

```python
In [18]: pd.DataFrame.drop_duplicates
Out[18]: <function pandas.core.frame.DataFrame.drop_duplicates(self, subset: 'Hashable | Sequence[Hashable] | None' = None, keep: "Literal['first'] | Literal['last'] | Literal[False]" = 'first', inplace: 'bool' = False, ignore_index: 'bool' = False) -> 'DataFrame | None'>
```

通过参数`keep`的属性值来设置：

- `first`：所有重复行删除，保留第一行；
- `last`：所有重复行删除，保留最后一行。

## 数据处理与可视化

### 绘图前的小准备

#### 画图格式的定义

如何在`Matplotlib`中显示中文：

```python
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False
```

如何定义`Seaborn`绘图的默认字体属性：

```python
sns.set(
    style='darkgrid',  # 绘图风格
    font='Times New Roman',  # 默认字体
    font_scale=3  # 默认字体比例
)
```

#### 如何实现子图编号

```py
CHAR = [chr(i) for i in range(97, 123)]  # 获取26个英文字母，用于给子图编号
```

定义一个26个英文字母的`list`，循环绘制子图的时候直接调用即可。

#### 重复代码的打包

每次进行数据分析我都会新建一个`.ipynb`文件，而数据分析前都需要经过数据表合并、数据清洗等工作，那么最好的方式其实是将数据分析前的准备工作进行一个打包，然后在`.ipynb`文件的第一行引入包即可。

例如：我新建一个`ResearchMain.py`文件，然后将所有数据表合并、数据清洗的代码都放在这个文件里：

```python
# 引入数据分析常用的包
...

# 读取文件
...

# 合并文件
...

# 数据表清洗
...
```

然后在每次新建`.ipynb`文件进行数据分析时，我都会在第一行使用：

```python
from ResearchMain import *
```

来引入所有`ResearchMain.py`文件中定义的变量与方法。

### QPCR标准曲线可视化

> ##### Tips / 提示
>
> 什么是标准曲线？
>
> 简单来说，自变量x与因变量y之间存在某种线性关系——$y=ax+b$，那么我们可以通过多次改变自变量x的值，然后观察y的值并记录，得到几组对应的x~1~、x~2~、x~3~、x~4~、x~5~、...与y~1~、y~2~、y~3~、y~4~、y~5~、...，那么我们就可以通过得到的这几组数据来对自变量x与因变量y进行线性拟合，从而得到一个标准曲线$y=ax+b$，有了标准曲线之后，我们就可以直接输入任意的自变量x值，计算出因变量y的值。

在`Numpy`中，拟合函数直接有现成的，可以直接调用：

```python
In [37]: x, y = [1, 2, 3, 4, 5], [2, 4, 6, 8, 11] # 需要进行拟合的自变量与因变量

In [38]: np.polyfit(x, y, 1)  # 对自变量x与因变量y进行拟合，且拟合为一次函数
Out[38]: array([ 2.2, -0.4])  # 拟合结果为y=2.2x-0.4
```

拟合完毕如何拼接拟合方程？

```python
In [25]: fitting_arr = np.polyfit(x, y, 1)

In [26]: fitting_equ = np.poly1d(fitting_arr)  # 获取拟合方程

In [27]: fitting_equ([6, 7, 8, 9, 10])  # 利用拟合方程计算任意自变量对应的因变量
Out[27]: array([12.8, 15. , 17.2, 19.4, 21.6])

In [28]: fitting_equ(11)
Out[28]: 23.799999999999997  # 23.799999999999997=11*2.2-0.4
```

获取拟合的$R^2$：

```python
In [43]: np.corrcoef(y, fitting_equ(x))[0, 1] ** 2
Out[43]: 0.9918032786885246
```

> $R^2$的计算原理其实就是把自变量带入拟合方程，将计算出来的因变量与原始的因变量进行比较，计算其相关性。

搞懂上面的原理之后，进行标准曲线的可视化其实就很简单了。

![](https://image.manyacan.com/202303102036795.png-wm04)

图中可以看出，还生成了一个拼接的一元一次方程，方程的拼接可以直接用我写好的函数，函数的具体用法以及讲解已经在注释里说的很清楚了：

> ##### Tips / 提示
>
> 函数的主要作用就是传入`np.polyfit(X, Y, DEG)`返回的`list`对象，比如返回的是`[2, 3, -4]`，那么就输出`2x^2+3x-4`，可以自动识别n元函数。

```python
def make_fit_equ_str(paras):
    """
    拼接拟合方程
    :param paras: 传入参数为np.polyfit(X, Y, DEG)返回的对象
    :return: 返回一个拼接拟合方程的字符串(LaTex格式)
    """

    fit_equ_str = ''

    for i in range(len(paras)):
        # 遍历常数项的过程中，主要需要进行两步“组装”：①如果该常数项不是第一个常数项，且该常数项大于0，需要转化为字符串并在前面添加一个“+”；
        if paras[i]:  # 如果常数项不为0(为0直接跳过这一项)
            cur_item = str(round(paras[i], 4))  # 每个常数项都保留两位小数，并转化为字符串
            # 如果不是第一个常数项，且该常数项大于0，需要转化为字符串前面添加一个“+”
            if paras[i] > 0 and fit_equ_str != '':  # 注意两种特殊情况：[0, 0, -5, 7]、[0, -5, 7]如何排除
                cur_item = '+' + cur_item

            # ②为每一个常数项添加x的幂指数
            if i == (len(paras) - 1) and cur_item != '0':  # 最后一个项只有一个常数，不需要添加x了（如果为0就不用+了）
                fit_equ_str += cur_item
            else:  # 前面的每一项都需要在最后加上一个x^幂数
                idempotent_num = len(paras) - 1 - i  # 对应的幂指数
                if idempotent_num == 1:  # 拟合为一次多项式
                    fit_equ_str += cur_item + 'x'
                else:  # 拟合为二次及二次以上的多项式
                    fit_equ_str += cur_item + 'x^' + str(idempotent_num)  #方程拼接

    # fit_equ_str = '$' + fit_equ_str + '$'  # 将多项式转化为LaTex公式
    return fit_equ_str
```

### QPCR数据处理

上一步的操作已经制作出来的功能基因的标准曲线，接下来就是把仪器测得的CT值（自变量）转化为丰度值（因变量），就是带进拟合的标准曲线计算下就OK了。

具体绘图代码直接看GitHub代码即可，没有什么难度。

![](https://image.manyacan.com/202303102103212.png-wm04)

### 三维荧光数据可视化

#### 读取数据表

使用日立F-7000荧光光谱仪对沉积物中溶解性有机质（Dissolved Organic Matter, DOM）结构特征和组成成分进行表征。仪器得到的数据是`.txt`格式，且有用的数据表是从`Data Points`这一行后面开始的。

![得到的txt文件](https://image.manyacan.com/202303102114479.png-wm04)

所以说，我们要先读取`.txt`文件，循环读取每一行，直到读取到`Data Points`这一行，说明已经到数据表了。

```python
def get_skip_rows(path):
    """
    读取txt文件，并在文件中查找含有'Data Points'的行，数据矩阵就在这一行的下面
    :param path: 文件路径
    :return: 数据矩阵开始的行号
    """
    f = open(path)
    for index, line in enumerate(f.readlines()):
        if 'Data Points' in line:
            return index + 1
```

这个函数的作用就是输入`.txt`文件的路径，然后会返回需要的数据表是在第几行开始的。

然后使用`pd.read_table()`方法读取`.txt`，并通过设置`skiprows`的值，来跳过前面无用的数据。例如利用`get_skip_rows()`函数获取到`.txt`文件中数据表从第156行开始：

```python
df = pd.read_table(search_info['Path'], skiprows=156, index_col=0)
```

这样的话就可以完美跳过`.txt`文件前面无用的数据，直接读取所需的数据表。

#### 消除瑞利散射

![瑞利散射消除前后对比](https://image.manyacan.com/202303102127862.png-wm04)

瑞利散射的消除其实很简单，观察数据表就可以看出来，瑞利散射其实就是不该出现在某个区间内数据峰，我们只需要慢慢根据$E_x$与$E_m$的设置范围来进行消除就行了：

```python
for i in range(modify_df.shape[0]):  # 遍历ex
    for j in range(modify_df.shape[1]):  # 遍历em
        ex = 200 + i * 5  # ex范围为200~450，间隔为5
        em = 280 + j * 5  # em范围为280~550，间隔为5
        if ex + 455 > em > ex + 145:
            modify_df.iloc[i, j] = 0  # 不能使用0, 因为0是有意义的数据
```

#### 光谱图的分区

直接使用：

```python
cur_ax.axvline(250, color='white', linestyle='--', linewidth=5)  # 垂直线
cur_ax.axhline(330, color='white', linestyle='--', linewidth=5)  # 水平线
```

就可以在图上画线了。

![](https://image.manyacan.com/202303102134952.png-wm04)

需要讲解的其实就这几个部分，别的直接看代码就行了。

### 粒径数据可视化

沉积物粒径百分比分布使用Malvern Mastersizer 2000型激光粒度仪进行分析。得到数据后需要手动整理为`.csv`格式。

![数据格式](https://image.manyacan.com/202303102140733.png-wm04)

首先来讲解下数据格式，每一列代表一个样本，每一行代表对应粒径所占百分比。例如图中红方框所示就是代表`D-N4`样本点对应粒径为`0.955 μm`颗粒占比为`0.03%`。

#### 清除空行

上图中可以看出，数据有很多空行，那么首先第一步就是清除掉这些空行：

```python
df.dropna(how='all', inplace=True)  # 删除缺失值（行全为空）
df.dropna(how='all', axis=1, inplace=True)  # 删除缺失值（列全为空）
```

绘图还是直接看代码吧。

![](https://image.manyacan.com/202303102146411.png-wm04)

### 常规指标数据可视化

```python
# 子图摆放设置
ROW_NUM = 2
COL_NUM = int(len(df.columns) / ROW_NUM)

# 绘图设置
sns.set(
    style='darkgrid',
    font='Times New Roman',
    font_scale=2
)


# 布局设置
plt.figure(dpi=DPI)
fig, ax_arr = plt.subplots(ROW_NUM, COL_NUM, sharex='col', figsize=(20, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# 循环绘图
for index, s in enumerate(df.iteritems()):
    cur_df = pd.concat([s[1], affix_info], axis=1)
    # 计算行列数
    row_num = index // COL_NUM
    col_num = index - row_num * COL_NUM
    cur_ax = ax_arr[row_num][col_num]
    cur_plot = sns.boxplot(  # 绘图
        x="Period",  # X轴
        y=s[0],  # Y轴
        hue="River",  # 颜色分类
        data=cur_df,  # 数据表
        dodge=True,  
        palette="Set1",  # 配色
        ax=cur_ax  # 绘图坐标轴
    )
    # 子图图名、x轴、y轴、图例
    cur_ax.tick_params(axis='x', rotation=90)
    if len(df.columns) - (index + 1) >= COL_NUM:
        cur_plot.set_xlabel(None)
    cur_plot.set_title(f'({CHAR[index]}) ${s[0][2:]}$ ({unit[s[0]]})',fontproperties="Times New Roman", fontsize=20)
    cur_ax.get_legend().remove()  # 移除子图图例
    cur_plot.set_ylabel(None)

# 子图的图例相同，获取最后一个子图的图例
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, ncol=3, loc='lower center',
           bbox_to_anchor=(0.5, -0.16))

# 保存图片
# fig.savefig(EXPORT_PATH + '沉积物环境因子时空变化' + IMG_TYPE, dpi=DPI, bbox_inches='tight' )
```

![](https://image.manyacan.com/202303102321718.png-wm04)

## 数据分析

### 主成分分析

主成分分析（Prin­ci­pal Com­po­nent Analy­sis, PCA）的具体逻辑及Python实现方法可以看：[PCA主成分析原理、理解和代码实现](https://blog.manyacan.com/archives/1946/)。

### 聚类分析

#### K-means聚类

```python
class KMeans(object):
    """
    KMeans算法类
    """

    def __init__(self, df, K, iterate_num=50):
        """
        类或对象的初始化
        :param df: 输入矩阵 m×n
        :param K: 聚类蔟数
        :param iterate_num: 迭代循环次数，默认为50次
        """
        self.df = df
        self.K = K
        self.iterate_num = iterate_num
        self.centers = None  # 存放簇中心点，第一次随机选择，之后通过迭代不断更新

    def run(self):
        """
        类、对象入口函数
        """
        hues.info(f'{self.__class__.__name__}已运行...')
        # 1、随机选择K个簇中心点(存入self.centers中)
        self.df['near_center_id'] = np.nan  # 在df中添加一列作为分簇的依据
        self.make_random_center()
        # 2、点归属——求每个点到暂定簇中心的最近距离
        self.calc_distance()
        # 3、更根据分类，更新簇中心
        self.update_centers()
        # 4、迭代循环
        self.do_iteration()
        # 5、绘制聚类结果（仅针对二维数据）
        self.plot_scatter_2D()

    def make_random_center(self):
        """
        从df中随机选择K个点作为K个簇中心点
        """
        random_ids = np.random.permutation(self.df.shape[0])  # 获取传入的DataFrame有多少行数据，对DataFrame行值进行洗牌打乱
        self.centers = self.df.iloc[:, :-1].loc[random_ids[:self.K], :]  # 从df中随机获取K行数据作为随机开始点

    def calc_distance(self):
        """
        计算每个点到簇中心的距离
        df: m*n centers: K*n  dis_df 各个点到各个暂定簇中心的距离 m*K
        """
        dis_df = np.zeros((self.df.shape[0], self.K))  # 定义一个用于存放每个点到每个簇中心距离的df
        # 求出每个点到每个暂定簇中心的欧氏距离
        for i in range(len(self.df)):
            for j in range(self.K):
                dis_df[i, j] = np.sqrt(sum((self.df.iloc[i, :-1] - self.centers.iloc[j]) ** 2))

        # 对比每个点到每个簇中心的距离，将距离某点距离最小的簇中心的ID记录，作为最后分分类的依据
        self.df['near_center_id'] = np.argmin(dis_df, axis=1)

    def update_centers(self):
        """
        在计算距离之后，需要根据新的near_center_id分类来计算新的K个簇中心
        """
        for i in range(self.K):  # 循环更新K个簇中心
            self.centers.iloc[i, :] = self.df[self.df['near_center_id'] == i].iloc[:, :-1].mean()

    def do_iteration(self):
        for _ in range(self.iterate_num):  # 进行迭代循环
            self.calc_distance()  # 计算距离
            # 在每次更新centers位置前，对centers深拷贝，如果更新后没变化，说明迭代完成
            old_centers = self.centers.copy(deep=True)
            # print('Old Centers:')  # DEBUG
            # print(old_centers)
            self.update_centers()  # 更新簇中心
            # print('New Centers:')
            # print(self.centers)
            # print("=" * 20)
            if old_centers.equals(self.centers):
                hues.success(f'结束运行, 共进行了【{_}】次迭代.')
                break

    def plot_scatter_2D(self):
        """
        如果传入的df（m×n）是二维数据（n=2），可以直接调用这个函数来实现画图，如果数据维度大于2（n＞2），则需要进行降维操作后绘制图形
        绘制二维平面图
        """
        get_ipython().run_line_magic('matplotlib', 'notebook')
        sns.scatterplot(x=self.df.iloc[:, 0], y=self.df.iloc[:, 1], hue=self.df['near_center_id'])
        sns.scatterplot(x=self.centers.iloc[:, 0], y=self.centers.iloc[:, 1], marker="*", s=500)
```

#### 层次聚类

层次聚类（Hierarchical Clustering）是聚类算法的一种，通过计算不同类别数据点间的相似度来创建一棵有层次的嵌套聚类树。在聚类树中，不同类别的原始数据点是树的最低层，树的项层是一个集类的根节点。基于层次的聚类算法可以是凝聚的（Agglomerative）或者分裂的（Divisive），取决于层次的划分是“自底向上”还是“自项向下”。

```python
# 引包
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  # 计算距离、绘图、得到标签结果

# 加载数据集
iris_0 = sns.load_dataset('iris')
iris = iris_0.copy()

# 数据集本身是有标签的，弹出species列，假装没有标签
species = list(iris.pop('species'))  

# 进行层次聚类
X = iris.values  # 提取数据集中的元素矩阵，格式为numpy.ndarray
mergings = linkage(X, method='average')  # method参数：single: 两个组合点中的最近点, complete：两个组合点中的最远点，average：两个组合点的平均值的距离

# 树状图展示
fig = plt.figure(figsize=(20, 8))
dendrogram(mergings, labels=species, leaf_rotation=90, leaf_font_size=6)
plt.show()
```

### Mantel Test

见博文：[Mantel Test算法原理讲解、代码实现、绘图可视化](https://blog.manyacan.com/archives/2028/)。
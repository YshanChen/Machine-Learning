### F1 Score 

#### 1. F1 Score

$$F1\ Score = 2*(\frac{Precision\  *\ Recall}{Precision\ +\ Recall})$$

$$Precision=\frac{TP}{TP+FP}$$

$$Recall = \frac{TP}{TP+FN}$$



#### 2. F1 Score micro & macro

'$$micro$$': 通过先计算总体的TP，FN和FP的数量，再计算F1。

'$$macro$$': 分布计算每个类别的F1，然后做平均（各类别F1的权重相同）。



#### 3. 举例

$$Y\_true=[1,1,1,1,1,2,2,2,2,3,3,3,4,4]$$

$$Y\_pred=[1,1,1,0,0,2,2,3,3,3,4,3,4,3]$$

|      | 1类  | 2类  | 3类  | 4类  | 总数 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| $TP$ |  3   |  2   |  2   |  1   |  8   |
| $FP$ |  0   |  0   |  3   |  1   |  4   |
| $FN$ |  2   |  2   |  1   |  1   |  6   |

$micro:$

$$Precision=8/(8+4)=0.666;\ \ \ Recall=8/(8+6)=0.571$$

$$F1\_micro=0.6153$$



$macro:$

$$F1\ for\ class=1:\ \ Precision=3/(3+0)=1;\ \ Recall=3/(3+2)=0.6;\ \ F1=0.75$$

...

$$F1\_macro=mean(F1\_class1, F1\_class2, F1\_class3, F1\_class4)$$





参考：https://www.cnblogs.com/techengin/p/8962024.html








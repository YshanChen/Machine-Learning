## 逻辑回归

### 1. 逻辑回归

>给定 $y \in \{1,0\}$
>
>$$P(Y=1|x) = \hat{y} = \sigma(w^T x+b) = \frac{1}{1+e^{-(w^T x+b)}}$$
>
>$$P(Y=0|x) = 1-\hat{y} = 1-\sigma(w^Tx+b) = \frac{e^{-(w^T x+b)}}{1+e^{-(w^T x+b)}} = \frac{1}{1+e^{w^T x+b}} $$
>
>$$\sigma(z) = \frac{1}{1+exp(-z)}$$



### 2. 损失与代价函数

>**推导：**
>
>**基于极大似然估计$MLE$**
>
>$$if\ y_i\  =\ 1:\ \ \ p(y|x)=\hat{y_i}$$
>
>$$if\ y_i\  =\ 0:\ \ \ p(y|x)=1-\hat{y_i}$$
>
>$$So \ \ p(y|x) = \hat{y_i}^{y_i}·(1-\hat{y_i})^{(1-y_i)}$$ 
>
>**找到参数使得全部样本出现的概率最大：**
>
>$$\max \limits_{w,b} \prod p(y|x) = \prod (\hat{y_i}^{y_i}·(1-\hat{y_i})^{(1-y_i)}) = \sum y_i log(\hat{y_i})+(1-y_i)log(1-\hat{y_i})​$$
>
>转为最小化：
>
>$$\min \limits_{w,b} -\prod p(y|x) = -\prod (\hat{y_i}^{y_i}·(1-\hat{y_i})^{(1-y_i)}) = \sum -(y_i log(\hat{y_i})+(1-y_i)log(1-\hat{y_i}))$$ 
>
>可得 损失函数：
>
>$$交叉熵损失函数： \ \ \ L(y_i,\hat{y_i}) _{w,b} = -(y_i log(\hat{y_i})+(1-y_i)log(1-\hat{y_i}))$$ 
>
>可得 代价函数：
>
>$$J(w,b) = \frac{1}{N} \sum \limits_{i=1}^{N}L(y_i,\hat{y_i}) = - \frac{1}{N} \sum \limits_{i=1}^{N} [y_i log(\hat{y_i})+(1-y_i)log(1-\hat{y_i})]$$



### 3. 求解$w,b$

>**梯度下降：**
>
>$$\begin{align}
>&repeat \{ \\
>& \ \ \ \ \ \ \ & w:=w-\alpha \frac{\partial J(w,b)}{\partial w} \\   
>& \ \ \ \ \ \ \ & b:=b-\alpha \frac{\partial J(w,b)}{\partial b} \\   
>&\}
>\end{align}$$



### 4. 梯度推导

https://my.oschina.net/chensanti234/blog/1563455


# 多元函数的泰勒(Taylor)展开式

实际优化问题的目标函数往往比较复杂。为了使问题简化，通常将目标函数在某点附近展开为泰勒(Taylor)多项式来逼近原函数。

- **一元函数在点$x_k$处的泰勒展开式为：** 

  $$f(x) = f(x_k)+(x-x_k)f'(x_k)+\frac{1}{2!}(x-x_k)^2f''(x_k)+o^n$$

* **二元函数在点$(x_k,y_k)$处的泰勒展开式为：** 
  $$f(x,y)=f(x_k,y_k)+(x-x_k)f'_x(x_k,y_k)+(y-y_k)f'_y(x_k,y_k)\\ +\frac1{2!}(x-x_k)^2f''_{xx}(x_k,y_k)+\frac1{2!}(x-x_k)(y-y_k)f''_{xy}(x_k,y_k)\\ +\frac1{2!}(x-x_k)(y-y_k)f''_{yx}(x_k,y_k)+\frac1{2!}(y-y_k)^2f''_{yy}(x_k,y_k)\\ +o^n$$
* **多元函数(n)在点处的泰勒展开式为：** 
  $$f(x^1,x^2,\ldots,x^n)=f(x^1_k,x^2_k,\ldots,x^n_k)+\sum_{i=1}^n(x^i-x_k^i)f'_{x^i}(x^1_k,x^2_k,\ldots,x^n_k)\\ +\frac1{2!}\sum_{i,j=1}^n(x^i-x_k^i)(x^j-x_k^j)f''_{ij}(x^1_k,x^2_k,\ldots,x^n_k)\\ +o^n$$

**把Taylor展开式写成矩阵的形式：**

$$f(\mathbf x) = f(\mathbf x_k)+[\nabla f(\mathbf x_k)]^T(\mathbf x-\mathbf x_k)+\frac1{2!}[\mathbf x-\mathbf x_k]^TH(\mathbf x_k)[\mathbf x-\mathbf x_k]+o^n$$

**其中：**

$$H(\mathbf x_k)= \left[ \begin{matrix} \frac{\partial^2f(x_k)}{\partial x_1^2} & \frac{\partial^2f(x_k)}{\partial x_1\partial x_2} & \cdots & \frac{\partial^2f(x_k)}{\partial x_1\partial x_n}  \\ \frac{\partial^2f(x_k)}{\partial x_2 \partial x_1} & \frac{\partial^2f(x_k)}{\partial x_2^2} & \cdots & \frac{\partial^2f(x_k)}{\partial x_2\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2f(x_k)}{\partial x_n\partial x_1} & \frac{\partial^2f(x_k)}{\partial x_n\partial x_2} & \cdots & \frac{\partial^2f(x_k)}{\partial x_n^2} \\ \end{matrix} \right]$$

![img](https://raw.githubusercontent.com/YshanChen/Machine-Learning/master/JPG/%E6%B3%B0%E5%8B%92%E5%B1%95%E5%BC%80.jpg)


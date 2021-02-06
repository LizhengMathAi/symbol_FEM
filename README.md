<h4><b>Introduction</b></h4>
The finite element method (FEM) is the most widely used method for solving problems of engineering and mathematical models.
Typical problem areas of interest include the traditional fields of structural analysis, heat transfer, fluid flow, mass transport, and electromagnetic potential.
The FEM is a particular numerical method for solving partial differential equations.<br>
<font color="red">
However, traditional softwares can't solve complex PDEs because it's a hard work about deducing the derivative and integral formulations of some basis functions in any order.
Although, it's a great challenge for constructing a wildly used FEM System, but using symbolic computation to achieve it is possible.
To construct this symbolic computation FEM System, two sub-targets are needed.
</font><br>
<ol>
    <li>For a given type of interpolation(for instance Lagrange or Hermite), the machine should have the ability to deduce its basis functions in any grid.</li>
    <li>For a serial of shape functions <a><img src="https://github.com/LizhengMathAi/symbol_FEM/blob/main/src/1.png" /></a> in grid element $T$ and constant arrays $(\boldsymbol{a}, \boldsymbol{b}, \cdots)$, the machine should have the ability to deduce the result of following formula
    <p style="font-size:1.5vw">
    \begin{equation*}
        \int_{T}
        \left[
        \left( \prod_{\alpha_1 \cdots \alpha_{n_{\alpha}}} \boldsymbol{a}_{\alpha_1 \cdots \alpha_{n_{\alpha}}} \frac{\partial^{\alpha_1 + \cdots + \alpha_{n_{\alpha}}} \phi}{\partial \boldsymbol{x}_1^{\alpha_1} \cdots \partial \boldsymbol{x}_n^{\alpha_{n_{\alpha}}}} \right)
        \left( \prod_{\beta_1 \cdots \beta_{n_{\beta}}} \boldsymbol{b}_{\beta_1 \cdots \beta_{n_{\beta}}} \frac{\partial^{\beta_1 + \cdots + \beta_{n_{\beta}}} \psi}{\partial \boldsymbol{x}_1^{\beta_1} \cdots \partial \boldsymbol{x}_n^{\beta_{n_{\beta}}}} \right)
        \cdots
        \right]
        \mathrm{d} \boldsymbol{\sigma}
    \end{equation*}
    </p>
    </li>
</ol>
In order to build the framework symbolic computation for FEM system, this software must consist of four parts
<ol>
    <li>Sparse array module, this part should support</li>
    <ul>
        <li>Slice operator;</li>
        <li>+ - * operators;</li>
        <li>Reduce operators;</li>
        <li>Einstein summation convention.</li>
    </ul>
    <li>Polynomial $n$-dimensional array symbolic computation module, this part should support</li>
    <ul>
        <li>the calling method;</li>
        <li>Slice operator;</li>
        <li>+ - * operators;</li>
        <li>Shape operators;</li>
        <li>Derivative operators in any order and dimension;</li>
        <li>Integral operators in any dimension and simplex.</li>
    </ul>
    <li>Mesh grid module, create isotropic triangulation grid.</li>
    <li>Interpolation formulation deducing module. Given a type of interpolation, for example Lagrange or Hermite, this part should supply</li>
    <ul>
        <li>Degree of freedoms in each simplex;</li>
        <li>Position of nodes in simplices, it relies to the interpolation formulation.</li>
        <li>Shape functions in simplices;</li>
        <li>Map of the index of basis function in simplices to the index in global domain;</li>
    </ul>
</ol>
The process of solving PDEs in this system consists of five parts
<ul>
    <li>Using <font style="color:red">mesh module</font> to generate isotropic grid.</li>
    <li>Using <font style="color:red">symbol module</font> and grid to deduce interpolation formulations and shape functions.</li>
    <li>Using <font style="color:red">correct weak formulations</font> to generate nonlinear equations.</li>
    <li>Using <font style="color:red">optimizer</font> to estimate the value in grid nodes.</li>
    <li>Using the deduced <font style="color:red">interpolation formulations</font> to estimate the $L_n$ error.</li>
</ul>
To be cautions, before using this system, check the <font style="color:red">weak formulations</font> and the <font style="color:red">type of interpolations</font> correctly are the most important things!

Evaluating a Learning Algorithm
===============================

Motivational example
--------------------

**Debugging a learning algorithm** Suppose you have implemented
regularized liner regression to predict housing prices. However, when
you test your hypothesis in a new set of houses, you find that it makes
unacceptably large errors in its prediction. What should you try next?

-   Get more training examples

-   Try smaller sets of features

-   Try getting additional features

-   Try adding polynomial features ($$x_1^2, x_2^2, x_1x_2, etc.$$)

-   Try decreasing

-   Try increasing

Evaluating a Hypothesis
-----------------------

Once we have done some trouble shooting for error in our predictions:

-   Getting more training examples

-   Trying smaller sets of features

-   Trying additional features

-   Trying polynomial features

-   Increasing or decreasing

we can move on to evaluate our new hypothesis.

A hypothesis may have a low error for the training examples but still be
inaccurate (because of overfitting). Thus, to evaluate a hypothesis,
given a dataset of training examples, we can split up the data into two
sets: a **training set** and a **test set**. Typically, the training set
consists of 70% of your data and the test set is the remaining 30%.

The new procedure using these two sets is then:

1.  Learn and minimize $$J_{train}(\Theta)$$ using the training set

2.  Compute the test set error $$J_{test}(\Theta)$$

### The test set error

1.  For linear regression:
    $$J_{test}(\Theta)=\dfrac{1}{2m_{test}}\sum^{m_{test}}_{i=1}(h_{\Theta}(x^{(i)}_{test})-y^{(i)}_{test})^2$$

2.  For classification $$\sim$$ Misclassification error (aka 0/1
    misclassification error):

$$\boxed{err(h_\Theta(x),y) =
\begin{cases}
1, & \text{if } h_{\Theta{}}(x) \geq{} 0.5\ and\ y = 0\ or\ h_{\Theta{}}(x) < 0.5\ and\ y=1 \\
0 & \text{} otherwise
\end{cases}}$$

This gives us a binary 0 or 1 error result based on a misclassification.
The average test error for the test set is:
$$Test Error = \dfrac{1}{m_{test}}\sum^{m_{test}}_{i=1}err(h_{\Theta{}}(x^{(i)}_{test}),y^{(i)}_{test})$$
This gives us the proportion of the test data that was misclassified.

Model Selection and Train/Validation/Test Sets
----------------------------------------------

Just because a learning algorithm fits a training set well, that does
not mean it is a good hypothesis. It could over fit and as a result your
predictions on the test set would be poor. The error of your hypothesis
as measured on the data set with which you trained the parameters will
be lower than the error on any other data set.

Given many models with different polynomial degrees, we can use a
systematic approach to identify the 'best' function. In order to choose
the model of your hypothesis, you can test each degree of polynomial and
look at the error result.

One way to break down our dataset into the three sets is:

-   Training set: 60%

-   Cross validation set: 20%

-   Test set: 20%

We can now calculate three separate error values for the three different
sets using the following method:

1.  Optimize the parameters in using the training set for each
    polynomial degree.

2.  Find the polynomial degree d with the least error using the cross
    validation set.

3.  Estimate the generalization error using the test set with
    $$J_{test}(\Theta^{(d)})$$, (d = theta from polynomial with lower
    error);

This way, the degree of the polynomial d has not been trained using the
test set.

Bias vs. Variance
=================

Diagnosing Bias vs. Variance
----------------------------

In this section we examine the relationship between the degree of the
polynomial d and the underfitting or overfitting of our hypothesis.

-   We need to distinguish whether **bias** or **variance** is the
    problem contributing to bad predictions.

-   High bias is underfitting and high variance is overfitting. Ideally,
    we need to find a golden mean between these two.

The training error will tend to **decrease** as we increase d up to a
point, and then it will **increase** as d is increased, forming a convex
curve.

**High bias (underfitting)**: both $$J_{train}(\Theta{})$$ and
$$J_{CV}(\Theta{})$$ will be high. Also,
$$J_{CV}(\Theta{}) \approx J_{train}(\Theta{})$$.

$$J_{CV}(\Theta{})$$: CV stands for cross-validation.

**High variance(overfitting)**: $$J_{train}(\Theta{})$$ will be low and
$$J_{CV}(\Theta{})$$ will be much greater than $$J_{train}(\Theta{})$$.

This is summarized in the figure below:

![image](UnderAndOverfitting)

Regularization and Bias/Variance
--------------------------------

**Note**: \[The regularization term below and through out the video
should be $$\dfrac{\lambda{}}{2m}\sum^{n}_{j=1}\Theta{}_j^2$$ and **NOT**
$$\dfrac{\lambda}{2m}\sum^m_{j=1}\Theta^2_j$$\]

![image](LinearRegressionWithRegularization){width="15cm"}

In the figure above, we see that as increases, our fit becomes more
rigid. On the other hand, as approaches 0, we tend to overfit the data.
So how do we choose our parameter to get it 'just right'. In order to
choose the model and the regularization term , we need to:

1.  Create a list of lambdas (i.e.
    $$\lambda{} \in \{ 0, 0.01, 0.02, 0.04, 0.08. 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24 \}$$)

2.  Create a set of models with different degrees or any other variants.

3.  Iterate through the s and for each go through all the models to
    learn some .

4.  Compute the cross validation error using the learned (computed with
    ) on the $$J_{CV}(\Theta)$$ **without** regularization $$\lambda = 0$$.

5.  Select the best combination that produces the lowest error on the
    cross validation set.

6.  Using the best combination and , apply it on $$J_{test}(\Theta)$$ to
    see if that has a good generalization of the problem.

Learning Curves
---------------

Training an algorithm on a very small number of data points (such as 1,
2, or 3) will easily have 0 errors because we can always find a
quadratic curve that touches exactly this number of points. Hence:

-   As the training set gets larger, the error for a quadratic function
    increases.

-   The error value will plateau out after a certain m, or training set
    size.

**Experiencing high bias**:

**Low training set size**: causes $$J_{train}(\Theta{})$$$ to be low and
$$J_{CV}(\Theta{})$$ to be high.

**Large training set size**: causes both $$J_{train}(\Theta{})$$ and
$$J_{CV}(\Theta{})$$ to be high with
$$J_{train}(\Theta{}) \approx J_{CV}(\Theta{})$$.

If a learning algorithm is suffering from high bias, getting more
training data will not **(by itself)** help much.

![image](BiasVsVariance)

**Experiencing high variance**: **Low training set size**:
$$J_{train}(\Theta{})$$ will be low and $$J_{CV}(\Theta{})$$ will be high.
**Large training set size**: $$J_{train}(\Theta{})$$ increases with
training set size and $$J_{CV}(\Theta{})$$ continues to decrease without
leveling off. Also, $$J_{train}(\Theta{}) < J_{CV}(\Theta{})$$ but the
difference between them remains significant.

If a learning algorithm is suffering from **high variance**, getting
more training data is likely to help.

![image](BiasVsVarianceB)

Motivational example - revisited
--------------------------------

**Debugging a learning algorithm** Suppose you have implemented
regularized liner regression to predict housing prices. However, when
you test your hypothesis in a new set of houses, you find that it makes
unacceptably large errors in its prediction. What should you try next?

-   Get more training examples $$\rightarrow{}$$ fixes high variance; in
    other cases might not help at all.

-   Try smaller sets of features $$\rightarrow{}$$ fixes high variance,
    does not help with high bias.

-   Try getting additional features $$\rightarrow{}$$ fixes high bias
    problems to make hypothesis better able to fit the feature set

-   Try adding polynomial features ($$x_1^2, x_2^2, x_1x_2, etc.$$)
    $$\rightarrow{}$$ high bias problem

-   Try decreasing $$\rightarrow{}$$ quick and easy to try, fixes high
    bias

-   Try increasing $$\rightarrow{}$$ quick and easy to try, fixes high
    variance

Neural networks and overfitting, or: Diagnosing neural networks
---------------------------------------------------------------

-   A neural network with fewer parameters is **prone to underfitting**.
    It is also **computationally cheaper**.

-   A large neural network with more parameters is **prone to
    overfitting**. Is is also **computationally expensive**. In this
    case you can use regularization (increase ) to address overfitting.

Using a single hidden layer is a good starting default. You can train
your neural network on a number of hidden layers using your cross
validation set. You can then select the one that performs best by
checking the error in the cross-validation data set.

Model Complexity Effects
------------------------

-   Lower-order polynomials (low model complexity) have high bias and
    low variance. In this case, the model fits poorly consistently.

-   Higher-order polynomials (high model complexity) fit the training
    data extremely well and the test data extremely poorly. These have
    low bias on the training data, but very high variance.

-   In reality, we would want to choose a model simewhere in between,
    that can generalize well but also fits the data reasonably well.

### Test your understanding

Suppose you fit a nerual network with one hidden layer to a training
set. You find that the cross validation error $$J_{CV}(\Theta{})$$ is much
larger than the training error $$J_{train}(\Theta{})$$. Is increasing the
number of hidden units likely to help?

-   Yes, because this increases the number of parameters and lets the
    network represent more complex functions.

-   Yes, because it is currently suffering from high bias.

-   No, because it is currently suffering from high bias, so adding
    hidden units is unlikely to help.

-   No, because it is currently suffering from high variance, so adding
    hidden units is unlikely to help.

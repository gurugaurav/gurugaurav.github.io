---
layout: post
title: Multiple Linear Regression
tags:
- ''
- ml
---

In the previous example, we have trained and evaluated a model to predict the price of a pizza. While you are eager to demonstrate the pizza-price predictor to your friends and co-workers, you are concerned by the model's imperfect r-squared score and the embarrassment its predictions could cause you.

How can we improve the model?
Recalling your personal pizza-eating experience, you might have some intuitions
about the other attributes of a pizza that are related to its price. For instance, the
price often depends on the number of toppings on the pizza. Fortunately, your
pizza journal describes toppings in detail; let's add the number of toppings to our
training data as a second explanatory variable. We cannot proceed with simple linearregression, but we can use a generalization of simple linear regression that can use multiple explanatory variables called multiple linear regression. Formally, multiple linear regression is the following model:

$$y =  \alpha + \beta_1 x_1 +  \beta_2x _2 +  \beta_3x_3 ...... $$


this edit makes no sense. change to "Where simple linear regression uses a single
explanatory variable with a single coefficient, multiple linear regression uses a
coefficient for each of an arbitrary number of explanatory variables.

$$y = X \beta $$


For simple linear regression, this is equivalent to the following:




$$  \begin{bmatrix}y_{1}\\y_{2}\\...\\y_{4}\end{bmatrix} =  \begin{bmatrix} \alpha + \beta X_{1}\\\alpha +\beta X_{2}\\...\\\alpha + \beta X_{3}\end{bmatrix} =  \begin{bmatrix}1 + X_{1}\\1 + X_{1}\\...\\1 + X_{1}\end{bmatrix} × \begin{bmatrix} \alpha \\ \beta \end{bmatrix} $$


 Y is a column vector of the values of the response variables for the training examples.
$$\beta$$ is a column vector of the values of the model's parameters. X, called the design
matrix, is an m× n dimensional matrix of the values of the explanatory variables for
the training examples. m is the number of training examples and n is the number of
explanatory variables.Let's update our pizza training data to include the number of
toppings with the following values:

| Training Example  | Diameter | Number of toppings | Price (in dollars) |
|-------------------|----------|--------------------|--------------------|
| 1                 | 6        | 2                  | 7                  |
| 2                 | 8        | 1                  | 9                  |
| 3                 | 10       | 0                  | 13                 |
| 4                 | 14       | 2                  | 17.5               |
| 5                 | 18       | 0                  | 18                 |


We must also update our test data to include the second explanatory variable,
as follows:

| Training Example  | Diameter | Number of toppings | Price (in dollars) |
|-------------------|----------|--------------------|--------------------|
| 1                 | 8        | 2                  | 11                 |
| 2                 | 9        | 0                  | 8.5                |
| 3                 | 11       | 2                  | 15                 |
| 4                 | 16       | 2                  | 18                 |
| 5                 | 12       | 0                  | 11                 |


Our learning algorithm must estimate the values of three parameters: the coefficients for the two features and the intercept term. While one might be tempted to solve $$\beta$$  by
dividing each side of the equation by X , division by a matrix is impossible. Just as
dividing a number by an integer is equivalent to multiplying by the inverse of the
same integer, we can multiply $$\beta$$  by the inverse of X to avoid matrix division. Matrix
inversion is denoted with a superscript -1. Only square matrices can be inverted. X
is not likely to be a square; the number of training instances will have to be equal to
the number of features for it to be so. We will multiply X by its transpose to yield a
square matrix that can be inverted. Denoted with a superscript T , the transpose of a
matrix is formed by turning the rows of the matrix into columns and vice versa,
as follows:

$$  \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}^T =  \begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix} $$

We know the values of Y and X from our training data. We must find the values
of $$\beta $$ , which minimize the cost function. We can solve $$\beta$$ as follows:

$$\beta = ( X^T X)^1 X^TY $$

Lets solve it using python:

```
from sklearn.linear_model import LinearRegression
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
  print(Predicted: %s, Target: %s' % (prediction, y_test[i]))
print('R-squared: %.2f' % model.score(X_test, y_test))
```

> Predicted: [ 10.0625], Target: [11] 
> 
> Predicted: [ 10.28125], Target: [8.5]
> 
> Predicted: [ 13.09375], Target: [15]
> 
> Predicted: [ 18.14583333], Target: [18]
> 
> Predicted: [ 13.3125], Target: [11]
> 
> R-squared: 0.77

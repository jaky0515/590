"""
main python file for hw0
@author: Kevin Jang (kj460)
@course: ANLY-590
@last_updated: 09/18/18
@reference_links:
	- graphs: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html
	- regression: https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
	- regularization: http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf
	- lasso regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
	- ridge regeression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
"""

# libraries
from sklearn.linear_model import Lasso, Ridge
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def lasso_reg(train_df, test_df):
	lasso = Lasso(normalize=True, max_iter=1e5)
	# include all the column values, but Salary column
	x = train_df.loc[:, train_df.columns != 'Salary'].values
	y = train_df['Salary'].values
	lasso.fit(x, y)
	predict = lasso.predict(x)
	# calculate residual sum of squares
	rss = sum((predict-y)**2)
	coefs = lasso.coef_
	result = [rss]
	result.extend([lasso.intercept_])
	result.extend(coefs)
	# visualization
	draw_visual(train_df.columns[train_df.columns != 'Salary'], coefs, 'Predictor', 'Coefficients', 'Coefficient Trajectories (Lasso Regression)')

def ridge_reg(train_df, test_df):
	ridge = Ridge(normalize=True)
	# include all the column values, but Salary column
	x = train_df.loc[:, train_df.columns != 'Salary'].values
	y = train_df['Salary'].values
	ridge.fit(x, y)
	predict = ridge.predict(x)
	# calculate residual sum of squares
	rss = sum((predict-y)**2)
	coefs = ridge.coef_
	result = [rss]
	result.extend([ridge.intercept_])
	result.extend(coefs)
	# visualization
	draw_visual(train_df.columns[train_df.columns != 'Salary'], coefs, 'Predictor', 'Coefficients', 'Coefficient Trajectories (Ridge Regression)')

def draw_visual(x, y, x_label, y_label, title):
	plt.clf()	# clear the figure before creating new
	axes = plt.gca()
	axes.plot(x, y)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.axis('tight')
	plt.grid(True)
	plt.show()

def main():
	# select numeric predictors only from the data set
	df = pd.read_csv('Hitters.csv').select_dtypes(include='number')
	train_df = df.loc[df['Salary'].notnull()]
	test_df = df.loc[df['Salary'].isnull()]
	lasso_reg(train_df, test_df)
	ridge_reg(train_df, test_df)

if __name__ == "__main__":
	main()
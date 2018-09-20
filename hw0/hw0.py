"""
main python file for hw0
@author: Kevin Kyunggeun Jang (kj460)
@course: ANLY-590
@last_updated: 09/19/18
@reference_links:
	- graphs: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html
	- regression: https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
	- regularization: http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf
	- lasso regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
	- ridge regeression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
	- lasso.path(): http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html
"""

# libraries
from sklearn.linear_model import Lasso, Ridge, lasso_path, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale 
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
this function performs lasso regression
"""
def lasso_reg(df, x, y):
	lasso = Lasso(normalize=True, max_iter=1e5)
	alphas = np.logspace(-3, 2, 200)	# list of alpha values
	# get coefficients
	coefs = get_coefs(lasso, alphas, x, y)
	# draw coefficient trajectories
	draw_coef_traject(alphas, coefs, 'alpha', 'coefficients', 'Coefficient Trajectories (Lasso Regression)', df.columns[df.columns != 'Salary'])
	# cross-validate and get the optimal alpha, model, and MSE
	opt_alpha, opt_lasso, mse, train_mse = cross_validate(x, y, True, alphas)
	# print coefficitns of optimal model to get the remaining predicators
	pprint(opt_lasso.coef_)
	# print results
	print('lasso optimal alpha = {}'.format(opt_alpha))
	print('lasso optimal MSE = {}'.format(mse))
	print('lasso train MSE = {}'.format(train_mse))

"""
this function performs ridge regression
"""
def ridge_reg(df, x, y):
	ridge = Ridge(normalize=True)
	alphas = np.logspace(-4, 4, 200)	# list of alpha values
	# get coefficients
	coefs = get_coefs(ridge, alphas, x, y)
	# draw coefficient trajectories
	draw_coef_traject(alphas, coefs, 'alpha', 'coefficients', 'Coefficient Trajectories (Ridge Regression)', df.columns[df.columns != 'Salary'])
	# cross-validate and get the optimal alpha, model, and MSE
	opt_alpha, opt_ridge, mse, train_mse = cross_validate(x, y, False, alphas)
	pprint(opt_ridge.coef_)
	# print results
	print('ridge optimal alpha = {}'.format(opt_alpha))
	print('ridge optimal MSE = {}'.format(mse))
	print('ridge train MSE = {}'.format(train_mse))

"""
this function performs cross-validation
"""
def cross_validate(x, y, is_lasso, alphas):
	# cross-validation to find the optimal alpha value
	if is_lasso:
		cv = LassoCV(max_iter=1e5, normalize=True, alphas=None, cv=10)
	else:
		cv = RidgeCV(normalize=True, alphas=alphas, cv=10)
	cv.fit(x, y)
	# split the data into two: train and test
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
	# create new model with the optimal alpha value
	if is_lasso:
		model = Lasso(normalize=True, max_iter=1e5, alpha=cv.alpha_)
	else:
		model = Ridge(normalize=True, alpha=cv.alpha_)
	model.fit(x_train, y_train)
	# compare our model's results with the real results and get MSE
	mse = mean_squared_error(y_test, model.predict(x_test))
	train_mse = mean_squared_error(y_train, model.predict(x_train))
	return cv.alpha_, model, mse, train_mse

"""
this function returns the list of coefficients over different alpha values
"""
def get_coefs(model, alphas, x, y):
	coefs = []	# list to store coefficients
	# run with every alpha value and store each coefficient
	for alpha in alphas:
		model.set_params(alpha=alpha)
		model.fit(x, y)
		coefs.append(model.coef_)
	return coefs

"""
this function draws a coefficient trajectories
"""
def draw_coef_traject(x, y, x_label, y_label, title, legend):
	plt.clf()
	axes = plt.gca()
	axes.set_color_cycle([plt.cm.rainbow(i) for i in np.linspace(0, 1, 16)])
	axes.plot(x, y)
	axes.set_xscale('log')
	axes.set_xlim(axes.get_xlim()[::-1])
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend(legend)
	plt.axis('tight')
	plt.show()

"""
this function prepares the dataset by removing unnecessary predicators and rows
"""
def prepare_dataset(file_name):
	# select numeric predictors only from the data set
	df = pd.read_csv(file_name).select_dtypes(include='number')
	# drop all the rows with Salary value equals to NA
	df = df.loc[df['Salary'].notnull()]
	return df, df.loc[:, df.columns != 'Salary'].values, df['Salary'].values

"""
main function
"""
def main():
	df, x, y = prepare_dataset('Hitters.csv')
	lasso_reg(df, x, y)
	ridge_reg(df, x, y)

if __name__ == "__main__":
	main()
# Author : Sachin Vishwakarma

from numpy import mean
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
 
# generate 2d classification dataset
X, y = make_blobs(n_samples=1000, centers=3, n_features=2)

# define the model here any model can be defined
model = KNeighborsClassifier()

# define the space to search hyperparameters
search_space = [Integer(1, 5, name='n_neighbors'), Integer(1, 2, name='p')]
 
# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
	# something
	model.set_params(**params)

	# calculate 5-fold cross validation
	# here cv, n_jobs and scoring function can be chaged based on requirement
	result = cross_val_score(model, X, y, cv=5, n_jobs=10, scoring='accuracy')
	
	# calculate the mean of the scores
	estimate = mean(result)
	return 1.0 - estimate
 
# perform optimization using gaussian process with default parameter
# for more detail on gp_minimize see the documentation https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
result = gp_minimize(evaluate_model, search_space)

# Print the final result
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))

# TODO : file for loss hyperparameter tuning

# parameters:
# LOSS: l1, inv, dev, cond, lambda_diag 
# MODEL: hidden, sparse, tol, diagonal_bias
# TRAINING: lr, batch_size

# EVAL
# each resulting combination should be scored by a its evaluation of near spartiy (np.allclose)
# and preconditioning over a few batches of data. Save these results.
# also save a few images from each set using the eval_utility.inspect_instance to see their patter

# NOTES:
# make sure to wrap a each case in a try catch incase it errors out

# TEST
# testing initally with small epoches, ensure each one is producing different results






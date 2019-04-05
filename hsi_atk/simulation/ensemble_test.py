from hsi_atk.simulation.sim_scratch import randomProblem

n_histograms = 40
n_inf = 3

X, V, inf = randomProblem(5000, n_histograms, 10, n_inf)
Xtrain = X[:4750,:,:]
Vtrain = V[:4750,:]
Xtest = X[4750:,:,:]
Vtest = V[4750:,:]

# preprocessing_cases = {'case1': [SliceNDice(0)],
#                        'case2': [SliceNDice(1)],
#                        'case3': [SliceNDice(2)]}

# estimator_per_cases = {'case1': [('svr',SVR()), ('las',Lasso()), ('lin',LinearRegression())],
#                        'case2': [('svr',SVR()), ('las',Lasso()), ('lin',LinearRegression())],
#                        'case3': [('svr',SVR()), ('las',Lasso()), ('lin',LinearRegression())]}

# ensemble = SuperLearner()
# scorer = make_scorer(rmse, greater_is_better=False)
# evaluator = Evaluator(scorer=scorer)

# ensemble.add([('svr',SVR()), ('las',Lasso()), ('lin',LinearRegression())])
# ensemble.add(estimators=estimator_per_cases, preprocessing=preprocessing_cases)
# ensemble.add_meta(LinearRegression())

# evaluator.fit(Xtrain, Vtrain, [('svr',SVR()), ('las',Lasso()), ('lin',LinearRegression())], preprocessing=preprocessing_cases, n_iter=10)
# print(evaluator.results)

# for i in range(n_histograms):
#     for j in range(n_inf):
#         ensemble = SuperLearner()
#         ensemble.add([('svr',SVR(gamma='auto')), ('las',Lasso()), ('lin',LinearRegression())])
#         ensemble.add_meta(LinearRegression())
#         ensemble.fit(Xtrain[:,i,:], Vtrain[:,j])
#         preds = ensemble.predict(Xtest[:,i,:])
#         # print("The ", j, " label vector")
#         # print("The ", i, " hist")
#         print("The ", i, " hist for the ", j, " label vector", rmse(Vtest[:,j], preds))


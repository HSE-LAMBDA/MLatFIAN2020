# Homework

 - Read the docs about stacking:
   - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
   - https://scikit-learn.org/stable/modules/ensemble.html#stacking
   - (Note that to avoid overfitting the stacking classifier/regressor uses `sklearn.model_selection.cross_val_predict` inside to make predictions with individual models)
 - Use `StackingClassifier` to fit the Titanic data (see seminar 1)
 - Use 5-fold cross validation (with shuffling the data) to estimate the score of your stacked model, as well as of the individual base models. Compare them - does stacking improve wrt the base models?
   - Note: Stacking is computationally heavy on its own, due to `sklearn.model_selection.cross_val_predict`, so it might be hard to optimize the hyperparameters with KFold CV

from xgboost import XGBClassifier
import numpy as np

class BDTClassifier:
    def __init__(self, train_params=None, do_eval=False):

        if train_params is not None:
            self.model = XGBClassifier(
                subsample=train_params["subsample"],
                n_estimators=train_params["n_estimators"],
                gamma=train_params["gamma"],
                max_depth=train_params["max_depth"],
                eta=train_params["eta"],
                reg_lambda=train_params["reg_lambda"],
                reg_alpha=train_params["reg_alpha"],
                scale_pos_weight=train_params["scale_pos_weight"]
            )
        else:
            self.model = XGBClassifier()
        
        self.do_eval = do_eval
        self.eval_metric = ["logloss"]
        self.train_params = train_params

    def train(self, train_set, test_set=None):
        if self.do_eval:
            if len(train_set) == 2:
                X_train, y_train = train_set
                X_test, y_test = test_set
                w_train = np.ones_like(y_train)
                w_test = np.ones_like(y_test)
            else:
                X_train, y_train, w_train = train_set
                X_test, y_test, w_test = test_set

            self.model.set_params(eval_metric=self.eval_metric)
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
        else:
            if len(train_set) == 2:
                X_train, y_train = train_set
                w_train = np.ones_like(y_train)
            else:
                X_train, y_train, w_train = train_set
            self.model.fit(X_train, y_train)
        
    def predict(self, X):   
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        if self.do_eval:
            results = self.model.evals_result()
            epochs = len(results['validation_0'][self.eval_metric[0]])
            return epochs, results
        else:
            print("No evaluation performed since do_eval is set to False.")
            return None
    
    def save_model(self, filename):
        self.model.save_model(filename)
    
    def load_model(self, filename):
        self.model.load_model(filename)

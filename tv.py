class tv():
    def __init__(self):
        self.model = Prophet()
    
    def transform(self, X_train):
        X_train.time_step = pd.to_datetime(X_train.time_step)
        df = pd.DataFrame({"ds": X_train.time_step})
        return df
    
    def fit(self, X_train, y_train):
        df = pd.DataFrame({"ds": X_train, "y": y_train})
        self.model.fit(df)
        
    def predict(self, X_test):
        pred = self.model.predict(X_test)
        y_pred = np.where(pred.yhat <0, 0, pred.yhat)
        return y_pred
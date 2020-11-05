class washing_machine():
    
    def __init__(self):
        self.model = 0
        
    def shift_data(self, data, lag=30):
        series = [data.shift(i) for i in range(lag-1, -1, -1)]
        new_df = pd.concat(series, axis=1)
        name_col = []
        
        for i in range(lag, 0, -1):
            var = "consumption -" + str(i)
            name_col.append(var)
        
        new_df.columns = name_col
        new_df["std"] = new_df.std(axis=1)
        new_df["mean"] = new_df.mean(axis=1)
        
        return new_df
    
    def transform(self, X_train):
        
        #Environment information
        X_train = X_train.drop(columns="Unnamed: 9")
        X_train.time_step = pd.to_datetime(X_train.time_step)
        X_train["week"] = X_train["time_step"].map(lambda x: x.week)
        X_train["weekday"] = X_train["time_step"].map(lambda x: x.weekday)
        X_train["hour"] = X_train["time_step"].map(lambda x: x.hour)
        
        #Missing values
        X_train.fillna(method="ffill", inplace=True) #Get the previous non null values
        
        #Supervised consumption
        supervised_consumption = self.shift_data(X_train.consumption)
        X_train = X_train.join(supervised_consumption)
        
        #One hot encoding
        X_train_hour = pd.Categorical(X_train.hour, categories = [i for i in range(24)])
        X_train_weekday = pd.Categorical(X_train.weekday, categories = [i for i in range(7)])
        X_train_week = pd.Categorical(X_train.week, categories = [i for i in range(52)])
        
        X_train = X_train.join(pd.get_dummies(X_train_hour, prefix="hour"))
        X_train = X_train.join(pd.get_dummies(X_train_weekday, prefix="weekday"))
        X_train = X_train.join(pd.get_dummies(X_train_week, prefix="week"))
        
        X_train.drop(columns=["time_step", "hour", "weekday", "week"], inplace=True)
        
        return X_train
    
    def fit(self, X_train, y_train):
        lgb_train = lgb.Dataset(X_train.values, y_train)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0}

        self.model = lgb.train(params, lgb_train,
                             num_boost_round=10000)
        
    def predict(self, X_test):
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = np.where(y_pred<0,0,y_pred)
        return y_pred
    
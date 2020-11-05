class fridge_freezer():
    def __init__(self):
        self.model = lgb.LGBMRegressor(n_estimators=1700, min_child_weight=1.5, max_depth=12, 
                               max_bin=100, colsample_bytree=0.6)
    
    
    def shift_data(self, data, lag=30):
        series = [data.shift(i) for i in range(lag-1, -1, -1)]
        new_df = pd.concat(series, axis=1)
        name_col = []
        
        for i in range(lag, 0, -1):
            var = "consumption -" + str(i)
            name_col.append(var)
        
        new_df.columns = name_col
        return new_df
    
    def transform(self, X_train):
        X_train = X_train.drop(columns=["Unnamed: 9"])
        X_train.time_step = pd.to_datetime(X_train.time_step)
        X_train["week"] = X_train["time_step"].map(lambda x: x.week)
        X_train["weekday"] = X_train["time_step"].map(lambda x: x.weekday)
        X_train["hour"] = X_train["time_step"].map(lambda x: x.hour)
        
        #Missing values
        X_train.fillna(method="ffill", inplace=True) #Get the previous non null values
        
        #Conditions on consumption
        X_train["capped_consumption"] = np.where(X_train.consumption>=350, 350, X_train.consumption)
        
        #Supervised consumption
        supervised_consumption = self.shift_data(X_train.capped_consumption)
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
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        y_pred = np.where(y_pred<0, 0, y_pred)
        return y_pred
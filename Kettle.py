class kettle():
    
    def __init__(self):
        self.model_0 = xgb.XGBRegressor(min_child_weight=1.5, max_depth= 8, gamma=0.6, colsample_bytree=0.9)
        self.model_1 = xgb.XGBRegressor(min_child_weight=1.5, max_depth= 8, gamma=0.6, colsample_bytree=0.9)
        
    def shift_data(self, data, lag=20):
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
        X_train = X_train[["time_step", "consumption"]]
        X_train.time_step = pd.to_datetime(X_train.time_step)
        
        X_train["hour"] = X_train["time_step"].map(lambda x: x.hour)
        X_train["weekday"] = X_train["time_step"].map(lambda x: x.dayofweek)
        X_train["week"] = X_train["time_step"].map(lambda x: x.week)
        
        #More information on consumption
        X_train["cluster"] = np.where(X_train.consumption > 600, 1, 0)
        
        #Missing values
        X_train.fillna(method="ffill", inplace=True) #Get the previous non null values
        
        #Supervised consumption
        supervised_consumption = self.shift_data(X_train.consumption)
        
        #One hot encoding
        X_train_hour = pd.Categorical(X_train.hour, categories = [i for i in range(24)])
        X_train_weekday = pd.Categorical(X_train.weekday, categories = [i for i in range(7)])
        X_train_week = pd.Categorical(X_train.week, categories = [i for i in range(52)])

        X_train = X_train.join(pd.get_dummies(X_train_hour, prefix="hour"))
        X_train = X_train.join(pd.get_dummies(X_train_weekday, prefix="weekday"))
        X_train = X_train.join(pd.get_dummies(X_train_week, prefix="week"))
        
        X_train.drop(columns=["time_step","hour", "weekday", "week"], inplace=True)
        
        return X_train.join(supervised_consumption)
    
    def fit(self, X_train, y_train):
        X_train_0 = X_train[X_train.cluster == 0]
        X_train_1 = X_train[X_train.cluster == 1]

        y_train_0 = y_train[X_train.cluster == 0]
        y_train_1 = y_train[X_train.cluster == 1]
        
        self.model_0.fit(X_train_0, y_train_0)
        self.model_1.fit(X_train_1, y_train_1)
        
    def predict(self, X_test):
        y_pred_kettle_0 = self.model_0.predict(X_test[X_test.cluster==0])
        y_pred_kettle_1 = self.model_1.predict(X_test[X_test.cluster==1])
        
        df_0 = pd.DataFrame(y_pred_kettle_0, index=X_test[X_test.cluster == 0].index)
        df_1 = pd.DataFrame(y_pred_kettle_1, index=X_test[X_test.cluster == 1].index)
        y_pred_kettle = pd.concat([df_0, df_1])
        y_pred_kettle.sort_index(inplace=True)
        
        y_pred_kettle = np.where(y_pred_kettle<0, 0, y_pred_kettle)
        
        return y_pred_kettle
    
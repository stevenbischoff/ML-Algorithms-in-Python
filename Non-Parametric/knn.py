"""
This module contains classes that implement K-Nearest Neighbors classification
and regression.

Author: Steve Bischoff
"""

import itertools
import collections

class KNN():
    """
    Parent class for KNN_regressor and KNN_classifier classes
    Args:
        train_df: Pandas Dataframe containing both independent and dependent
            variables
        features: list-like, names of independent variables in train_df
        dep_var: str or int, name of dependent variable in train_df
        k: int, number of nearest points to consider
        p: numeric, hyperparameter for Value Difference Metric
        sigma: numeric, spread parameter of kernel used in KNN_regressor
        categorical_variables: list-like, names of categorical independent variables
        metric: str, 'euclidean' or 'value_difference'. 'euclidean' is for
            numerical data, 'value_difference' is for categorical.
    """

    def __init__(self, train_df,
                 features, dep_var,
                 k=1, p=1, sigma=1,
                 categorical_variables=[],
                 metric='euclidean'):

        self.k = k
        self.p = p
        self.sigma = sigma

        self.train_df = train_df
        self.X = train_df[features]
        self.y = train_df[dep_var]

        self.features = features
        self.dep_var = dep_var

        self.metric = metric

        self.categorical_variables = categorical_variables
        self.all_category_distances = {}
 
    def get_value_difference(self, x1, x2):
        """
        This method gets the pre-calculated Value Difference Metric for x1 and x2
        Args:
        Returns:
        """
        return sum([self.all_category_distances[i][(x1[i],x2[i])] for i in x1.index])**(1/self.p)

    
    def get_distances(self, x1, X):
        """
        This method calculates the distance between point x1 and every point in
        dataset X. Distance calculation depends on the value of self.metric
        Args:
            x1: Pandas Series
            X: Pandas dataframe
        Returns:
            Pandas Series
        """
        if self.metric == 'euclidean':
            return np.sqrt(((X - x1)**2).sum(axis=1)) # only works for DF X
        elif self.metric == 'value_difference':
            return X.apply(self.get_value_difference, x2=x1, axis=1)

    def classify(self, x1, X):
        """Placeholder"""
        pass

    def predict(self, X_test):
        """
        This method predicts the value for each member of X_test.
        Args:
            X_test: numeric dataframe
        Returns:
            Series with dtype determined by subclass "classify" method
        """
        
        return pd.Series([self.classify(X_test.iloc[i], self.X,
            ) for i in range(len(X_test))], index=X_test.index)

    def value_difference_metric(self, categorical_variable, val1, val2):
        """
        This method calculates the Value Difference Metric for two categorical
        variable values.
        Args:
            categorical_variable: column name in self.X
            val1: value of categorical_variable
            val2: value of categorical_variable
        Returns:
            float, Value Difference Metric
        """
        if val1 == val2:
            return 0

        x = self.X[categorical_variable]

        x1 = x[x == val1]
        x2 = x[x == val2]

        c1 = len(x1)
        c2 = len(x2)

        if (c1 == 0) or (c2 == 0): # for Voting dataset
            return 0.5

        d = 0

        for class_ in self.y.unique():

            y_class = self.y[self.y == class_]
            y1 = y_class[y_class.index.isin(x1.index)]
            y2 = y_class[y_class.index.isin(x2.index)]

            d += abs(len(y1)/c1 - len(y2)/c2)**self.p

        return d

    def set_category_distances(self, values = None):
        """
        This method calls value_difference_metric to store that measure for each
        categorical variable in the training data.
        Args:
            values: list-like or False, if category values need manual input
        """
        if values!=False:
            for category in self.categorical_variables:
                category_dict = {}

                if values: # for Voting dataset
                    unique_values = values
                else:
                    unique_values = self.X[category].unique()

                for comb in itertools.product(unique_values, repeat=2):
                    d = self.value_difference_metric(category, comb[0], comb[1])
                    category_dict[comb] = d

                self.all_category_distances[category] = category_dict

        
# Regressor subclass                
class KNN_regressor(KNN):
   
    def __init__(self, train_df,
                 features, dep_var,
                 k=1, p=1, sigma=1,
                 categorical_variables=None,
                 metric='euclidean'):

        self.k = k
        self.p = p
        self.sigma = sigma

        self.train_df = train_df
        self.X = train_df[features]
        self.y = train_df[dep_var]
        
        self.features = features
        self.dep_var = dep_var

        self.metric = metric

        self.categorical_variables = categorical_variables
        self.all_category_distances = {}       

    def classify(self, x1, X):
        """
        This method "classifies" (regresses) point x1 relative on train data X
        Args:
            x1: array-like
            X: dataframe
        Returns:
            float
        """
        X_temp = X.loc[X.index != x1.name, self.features]
        
        distances = self.get_distances(x1, X_temp).sort_values()
        
        k_distances = distances[:self.k]
        y_k = self.y[k_distances.index]        
        kernel = np.exp(-1*k_distances/self.sigma) # radial basis function

        return (kernel*y_k).sum()/kernel.sum()
    
    def get_error(self, y_test, y_hat, measure='MSE', eps=1.0):
        """
        This method returns a measure of error between predicted and actual
        values.
        Args:
            y_test:
            y_hat:
            measure: str, 'MSE' or 'threshold'
            eps: float, distance threshold for 'threshold' measure
        Returns:
            float
        """
        if measure == 'MSE':
            return ((y_test-y_hat)**2).mean()
        elif measure == 'threshold':
            return len(y_test[abs(y_test - y_hat) < eps])

    # 
    def edit(self, test_df, eps, category_values=None):
        """
        This method performs the editing process and returns a new
        KNN_regressor. Uses the eps parameter to determine "correct
        classification".
        Args:
            test_df:
            eps:
            category_values:           
        Returns:
            KNN_regressor
        """
        y_hat = self.predict(test_df[self.features])
    
        p_best = self.get_error(test_df[self.dep_var], y_hat,
                                measure='threshold', eps=eps)

        self.y = self.y.sample(frac=1)
        self.X = self.X.sample(frac=1)

        cls = copy.deepcopy(self)

        improvement = True
        while improvement:
            for i, x in cls.X.iterrows():

                y_pred = cls.classify(x, cls.X[cls.X.index!=i])

                if abs(y_pred - cls.y[i])>=eps:

                    cls = KNN_regressor(cls.train_df.drop(index=i),
                                        cls.features, cls.dep_var,
                                        k=cls.k,categorical_variables=cls.categorical_variables,
                                        metric=cls.metric)               
                    cls.set_category_distances(category_values)                    

            y_hat = cls.predict(test_df[cls.features])
            
            p_temp = cls.get_error(test_df[cls.dep_var], y_hat,
                                        measure='threshold', eps=eps)
            
            if p_temp <= p_best: # classification using eps
                improvement = False                
            elif p_temp > p_best:
                p_best = p_temp

        return cls

       
    def condense(self, eps):
        """
        This method performs the condensing process in-place. 
        Args:
            eps:
        """
        self.X = self.X.sample(frac=1)

        Z = pd.DataFrame(columns = self.X.columns)
        y_new = pd.Series(dtype=self.y.dtype)
        # add initial point
        Z = pd.concat([Z, self.X.iloc[0].to_frame().T])
        y_new = pd.concat([y_new, self.y[[Z.index[0]]]])
        
        self.X = self.X.drop(index=Z.index[-1])
        self.y = self.y.drop(index=Z.index[-1])

        addition = True

        while addition:
            addition = False
            for i, x in self.X.iterrows():
                distances = self.get_distances(x, Z).sort_values()                
                if abs(y_new[distances.index[0]] - self.y[i])>=eps:
                    Z = pd.concat([Z, x.to_frame().T])
                    y_new = pd.concat([y_new, self.y[[Z.index[-1]]]])
                    self.X = self.X.drop(index=i)
                    self.y = self.y.drop(index=i)
                    addition = True

        self.X = Z
        self.y = y_new


# Classifier sub-class
class KNN_classifier(KNN):
    
    def classify(self, x1, X):
        """
        This method classifies point x1 relative on train data X
        Args:
            x1: array-like
            X: dataframe
        Returns:
            class prediction
        """
        X_temp = X.loc[X.index != x1.name, self.features]
        k_distances = self.get_distances(x1, X_temp).sort_values()[:self.k]
        y_k = self.y[k_distances.index]

        return y_k.mode().values[0] # .values[0] in case of multiple modes
       
    def get_error(self, y_test, y_hat):
        """
        This method gets the proportion correct between predicted and actual
        values.
        Args:
            y_test: iterable
            y_hat: iterable
        Returns:
            float in [0, 1]
        """
        return len(y_test[y_test==y_hat])/len(y_test)

    def edit(self, test_df, category_values=None):
        """
        This method performs the editing process and returns a new
        KNN_classifier.
        Args:
            test_df:
            category_values:
        Returns:
            KNN_classifier
        """
        y_hat = self.predict(test_df[self.features])
    
        p_best = self.get_error(test_df[self.dep_var], y_hat)

        self.y = self.y.sample(frac=1)
        self.X = self.X.sample(frac=1)

        cls = copy.deepcopy(self)

        improvement = True
        while improvement:
            for i, x in cls.X.iterrows():

                y_pred = cls.classify(x, cls.X[cls.X.index!=i])

                if y_pred != cls.y[i]:
                    cls = KNN_classifier(cls.train_df.drop(index=i),
                                         cls.features, cls.dep_var,
                                         k=cls.k,
                                         categorical_variables=cls.categorical_variables, metric=cls.metric)               
                    cls.set_category_distances(category_values)

            y_hat = cls.predict(test_df[cls.features])            
            p_temp = cls.get_error(test_df[cls.dep_var], y_hat)

            if p_temp <= p_best:
                improvement = False              
            elif p_temp > p_best:
                p_best = p_temp

        return cls

   def condense(self):
        """
        This method performs the condensing process in-place.
        """
        self.X = self.X.sample(frac=1)

        Z = pd.DataFrame(columns = self.X.columns)
        y_new = pd.Series(dtype=self.y.dtype)
        
        Z = pd.concat([Z, self.X.iloc[0].to_frame().T])
        y_new = pd.concat([y_new, self.y[[Z.index[0]]]])
        
        self.X = self.X.drop(index=Z.index[-1])
        self.y = self.y.drop(index=Z.index[-1])

        addition = True

        while addition:

            addition = False
            for i, x in self.X.iterrows():
                distances = self.get_distances(x, Z).sort_values()                
                if y_new[distances.index[0]] != self.y[i]: # check if same class
                    Z = pd.concat([Z, x.to_frame().T])
                    y_new = pd.concat([y_new, self.y[[Z.index[-1]]]])
                    self.X = self.X.drop(index=i)
                    self.y = self.y.drop(index=i)
                    addition = True

        self.X = Z
        self.y = y_new 

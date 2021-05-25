
# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# %%
data = pd.read_csv("Real-estate.csv",index_col=0)
data.columns = ['X1','X2','X3','X4','X5','X6','Y']


# %%
X = data.drop(data.columns[6],axis=1)
y = data.iloc[:,6]

X_train, X_test, y_train, y_test = train_test_split(X,y)


# %%
print(X)


# %%
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
std_data = std.fit_transform(data)

y = std_data[:,6]
X = np.delete(std_data,6,axis=1)

y = pd.DataFrame(y, columns = ['Y'])
X = pd.DataFrame(X, columns = ['X1','X2','X3','X4','X5','X6'])



X_train, X_test, y_train, y_test = train_test_split(X,y)

# %% [markdown]
# Let us visualize the correlations between features and the house prices with a simple heatmap. We use `numpy` to compute the correlations and `seaborn` for visualizations. We use a cyclic colormap to display both negative and positive correlations.

# %%
correlations = pd.concat([X_train, pd.Series(y_train)], axis=1).corr()
    
plt.figure(figsize=(12,10))
sns.heatmap(correlations, annot=True, cmap='twilight_shifted')
plt.show()

# %% [markdown]
# ## Regression Tree
# %% [markdown]
# ### Splitting Criteria: RSS
# %% [markdown]
# Let us now build the regression tree. We start by implementing the splitting criteria which is used to decide the most discriminative features at each step. We use $RSS$ which is computed as follows:
# 
# $$RSS =\sum_{\text {left }}\left(y_{i}-\hat{y_{L}}\right)^{2}+\sum_{\text {right }}\left(y_{i}-\hat{y_{R}}\right)^{2}$$
# 
# where $\hat{y_L} and \hat{y_r}$ are mean y-value of left and right nodes.

# %%
def rss(y_left, y_right):
    def squared_residual_sum(y):
        return np.sum((y - np.mean(y)) ** 2)
    
    return squared_residual_sum(y_left) + squared_residual_sum(y_right) 

def mse(ytrue, yhat) -> float:
        """
        Method to calculate the mean squared error 
        """
        # Getting the total number of samples
        n = len(ytrue)

        # Getting the residuals 
        r = ytrue - yhat 

        # Squering the residuals 
        r = r ** 2

        # Suming 
        r = np.sum(r)

        # Getting the average and returning 
        return r / n

# %% [markdown]
# We now plot RSS of two features with respect to each possible threshold. We choose continous features with the strongest correlation (LSTAT) and weakest correlation (DIS).
# 
# We observe that LSTAT can create much better splits than DIS, given its much lower RSS. We also observe parabola-like shapes for both features, indicating that both features have meaningful split points.

# %%
def compute_rss_by_threshold(feature):
    features_rss = []
    thresholds = X_train[feature].unique().tolist()
    thresholds.sort()
    thresholds = thresholds[1:]
    for t in thresholds:
        y_left_ix = X_train[feature] < t
        y_left, y_right = y_train[y_left_ix], y_train[~y_left_ix]
        features_rss.append(mse(y_left, y_right))
    return thresholds, features_rss

# %% [markdown]
# ### Splitting
# %% [markdown]
# We now implement the recursive splitting procedure as a function. We greedily find the best rules and split the data accordingly. We store the rules in a dictionary. Last, we use a helper function to find the best rule during each split.
# 
# We call the function to create a tree with depth 3 and visualize the result.

# %%
def find_best_rule(X_train, y_train):
    best_feature, best_threshold, min_rss = None, None, np.inf
    for feature in X_train.columns:
        thresholds = X_train[feature].unique().tolist()
        thresholds.sort()
        thresholds = thresholds[1:]
        for t in thresholds:
            y_left_ix = X_train[feature] < t
            y_left, y_right = y_train[y_left_ix], y_train[~y_left_ix]
            t_rss = mse(y_left, y_right)
            if t_rss < min_rss:
                min_rss = t_rss
                best_threshold = t
                best_feature = feature
    
    return {'feature': best_feature, 'threshold': best_threshold}


# %%
def split(X_train, y_train, depth, max_depth):
    if depth == max_depth or len(X_train) < 2:
        return {'prediction': np.mean(y_train)}
    
    rule = find_best_rule(X_train, y_train)
    left_ix = X_train[rule['feature']] < rule['threshold']
    rule['left'] = split(X_train[left_ix], y_train[left_ix], depth + 1, max_depth)
    rule['right'] = split(X_train[~left_ix], y_train[~left_ix], depth + 1, max_depth)
    return rule

rules = split(X_train, y_train, 0, 3)


# %%
print(rules)


# %%
import json
print(json.dumps(rules,indent=4))


# %%



# %%
def print_rules(rules):
    print(rules)
    if rules['left'] is not None:
        print_rules(rules['left'])
    if rules['right'] is not None:
        print_rules(rules['right'])


# %%
def print_info(self, width=4):
    """
    Method to print the infromation about the tree
    """
    # Defining the number of spaces 
    const = int(self.depth * width ** 1.5)
    spaces = "-" * const
    
    if self.node_type == 'root':
        print("Root")
    else:
        print(f"|{spaces} Split rule: {self.rule}")
    print(f"{' ' * const}   | MSE of the node: {round(self.mse, 2)}")
    print(f"{' ' * const}   | Count of observations in node: {self.n}")
    print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}")   

def print_tree(self):
    """
    Prints the whole tree from the current node to the bottom
    """
    self.print_info() 
    
    if self.left is not None: 
        self.left.print_tree()
    
    if self.right is not None:
        self.right.print_tree()

# %% [markdown]
# Now let us finish the regression tree by implementing the prediction function. We apply the rules in the rule tree until we arrive at a leaf

# %%
def predict(sample, rules):
    prediction = None
    while prediction is None:
        feature, threshold = rules['feature'], rules['threshold']
        if sample[feature] < threshold:
            rules = rules['left']
        else:
            rules = rules['right']
        prediction = rules.get('prediction', None)
    return prediction

# %% [markdown]
# ## Evaluation
# 
# We evaluate the regression tree by creating trees at different depths and measuring the $R^2$ on the test set

# %%
def evaluate(X, y):
    preds = X.apply(predict, axis='columns', rules=rules.copy())
    return r2_score(preds, y)


# %%
#X_train, y_train, X_test, y_test = prepare_dataset()
for max_depth in range(3, 8):
    rules = split(X_train, y_train, 0, max_depth)
    train_r2 = evaluate(X_train, y_train)
    test_r2 = evaluate(X_test, y_test)
    print('Max Depth', max_depth, 'Training R2:', train_r2, 'Test R2:',test_r2)


# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

DTR = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
DTR.fit(X_train, y_train)
y_pred = DTR.predict(X_test)
r2_score(y_test,y_pred)


# %%
from sklearn import tree
tree.plot_tree(DTR)

# %% [markdown]
# # Scrap

# %%
boston = load_boston()
X = pd.DataFrame(boston['data'], columns=boston['feature_names'])
y = pd.Series(boston['target'], name='House Price')
X_train, X_test, y_train, y_test = train_test_split(X,y)


# %%
def prepare_dataset():
    boston = load_boston()
    X_y = np.column_stack([boston['data'], boston['target']])
    np.random.seed(1)
    np.random.shuffle(X_y)
    X, y = X_y[:,:-1], X_y[:,-1]
    X_train, y_train, X_test, y_test = X[:400], y[:400], X[400:], y[400:]
    X_train = pd.DataFrame(X_train, columns=boston['feature_names'])
    X_test = pd.DataFrame(X_test, columns=boston['feature_names'])
    y_train = pd.Series(y_train, name='House Price')
    y_test = pd.Series(y_test, name='House Price')
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = prepare_dataset()
X_train.head()


# %%
lstat_thresholds, lstat_rss = compute_rss_by_threshold('X1')
dis_thresholds, dis_rss = compute_rss_by_threshold('X3')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.xlabel('Thresholds')
plt.ylabel('RSS')
plt.title('LSTAT (Strongest Correlation with House Price)')
plt.plot(lstat_thresholds, lstat_rss)

plt.subplot(1, 2, 2)
plt.xlabel('Thresholds')
plt.ylabel('RSS')
plt.title('DIS (Weakest Correlation with House Price)')
plt.plot(dis_thresholds, dis_rss)

plt.tight_layout()
plt.show()


# %%
# Initiating the Node
root = NodeRegression(y_train, X_train, max_depth=2, min_samples_split=3)
# Growing the tree
root.grow_tree()


# %%
root.print_tree()


# %%
# Data wrangling 
import pandas as pd 

# Array math
import numpy as np 

# Quick value count calculator
from collections import Counter


class NodeRegression():
    """
    Class to grow a regression decision tree
    """
    def __init__(
        self, 
        Y: list,
        X: pd.DataFrame,
        min_samples_split=None,
        max_depth=None,
        depth=None,
        node_type=None,
        rule=None
    ):
        # Saving the data to the node 
        self.Y = Y 
        self.X = X

        # Saving the hyper parameters
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        # Default current depth of node 
        self.depth = depth if depth else 0

        # Extracting all the features
        self.features = list(self.X.columns)

        # Type of node 
        self.node_type = node_type if node_type else 'root'

        # Rule for spliting 
        self.rule = rule if rule else ""

        # Getting the mean of Y 
        self.ymean = np.mean(Y)

        # Getting the residuals 
        self.residuals = self.Y - self.ymean

        # Calculating the mse of the node 
        self.mse = self.get_mse(Y, self.ymean)

        # Saving the number of observations in the node 
        self.n = len(Y)

        # Initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # Default values for splits
        self.best_feature = None 
        self.best_value = None 

    @staticmethod
    def get_mse(ytrue, yhat) -> float:
        """
        Method to calculate the mean squared error 
        """
        # Getting the total number of samples
        n = len(ytrue)

        # Getting the residuals 
        r = ytrue - yhat 

        # Squering the residuals 
        r = r ** 2

        # Suming 
        r = np.sum(r)

        # Getting the average and returning 
        return r / n

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list. 
        """
        return np.convolve(x, np.ones(window), 'valid') / window

    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.Y

        # Getting the GINI impurity for the base input 
        mse_base = self.mse

        # Finding which split yields the best GINI gain 
        #max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Getting the left and right ys 
                left_y = Xdf[Xdf[feature]<value]['Y'].values
                right_y = Xdf[Xdf[feature]>=value]['Y'].values

                # Getting the means 
                left_mean = np.mean(left_y)
                right_mean = np.mean(right_y)

                # Getting the left and right residuals 
                res_left = left_y - left_mean 
                res_right = right_y - right_mean

                # Concatenating the residuals 
                r = np.concatenate((res_left, res_right), axis=None)

                # Calculating the mse 
                n = len(r)
                r = r ** 2
                r = np.sum(r)
                mse_split = r / n

                # Checking if this is the best split so far 
                if mse_split < mse_base:
                    best_feature = feature
                    best_value = value 

                    # Setting the best gain to the current one 
                    mse_base = mse_split

        return (best_feature, best_value)

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data 
        df = self.X.copy()
        df['Y'] = self.Y

        # If there is GINI to be gained, we split further 
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Getting the best split 
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                # Creating the left and right nodes
                left = NodeRegression(
                    left_df['Y'].values.tolist(), 
                    left_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                    )

                self.left = left 
                self.left.grow_tree()

                right = NodeRegression(
                    right_df['Y'].values.tolist(), 
                    right_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                    )

                self.right = right
                self.right.grow_tree()

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | MSE of the node: {round(self.mse, 2)}")
        print(f"{' ' * const}   | Count of observations in node: {self.n}")
        print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()


# %%




# Auto Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agtmwebtoon/automl-data-science/blob/main/AutoML.ipynb#scrollTo=gLIIoFZXEGrf)



![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## AutoML in one image

![image](https://user-images.githubusercontent.com/71575861/174263501-a99f1445-e1a5-4704-a4b5-27f3bac7279a.png)

## About Function

- feed_input(): Set object's dataset  
      Also can random sampling by do_sampling option

- feature_cleaning(): Drop unused column  
        &nbsp;Encode string value using ordinalEncoder  
        &nbsp;Find Na value and fill it my df.mean

- split_dataset(): Split dataset

- find_best_combination(): Find best parameter and set best model

- set_model(): Set best model by using best parameter

- sampling(): Do random sampling

- feature_selection: Select best feature using selectKBest and scoring option is r_regression

- scaler(): Return **Normalized feature**

- pca(): Calculate PCA -> Return **Feature reduction by PCA**

- model_type(): Return **train method**

- make_subplot_layout(): Plot each features by subplots

- linear_regression(): Calculate MSE -> return **best linearRegreesion model** & **score of L logscale MSE**

- KNN(): Find best neighbors by using GridSearch  
   &nbsp;Scoring method is negative MAE -> return **best knn model** & **score of log scaled MSE**

## Finding best combination code

```python
def find_best_combination(self):
    self.split_dataset("Maximum Installs", False)
    param_k = np.arange(1, self.X.shape[1])
    param_scale_method = ['s', 'r', 'm']
    param_explained_var = np.linspace(0.01, 0.9999, 100)
    param_model_type = ['l', 'k']
    result = []
    param_list = list(itertools.product(param_k, param_scale_method, param_explained_var, param_model_type))

    for (k, scale_method, explained_var, param_model_type) in param_list:
        X = self.feature_selection(k)
        X = self.scaler(X, scale_method=scale_method)
        X = self.pca(X, explained_var=explained_var)

        X_train, X_test, y_train,  y_test = train_test_split(X, self.y, test_size = 0.2, random_state=7777)

        mt = self.model_type(param_model_type)
        _, score = mt(X_train, X_test, y_train, y_test)
        result.append(score)

    idx = np.argmin(result)
    self.score = np.min(result)
    self.best_param = param_list[idx]
    self.set_model(self.best_param)
```

## Example of explanation about module 

```python

    # Do feature selection (k = 1 .. size of feature)
    # Do normalization (StandardScaler, RobustScaler, MinMaxScaler)
    # Do PCA (explained_variance = 0..1)
    # find best parameter by using different parameter

    '''
    find_best_combination
    @Author: MinHyung Lee
    @Since: 2022/06/02
    find best parameter and set best model
    '''
```


## CardStack props

<table>
  <tr>
    <th>props</th>
    <th>type</th>
    <th>description</th>
    <th>required</th>
    <th>default</th>
  </tr>
  <tr>
    <td>dataset</td>
    <td>dataframe</td>
    <td>raw dataframe for machine learning</td>
    <td></td>
    <td>{}</td>
  </tr>
  <tr>
    <td>do_sampling</td>
    <td></td>
    <td>Flag that sampling the dataset</td>
    <td></td>
    <td>{}</td>
  </tr>
  <tr>
    <td>sample_size</td>
    <td>int</td>
    <td>Sample size for random sampling</td>
    <td></td>
    <td>{}</td>
  </tr>
  <tr>
    <td>unused_column</td>
    <td>list</td>
    <td>Features considered useless</td>
    <td></td>
    <td>[]</td>
  </tr>
  <tr>
    <td>string_column</td>
    <td>list</td>
    <td>Features consisted of String</td>
    <td></td>
    <td>[]</td>
  </tr>
  <tr>
    <td>target</td>
    <td>dataframe</td>
    <td>Dataset</td>
    <td></td>
    <td>{}</td>
  </tr>
  <tr>
    <td>best_param</td>
    <td>list</td>
    <td>parameter set calculated by find_best_combination</td>
    <td></td>
    <td>[]</td>
  </tr> 
  <tr>
    <td>scale_method</td>
    <td>string</td>
    <td>s: for standard scaler, r: for robustscaler, m: for minmaxscaler</td>
    <td></td>
    <td>"s"</td>
  </tr>
  <tr>
    <td>train_method</td>
    <td>string</td>
    <td>l: for linear_regression, k: for KNN_regression</td>
    <td></td>
    <td>""</td>
  </tr>
    
</table>

### feature_cleaning(self, unused_column=[], string_column=[])
<table>
    <tr>
        <th>Parameter</th>
    </tr>
    <tr>
        <td>self</td>
        <td>DataFrame to clean and encode</td>
    </tr>
    <tr>
        <td>unused_columnn</td>
        <td>List of features considered useless</td>
    </tr>
    <tr>
        <td>string_column</td>
        <td>List of features considered of String</td>
    </tr>
</table> 

### linear_regression(self, X_train, X_test, y_train, y_test)
<table>
    <tr>
        <th>Parameter</th>
    </tr>
     <tr>
        <td>self</td>
        <td>DataFrame to calculate MSE</td>
    </tr>
    <tr>
        <td>X_train</td>
        <td>DataFrame to train</td>
    </tr>
    <tr>
        <td>X_test</td>
        <td>DataFrame to test</td>
    </tr>
    <tr>
        <td>y_train</td>
        <td>Target DataFrame to train</td>
    </tr>
    <tr>
        <td>y_test</td>
        <td>Target DataFrame to test</td>
    </tr>
    <tr>
        <th>Return</th>
    </tr>
    <tr>
        <td>Model</td>
        <td>linearRegression model</td>
    </tr> 
    <tr>
        <td>score</td>
        <td>log scaled MSE</td>
    </tr>
</table>
    
    
### KNN(self, X_train, X_test, y_train, y_test)
<table>
    <tr>
        <th>Parameter</th>
    </tr>
     <tr>
        <td>self</td>
        <td>DataFrame to find best neighbors and MSE</td>
    </tr>
    <tr>
        <td>X_train</td>
        <td>DataFrame to train</td>
    </tr>
    <tr>
        <td>X_test</td>
        <td>DataFrame to test</td>
    </tr>
    <tr>
        <td>y_train</td>
        <td>Target DataFrame to train</td>
    </tr>
    <tr>
        <td>y_test</td>
        <td>Target DataFrame to test</td>
    </tr>
    <tr>
        <th>Return</th>
    </tr>
        <tr>
        <td>Model</td>
        <td>knn model</td>
    </tr> 
    <tr>
        <td>score</td>
        <td>log scaled MSE</td>
    </tr>
</table>

## Guideline
### - To add an algorithm
> 1. Clone the open source from the git hub and receive it.
> 2. Add the desired algorithm. Since various algorithms are already defined in AutoML.ipynb, they are converted to the following format.
> def new algorithms (self, X_train, X_test, y_train, y_test):
> .crystal
> 3. If you have modified it, you can upload the modified file to git through the git add command and check it through the status command.

### - How to Visualize
> 1. AutoML.ipynb provides various visualization libraries such as seaborn matplotlib.
> 2. Therefore, if you want to visualize, you can use various functions such as scatterplot, histogram, boxplot, etc.
> 3. After learning, you can view data errors by using the train and test datasets.
> Ex. Plt.scatter(X_test,y_test)

### - Sampling Method
> 1. AutoML.ipynb also provides a function of random sampling when the amount of data is enormous.
> 2. Randomly sample 20,000 datasets from the original dataset, and you can modify the sample_size shown below as many as you want.
> Ex. If you want to have 40,000 data, you can put sample_size=40000.
> ![image](https://user-images.githubusercontent.com/71575861/174481154-303ff996-c95c-4c58-91fe-0e1f5211042f.png)


## Dataset
Dataset from https://github.com/gauthamp10  
You can also find dataset at https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps?resource=download

# Auto Machine Learning
Dataset from https://github.com/gauthamp10  
You can also find dataset at https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps?resource=download

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## AutoML in one image

![image](https://user-images.githubusercontent.com/71575861/174263501-a99f1445-e1a5-4704-a4b5-27f3bac7279a.png)

## About Function

- feed_input()
- feature_cleaning()
- split_dataset()
- find_best_combination()
- set_model()
- sampling()
- feature_selection
- scaler()
- pca()
- model_type()
- make_subplot_layout()
- linear_regression()
- KNN()

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
    

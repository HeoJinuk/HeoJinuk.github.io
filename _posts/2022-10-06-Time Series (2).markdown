---
# multilingual page pair id, this must pair with translations of this page. (This name must be unique)
lng_pair: id_Examples
title: Time Series - (2)

# post specific
# if not specified, .name will be used from _data/owner/[language].yml
author: Jinuk Heo
# multiple category is not supported
category: Time Series
# multiple tag entries are possible
tags: [Time Series]
# thumbnail image for post
img: ":output_45_2.png"
# disable comments on this page
#comments_disable: true

# publish date
date: 2022-10-06 20:18:00 +0900

# seo
# if not specified, date will be used.
#meta_modify_date: 2022-02-10 08:11:06 +0900
# check the meta_common_description in _data/owner/[language].yml
#meta_description: ""

# optional
# if you enabled image_viewer_posts you don't need to enable this. This is only if image_viewer_posts = false
#image_viewer_on: true
# if you enabled image_lazy_loader_posts you don't need to enable this. This is only if image_lazy_loader_posts = false
#image_lazy_loader_on: true
# exclude from on site search
#on_site_search_exclude: true
# exclude from search engines
#search_engine_exclude: true
# to disable this page, simply set published: false or delete this file
#published: false
---

# 오차 측정 메트릭


```python
import numpy as np

TRUE = np.array([ 10,  20, 30, 40, 50])
PRED = np.array([ 20,  10,  0, 10, 20])
```

## Mean Forecast Error (MFE) 

- 테스트 세트 전체의 실제값과 예측값 간의 차이의 평균
    
$$MFE = \frac{1}{n}\sum_{i=1}^n (y_{actual} - \hat{y}_{forecast})$$


```python
def MFE(true, pred):
    return np.mean(true-pred)

MFE(TRUE, PRED)
```




    18.0



## Mean Absolute Error (MAE)
- 테스트 세트 전체의 실제값과 예측값 간의 차이의 절대값의 평균
- 절대값을 통해 과소평가되거나 과대평가된 많은 값을 상쇄할 수 있는 MFE의 약점 보완 

$$MAE = \frac{1}{n}\sum_{i=1}^n |y_{actual} - \hat{y}_{forecast}|$$


```python
def MAE(true, pred):
    return np.mean(np.abs(true-pred))

MAE(TRUE, PRED)
```




    22.0



## Mean Squared Error (MSE)

- 테스트 세트 전체의 실제값과 예측값 간의 차이의 제곱의 평균
- 제곱을 통해 과소평가되거나 과대평가된 많은 값을 상쇄할 수 있는 MFE의 약점 보완 

$$MSE = \frac{1}{n}\sum_{i=1}^n (y_{actual} - \hat{y}_{forecast})^2$$


```python
def MSE(true, pred):
    return np.mean((true-pred)**2)

MSE(TRUE, PRED)
```




    580.0



## Root Mean Squared Error (RMSE)

- MSE의 제곱근
- 오류의 평균 제곱근을 통해, MSE에서 차이를 제곱하여 생성된 메트릭의 단위를 원래대로 되돌린다.

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_{actual} - \hat{y}_{forecast})^2}$$


```python
def RMSE(true, pred):
    return np.sqrt(np.mean((true-pred)**2))

RMSE(TRUE, PRED)
```




    24.08318915758459



## Mean Absolute Percentage Error (MAPE)

- 실제값과 예측값 간의 차이를 실제값으로 나눠줌으로써 오차가 실제값에서 차지하는 상대적인 비율을 계산
- 위에서 언급한 모든 오류 측정값은 척도에 따라 달라지므로 정확도를 두 개의 고유한 값 집합에 걸쳐 비교해야 하는 경우 이러한 메트릭은 오해의 소지가 있을 수 있다.
- MAPE는 오류를 실제 값의 백분율로 표시하므로 척도와 무관합니다.


- 단점
    - 실제값에 0이 존재한다면 MAPE가 정의되지 않는다. 
    - MAE가 같더라도 실제값과 예측값과의 대소 관계에 따라 과대 추정하는 예측값에 패널티를 더 부여한다.


$$MAPE = \frac{100}{n}\sum_{i=1}^n \left|\frac{y_{actual} - \hat{y}_{forecast}}{y_{actual}}\right|$$


```python
def MAPE(true, pred):
    return 100*(np.mean(np.abs((true-pred)/(true))))

MAPE(TRUE, PRED)
```




    77.0




```python
TRUE_UNDER = np.array([10, 20, 30, 40, 50])
PRED_OVER = np.array([30, 40, 50, 60, 70])
TRUE_OVER = np.array([30, 40, 50, 60, 70])
PRED_UNDER = np.array([10, 20, 30, 40, 50])


print('평균 오차가 20일 때 실제값과 예측값의 대소 관계에 따른 MAE, MAPE 비교 \n')

print('실제값이 예측값 보다 작을 때 (예측값이 과대추정)')
print('MAE:', MAE(TRUE_UNDER, PRED_OVER))
print('MAPE:', MAPE(TRUE_UNDER, PRED_OVER))


print('\n실제값이 예측값 보다 클 때 (예측값이 과소추정)')
print('MAE:', MAE(TRUE_OVER, PRED_UNDER))
print('MAPE:', MAPE(TRUE_OVER, PRED_UNDER))
```

    평균 오차가 20일 때 실제값과 예측값의 대소 관계에 따른 MAE, MAPE 비교 
    
    실제값이 예측값 보다 작을 때 (예측값이 과대추정)
    MAE: 20.0
    MAPE: 91.33333333333333
    
    실제값이 예측값 보다 클 때 (예측값이 과소추정)
    MAE: 20.0
    MAPE: 43.71428571428571


## Symmetric Mean Absolute Percentage Error (SMAPE)

- MAPE가 지닌 한계점을 보완


- 단점
    - 분모에 예측값이 포함되므로, 예측값이 과소추정할 때 분모가 더 작아지므로 계산되는 오차가 커진다

$$SMAPE = \frac{100}{n}\sum_{i=1}^n \frac{|y_{actual} - \hat{y}_{forecast}|}{|y_{actual}| + |y_{forecast}|}$$


```python
def SMAPE(true, pred):
    return 100*(np.mean(np.abs((true-pred))/(np.abs(true)+np.abs(pred))))

SMAPE(TRUE, PRED)
```




    53.9047619047619




```python
TRUE2 = np.array([40, 50, 60, 70, 80])
PRED2_UNDER = np.array([20, 30, 40, 50, 60])
PRED2_OVER = np.array([60, 70, 80, 90, 100])

print('평균 오차가 20일 때 과소추정, 과대추정에 따른 MAE, SMAPE 비교 \n')

print('과대추정 시')
print('MAE:', MAE(TRUE2, PRED2_OVER))
print('SMAPE:', SMAPE(TRUE2, PRED2_OVER))

print('\n과소추정 시')
print('MAE:', MAE(TRUE2, PRED2_UNDER))
print('SMAPE:', SMAPE(TRUE2, PRED2_UNDER))
```

    평균 오차가 20일 때 과소추정, 과대추정에 따른 MAE, SMAPE 비교 
    
    과대추정 시
    MAE: 20.0
    SMAPE: 14.912698412698413
    
    과소추정 시
    MAE: 20.0
    SMAPE: 21.857142857142854


## Root Mean Squared Scaled Error (RMSSE)

- RMSSE는 MSE를 스케일링 할 때 훈련 데이터를 활용
- 훈련 데이터에 대해 naive forecasting을 했을 때의 MSE 값으로 나눠주기 때문에, 모델 예측값의 과소, 과대 추정에 따라 오차 값이 영향을 받지 않는다.
    - naive forecasting = 가장 최근 관측값으로 예측하는 방법


```python
def RMSSE(true, pred, train): 
    n = len(train)
    numerator = np.mean(np.sum(np.square(true - pred)))
    denominator = 1/(n-1)*np.sum(np.square((train[1:] - train[:-1])))
    msse = numerator/denominator
    
    return msse ** 0.5
```


```python
TRAIN = np.array([10, 20, 30, 40, 50])

RMSSE(TRUE, PRED, TRAIN)
```




    5.385164807134504



# 기본 예측 모델

## Naive 방법

- 마지막 관측값을 미래 예측으로 사용

```
y(forecast) = 마지막 관측값
```

![image.png](/assets/img/posts/navie.png)

## 단순 평균 방법

- 예측값 이전의 모든 과거의 관측값의 평균을 예측값으로 사용

```
y(forecast) = 모든 과거 관측값의 평균
```

![image.png](/assets/img/posts/simple_avg.png)

## 단순 이동 평균 방법

- 기본적인 단순 평균 방법의 확장
- 이동 윈도우(moving window)에 대한 이동 평균을 고려하여 데이터 세트의 일반적인 추세를 더 잘 포착
- 보다 현실적인 예측을 생성

```
y(forecast) = 이동 윈도우(moving window)의 평균
```

![image.png](/assets/img/posts/simple_moving_avg.png)

윈도우 크기는 모델의 정확도를 결정하는 중요한 역할
- 윈도우 크기가 클수록 윈도우 전체의 변화에 영향을 받을 가능성이 커진다. 
- 일반적으로 더 짧은 윈도우 크기가 더 잘 작동하여, 예측된 결과가 가장 최근 데이터의 영향을 더 많이 받는 경향이 있다.

또한 원래 데이터에 노이즈가 있는 경우, 이를 평균화하기 때문에 Naive 기법보다 훨씬 낫다.

위의 기본 예측 모델들은 데이터가 부족할 때 탐색할 수 있지만 시계열 데이터 간의 많은 관계를 식별하지 못한다. 
- 그렇기 때문에 상당한 데이터(> 10)가 있는 경우, 다양한 시계열 요소(레벨, 추세, 계절성)에 걸쳐 추세를 식별할 수 있는 보다 통계적인 기술을 사용한다.

## 가중 평균 방법

단순 이동 평균 방법과 달리 가중 평균 방법은 가중 매개변수를 사용하여 가장 최근의 결과에 더 높은 가중치를 제공 함으로써, 가장 최근의 관측값이 예측에 큰 영향을 미치게 할 수 있다. 
- 이는 예측 정확도를 크게 향상시킨다.

$$y_{t+1} = \frac{w_0y_0 + w_1y_1 + \cdots + w_ty_t}{(w_0 + w_1 + \cdots + w_t)}$$

## 단순 지수 평활화 (Simple Exponential Smoothing)

단순 지수 평활화은 이 기본 인수를 사용하여 시계열의 수준을 예측한다. 
- 다음 예측값($y_{t+1}$)은 현재 레벨($l_t$)의 함수로 처리되고, 이 레벨은 현재 값($y_t$)과 이전 레벨($l_{t-1}$)에 의해 결정된다. 
- 이 관계는 다음과 같이 일반화할 수 있다.

$$l_t = \alpha y_t + (1-\alpha) l_{t-1}$$

- 여기서 $\alpha$ = 평활 매개변수
    - 현재 레벨에서 현재 값의 가중치를 나타냄 
    - 0과 1 사이의 값
    - 이를 최적화해야 하는 하이퍼파라미터로 취급할 수 있다. 

결과적으로 예측값은 다음과 같이 표현할 수 있다.

$$\hat{y}_{t+1} = \alpha y_t + \alpha(1-\alpha) y_{t-1} + \alpha(1-\alpha)^2 y_{t-2}$$



```python
import pandas as pd

df = pd.read_csv('airline-passengers.csv', parse_dates=['Month'], index_col='Month')
df.index.freq = df.index.inferred_freq
```


```python
train_len = 120
train = df[0:train_len] # first 120 months as training set
test  = df[train_len:]  # last 24 months as out-of-time test set

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
model = SimpleExpSmoothing(train['Passengers'])
model_fit = model.fit(smoothing_level=0.2, optimized=False)
model_fit.params

y_hat_ses = test.copy()
y_hat_ses['ses_forecast'] = model_fit.forecast(24)

train['Passengers'].plot(figsize=(10,4))
test['Passengers'].plot()
y_hat_ses['ses_forecast'].plot()
```




    <AxesSubplot:xlabel='Month'>




    
![png](output_28_1.png)
    


## 홀트 지수 평활화 (Holt’s Exponential Smoothing)

- 단순 지수 평활화로 예측된 결과는 마지막 상태 수준의 함수로 처리된다. 
- Holt의 지수 평활화는 이 개념을 더욱 발전시키고 예측 값을 레벨과 추세 모두의 함수로 나타낸다.

$$\hat{y}_{t+1} = l_t + b_t$$

단순 지수 평활화에서 레벨 매개변수에 대한 계산하는 동안, 추세 매개변수는 추세 평활화 매개변수 $\beta$를 사용하여 계산된다.

$$l_t = \alpha y_t + (1 - \alpha)(l_{t-1} + b_{t-1})$$

$$\hat{b}_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1}$$

따라서 홀트 지수 평활화 방법은 추가 매개변수 $\beta$(평활 기울기)를 추가하여 레벨뿐만 아니라 시계열의 추세를 예측하는 데 도움이 된다.


```python
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(np.asarray(train['Passengers']),
                             seasonal_periods=12, trend='additive', seasonal=None)
model_fit = model.fit(smoothing_level=0.2, smoothing_trend=0.01, optimized=False)
print(model_fit.params)

y_hat_holt = test.copy()
y_hat_holt['holt_forecast'] = model_fit.forecast(24)

train['Passengers'].plot(figsize=(10,4))
test['Passengers'].plot()
y_hat_holt['holt_forecast'].plot()
```

    {'smoothing_level': 0.2, 'smoothing_trend': 0.01, 'smoothing_seasonal': None, 'damping_trend': nan, 'initial_level': 118.4666666666666, 'initial_trend': 2.060606060606069, 'initial_seasons': array([], dtype=float64), 'use_boxcox': False, 'lamda': None, 'remove_bias': False}





    <AxesSubplot:xlabel='Month'>




    
![png](output_35_2.png)
    


## 홀트-윈터 지수 평활화 (Holt Winters Exponential Smoothing)

홀트-윈터 지수 평활화는 이 개념을 한 단계 더 발전시켜, 시계열 데이터에 대한 레벨, 추세, 계절성을 예측한다. 
- 예측 방정식에는 이제 레벨과 추세 구성 요소 외에 계절성 구성 요소가 있다.

$$y_{t+1} = l_t + b_t + S_{t+1-m}$$

- 여기서 $m$은 한 기간에 계절이 반복되는 횟수를 나타낸다. 

계절 성분은 다음과 같이 나타낼 수 있다.

$$S_{t} = \gamma (y_t - l_{t-1} - b_{t-1}) + (1 - \gamma) S_{t-m}$$

- $\gamma$는 최근 관측값의 계절 성분에 할당된 가중치를 나타낸다.

레벨과 추세 구성 요소는 다음과 같이 업데이트할 수 있다.

$$\hat{b}_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1}$$

$$l_t = \alpha (y_t - S_{t-m}) + (1 - \alpha)(l_{t-1} + b_{t-1})$$

홀트-윈터 지수 평활화를 수행하는 방법에는 가법(additive)과 승법(multiplicative)의 두 가지 방법이 있다. 
- 시계열 데이터에서 계절성이 레벨 성분의 함수가 아니거나 그래프가 진행됨에 따라 시계열 데이터의 골 사이의 차이가 증가하지 않는 경우, Holt-Winters의 가법 방법이 가장 잘 작동한다. 
- 그러나 계절성이 레벨의 함수이고 그래프가 진행됨에 따라 시계열 데이터의 골 사이의 차이가 증가하는 경우, 승법 방법을 사용한다.

따라서 이 방법은 세 개의 매개변수를 훈련해야 한다.
- $\alpha$ (smoothing parameter)
- $\beta$ (smoothing slope)
- $\gamma$ (smoothing seasonal)


```python
y_hat_hwa = test.copy()
model = ExponentialSmoothing(np.asarray(train['Passengers']) ,seasonal_periods=12 ,trend='add', seasonal='add')
model_fit = model.fit(optimized=True)
print(model_fit.params)

y_hat_hwa['hw_forecast'] = model_fit.forecast(24)

train['Passengers'].plot(figsize=(10,4))
test['Passengers'].plot()
y_hat_hwa['hw_forecast'].plot()
```

    {'smoothing_level': 0.23676670348008969, 'smoothing_trend': 0.0, 'smoothing_seasonal': 0.7632332965199103, 'damping_trend': nan, 'initial_level': 119.17632782082607, 'initial_trend': 2.276910166883953, 'initial_seasons': array([ -9.4692609 ,  -3.86898675,   8.7227809 ,   3.71252206,
            -4.91683426,   9.2693423 ,  21.52217623,  19.18810308,
             5.07258493, -13.80075349, -28.50285806, -12.35305822]), 'use_boxcox': False, 'lamda': None, 'remove_bias': False}





    <AxesSubplot:xlabel='Month'>




    
![png](output_45_2.png)
    


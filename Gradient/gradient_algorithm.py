# import numpy as np
# from sklearn import linear_model
# from scipy.stats import linregress

# def gradient(x,y):
#     m_curr,intercept=0,0
#     iter=1000
#     learning_rate=0.02
    
#     # reg=linear_model.LinearRegression()
#     # reg.fit([x],y)
#     print(linregress(x, y))
#     #print("Expected Slope {} intercept {}".format(reg.coef_,reg.intercept_))
#     for i in range(iter):
#         y_predicted=m_curr*x+intercept
#         m_dirv=-((2/len(x))*sum(x*(y-y_predicted)))
#         intercept_dirv=-((2/len(x))*sum(y-y_predicted))
#         m_curr=m_curr-learning_rate*m_dirv
#         intercept=intercept-learning_rate*intercept_dirv
#         cost_fuction=1/len(x)*(sum([val**2 for val in y-y_predicted]))
#         print("Slope m {} intercept {} cost {} iter {} ".format(m_curr,intercept,cost_fuction,i))
        

        
# x=np.array([1,2,3,4,5])
# y=np.array([5,7,9,11,13])
# gradient(x,y)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def predict_using_sklean():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002

    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))

    return m_curr, b_curr
 
if __name__ == "__main__":
    df = pd.read_csv(r"F:\AWS machine learning ceritification\ML learning\test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklean()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))

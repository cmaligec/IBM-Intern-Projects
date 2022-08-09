import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

# For information on polynomial features in linear regression, see https://data36.com/polynomial-regression-python-scikit-learn/
# Also see for documentation https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures
# "Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]."

class GraphAnalysis:

    def __init__(self, csv_path, x_label, y_label, degree, graph_name):
        self.csv_path = csv_path
        self.x_label = x_label
        self.y_label = y_label
        self.degree = degree
        self.graph_name = graph_name
    
    def graph_analysis(self):
        df = pd.read_csv(self.csv_path)
        #df.describe()
        run_time = df.loc[:, self.y_label]
        voc = df.loc[:, self.x_label].to_numpy()
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        voc_poly = poly.fit_transform(voc.reshape(-1, 1))
        Voc_poly = np.c_[np.ones(voc_poly.shape), voc_poly]
        model = RANSACRegressor(random_state=123)
        model.fit(Voc_poly, run_time) 
        run_time_pred = model.predict(Voc_poly)
        return voc, Voc_poly, run_time, run_time_pred, model
    
    
    def gen_graph(self):   
        voc, _, run_time, run_time_pred, _ = self.graph_analysis()
        plt.figure(figsize = (10,6))
        plt.scatter(voc, run_time)
        plt.plot(voc, run_time_pred, c = 'red')
        plt.title(f'{self.y_label}_algorithm_poly – vocab vs. running time (sec)')
        plt.xlabel('Vocabulary') 
        plt.ylabel('Running time (sec)')
        plt.savefig(self.graph_name)



if __name__ == '__main__':
    max_degree = 4 
    logname = r'C:\Users\CFSM\Desktop\Embeddings\timings\timing_log_DRAFT.csv'
    #logname = r'C:\Users\CFSM\Desktop\Embeddings\timings\timing_log_2022_08_07__07_32_18_PM.csv'
    for deg in range(1, max_degree+1):
        graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\sk_graph_poly_{deg}_DRAFT.png'
        GraphAnalysis(logname, 'vocab', 'sk', deg, graph_name).gen_graph()
        print(f'sk results for polynomial of degree {deg}:')
        _, Voc_poly, run_time,_, model = GraphAnalysis(logname, 'vocab', 'sk', deg, graph_name).graph_analysis()
        print(f'sk_poly_{deg} R2: {model.score(Voc_poly, run_time)}')
        print(f'sk_poly_{deg} coefficients: {model.estimator_.coef_}')
        print(f'sk_poly_{deg} intercept: {model.estimator.intercept_}\n')
    
    print()
    
    for deg in range(1, max_degree+1):
        graph_name = rf'C:\Users\CFSM\Desktop\Embeddings\timings\dp_graph_poly_{deg}_DRAFT.png'
        GraphAnalysis(logname, 'vocab', 'dp', deg, graph_name).gen_graph()
        print(f'dp results for polynomial of degree {deg}:')
        _, Voc_poly, run_time,_, model = GraphAnalysis(logname, 'vocab', 'dp', deg, graph_name).graph_analysis()
        print(f'dp_poly_{deg} R2: {model.score(Voc_poly, run_time)}')
        print(f'dp_poly_{deg} coefficients: {model.coef_}')
        print(f'dp_poly_{deg} intercept: {model.intercept_}\n')

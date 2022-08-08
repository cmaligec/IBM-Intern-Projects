import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss 
import scipy.optimize as so
from sklearn import linear_model

class GraphAnalysis:

    def __init__(self, csv_path, x_label, y_label, graph_name):
        self.csv_path = csv_path
        self.x_label = x_label
        self.y_label = y_label
        self.graph_name = graph_name
        
    def linearModelPredict(self, b, X):
        return np.dot(X,b)
        
    def linearModelLossRSS(self, b, X, y):
        yh = self.linearModelPredict(b, X)
        res = y-yh
        rss = np.sum(np.square(res))
        gradient = -2*np.dot(X.T,res)
        return(rss,gradient)
    
    def linearModelLossLAD(self, b, X, y):
        yh = self.linearModelPredict(b, X)
        res = y-yh
        absolute = np.sum(np.abs(res))
        gradient = -2*np.dot(X.T,res)
        return(absolute,gradient)
    
    def linearModelFit(self, X, y, loss_func):
        bstart=np.zeros((2,1))
        betas= so.minimize(loss_func, bstart, args=(X, y), jac=True).x
        yh = self.linearModelPredict(betas,X)
        res = y-yh
        RSS = np.sum(np.square(res))
        yavg = np.sum(y) / len(y)
        diff = y - yavg
        TSS = np.sum(np.square(diff))
        R2 = 1 - (RSS/TSS)
        return betas, R2
    
    def gen_graph(self):   
        df = pd.read_csv(self.csv_path)
        df.plot.scatter(x= self.x_label, y= self.y_label,alpha=0.5)
        #df.describe()
        y = df.loc[:, self.y_label]
        voc = df.loc[:, self.x_label]
        Voc = np.c_[np.ones(voc.size), voc]
        b,R2 = self.linearModelFit(Voc, y, self.linearModelLossRSS)

        x_grid = np.linspace(voc.min(), voc.max(),100)
        Xn = np.c_[np.ones(x_grid.size), x_grid] # Make Design.
        yp=self.linearModelPredict(b,Xn) # Get prediction.
        fig, ax = plt.subplots(dpi = 120)
        ax.plot(x_grid, yp, color = 'red')
        df.plot.scatter(ax = ax, x= self.x_label, y= self.y_label,alpha=0.5)
        plt.savefig(self.graph_name)
        print(f'OLS_R2 for {self.y_label} graph: {R2}')
        
if __name__ == '__main__':
        
    logname = r'C:\Users\CFSM\Desktop\Embeddings\timings\timing_log_DRAFT.csv'
    graph_name = r'C:\Users\CFSM\Desktop\Embeddings\timings\sk_graph_DRAFT.png'
    GraphAnalysis(logname, 'vocab', 'sk', graph_name).gen_graph()
    logname = r'C:\Users\CFSM\Desktop\Embeddings\timings\timing_log_DRAFT.csv'
    graph_name = r'C:\Users\CFSM\Desktop\Embeddings\timings\dp_graph_DRAFT.png'
    GraphAnalysis(logname, 'vocab', 'dp', graph_name).gen_graph()
    
    print('sk results')
    df = pd.read_csv(logname)
    y = df.loc[:, 'sk']
    voc = df.loc[:, 'vocab']
    Voc = np.c_[np.ones(voc.size), voc]
    model = linear_model.LinearRegression()
    model.fit(Voc, y)
    print(model.score(Voc, y))
    print(model.coef_)
    print()
    print('dp results')
    y = df.loc[:, 'dp']
    model = linear_model.LinearRegression()
    model.fit(Voc, y)
    print(model.score(Voc, y))
    print(model.coef_)
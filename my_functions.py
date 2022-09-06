#Importo le librerie di cui avremo bisogno
from xmlrpc.client import Boolean
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import preprocessing
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
#Funzioni custom create da me e allegate a questo notebook

plt.figure(figsize=(32, 18))
sns.set_style("darkgrid")

def remove_outliers(dt_frm,col_list):
    """
    Funzione che rimuove gli outliers all'interno di specifiche colonne di un dataframe
    e restituisce un dataframe secondario contenente gli outliers\n
    Parametri:
    ---------- \n
    dt_frm: dataframe da cui togliere gli outlier
    col_list: lista delle colonne in cui togliere gli outliers
    """
    bounds = []
    for col in col_list:
        quantile1, quantile3 = np.percentile(dt_frm[col], [25, 75])
        iqr = quantile3 - quantile1
        lower_bound = quantile1 - (1.5 * iqr)
        upper_bound = quantile3 + (1.5 * iqr)
        bounds.append([lower_bound, upper_bound, str(col)])
    #print(bounds)

    outliers = []
    for bound in bounds:
        lower_cond = (dt_frm[dt_frm[bound[2]] <= bound[0]]).empty
        upper_cond = (dt_frm[dt_frm[bound[2]] >= bound[1]]).empty

        if lower_cond != True:
            outliers.append(dt_frm[dt_frm[bound[2]] <= bound[0]])

        if upper_cond != True:
            outliers.append(dt_frm[dt_frm[bound[2]] >= bound[1]])

        dt_frm.drop(dt_frm[dt_frm[bound[2]] <= bound[0]].index,inplace=True)
        dt_frm.drop(dt_frm[dt_frm[bound[2]] >= bound[1]].index,inplace=True)
    #print(outliers)
    
    z = pd.DataFrame()
    for out in enumerate(outliers):
        if out[0] < (len(outliers) - 1):
            z = pd.concat([z, outliers[out[0] + 1]], )

    z.reset_index(inplace=True, drop=True)
    #print(z.head(3))
    return z

def corr_coeff_2col(df, x, y):
    """
    Funzione che calcola il coefficiente di correlazione tra due colonne dello stesso dataframe
    Parametri:
    ----------\n
    df: Dataframe di partenza
    x: prima colonna
    y: seconda colonna\n
    Retern:
    -------\n
    Coefficienze di correlazione
    """
    x_mean = df[x].mean()
    y_mean = df[y].mean()
    # Calcolo la deviazione standard
    x_std = df[x].std()
    y_std = df[y].std()
    # Standardizzazione
    total_prod = (((df[x] - x_mean) / x_std) *
                      (((df[y] - y_mean) / y_std))).sum()
    corr = total_prod / (df.shape[0] - 1)
    return corr

def baseline_reg(estim,X,alternative_X,y,alternative_y,prnt=False,cv=10,mae=False):
    """Funzione che dati 4 dataset in entrata X e y di 2 versioni diverse, restituisce i corrispettivi valori di R2 e MAE dopo una cross validation e permette così di valutare i risultati.

    Args:
        estim (_type_): Stimatore usato
        X (_type_): Prima versione del dataset di X
        alternative_X (_type_): Seconda versione del dataset di X
        y (_type_): Prima versione del dataset di y
        alternative_y (_type_): Seconda versione del dataset di y
        prnt (bool, optional): Stampa a schermo un log. Defaults to False.
        cv (int, optional): Numero di cross validation da eseguire. Defaults to 10.

    Returns:
        list: Una lista contenente tutti i valori di MAE e R2 per le diverse varianti di dataset.
    """
    
    k = cross_validate(estim,X,y,cv=cv,scoring=("r2","neg_mean_absolute_error"),return_train_score=True)
    
    r2_test = np.mean(k["test_r2"]).round(3)
    mae_test = np.mean(k["test_neg_mean_absolute_error"]).round(1)
    
    r2_train = np.mean(k["train_r2"]).round(3)
    mae_train = np.mean(k["train_neg_mean_absolute_error"]).round(1)
    
    k1 = cross_validate(estim,alternative_X,alternative_y,cv=cv,scoring=("r2","neg_mean_absolute_error"),return_train_score=True)
        
    r2_test_1 = np.mean(k1["test_r2"]).round(3)
    mae_test_1 = np.mean(k1["test_neg_mean_absolute_error"]).round(1)
    
    r2_train_1 = np.mean(k1["train_r2"]).round(3)
    mae_train_1 = np.mean(k1["train_neg_mean_absolute_error"]).round(1)

    if prnt == True:
        print(f"{estim} \n"
                  f"R2 del primo test set vale: {r2_test} \nR2 del secondo test set vale: {r2_test_1} \nLa loro differenza (in termini assoluti) vale: {abs(r2_test-r2_test_1).round(2)}\n\n"
                  )
        if mae == True:
            
            print(f"MAE del primo test set vale: {mae_test} \nMAE del secondo test set vale: {mae_test_1} \nLa loro differenza (in termini assoluti) vale: {abs(mae_test-mae_test_1).round(2)}\n\n") 
    
    if mae == True:
            
        return [str(estim),r2_test,r2_test_1,mae_test,mae_test_1,r2_train,r2_train_1,mae_train,mae_train_1]
            
    return [str(estim),r2_test,r2_test_1,np.nan,np.nan,r2_train,r2_train_1,np.nan,np.nan]
        
    
def pair_plot(df,col_list,index_list,title_list):
    # Make the PairGrid
    g = sns.PairGrid(data=df,
                 x_vars=col_list, y_vars=index_list,
                 height=10, aspect=.25)

    # Draw a dot plot using the stripplot function
    g.map(sns.stripplot, size=10, orient="h", jitter=False,
      palette="flare_r", linewidth=1, edgecolor="w")

    # Use the same x axis limits on all columns and add better labels
    g.set(xlim=(0, 1), xlabel="", ylabel="")

    # Use semantically meaningful titles for the columns
    titles = title_list

    for ax, title in zip(g.axes.flat, titles):

        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)
    

def corr_list_with_target(df,target_col):
    Xcol = df.drop(target_col,axis=1)
    # Creazione lista di appoggio
    target_corr = []
    # Ciclo for per controllare il coeff. di correlazione tra tutte le features e la colonna "target"
    for col in Xcol:
        # Per informazioni sulla f.ne corr_coeff_2col() vedere documentazione o il file file my_functions.py allegato
        c = corr_coeff_2col(df, col, target_col).round(3)
        target_corr.append([col, c])

    # Riordina la lista basandosi sui valori di correlazioni delle singole features
    target_corr = sorted(target_corr, key=lambda x: abs(x[1]), reverse=True)
    return target_corr


def plot(df,Xcol,ycol1,ycol2,color1,color2,legend,xlabel,ylabel):
    ax = sns.lineplot(data=df,
                  x=Xcol,
                  y=ycol1,
                  palette=color1)
    sns.lineplot(data=df,
             x=Xcol,
             y=ycol2,
             palette=color2,
             ax=ax)
    ax.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def plot_kde(y_test, y_test_pred):
    
    #Kernel Density Estimation plot
    ax = sns.kdeplot(y_test, color='orange', label='Actual Values') #actual values
    sns.kdeplot(y_test_pred, color='blue', label='Predicted Values', ax=ax) #predicted values

    #showing title
    plt.title('Actual vs Precited values')
    #showing legend
    plt.legend()
    #showing plot
    plt.show()
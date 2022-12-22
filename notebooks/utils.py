import warnings
import pandas as pd
import numpy as np
# libraries for visualization
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline') 
get_ipython().magic(u"config InlineBackend.figure_format='retina'")
import seaborn as sns
sns.set_theme(style="white")
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="iair_linker_cornershop_challenge")
from geopy.distance import geodesic
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,Normalizer,RobustScaler
from sklearn.model_selection import GridSearchCV,cross_validate,cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve,classification_report


def get_period_day(x):
    if((x>=5) & (x<12)):
        return 'morning'
    elif((x>=12) & (x<19)):
        return 'afternoon'
    elif((x>=19) or (x<5)):
        return 'night'
    
def is_high_season(x):
    if( (x <= datetime(2017, 3, 3).date()) or  (x >= datetime(2017, 12, 15).date())):
        return 1
    elif( (x >= datetime(2017, 7, 15).date()) or  (x >= datetime(2017, 7, 31).date())):
        return 1
    elif( (x >= datetime(2017, 9, 11).date()) or  (x >= datetime(2017, 9, 30).date())):
        return 1
    else:
        return 0


def delete_outliers(data):
    aux = data.groupby(['Emp-Vlo-I']).count().sort_values(by='atraso_15').reset_index()
    aux.rename(columns={'atraso_15':'NVuelos-Emp-Vlo-I'}, inplace=True)
    lista = aux[(aux['NVuelos-Emp-Vlo-I']<2)]['Emp-Vlo-I'].values
    data['Emp-Vlo-delete'] = data['Emp-Vlo-I'].apply(lambda x: 1 if(x in lista) else 0)
    result = data[data['Emp-Vlo-delete']==0]
    #result = result[result['NVuelos-Ori-I']>100]
    return result

def create_variables(data):
    # create the variable of number of flights in origin by date operated and merge it with the database
    aux = data.groupby(['fecha_operacion'])['Ori-O'].count().reset_index().sort_values(by='Ori-O')
    aux.rename(columns={'Ori-O':'NVuelos-Ori-O'}, inplace=True)
    data = pd.merge(data, aux, on = ['fecha_operacion'], how='left')

    # create the variable of number of flights in origin by date scheduled and merge it with the database
    aux = data.groupby(['fecha_operacion'])['Ori-I'].count().reset_index().sort_values(by='Ori-I')
    aux.rename(columns={'Ori-I':'NVuelos-Ori-I'}, inplace=True)
    data = pd.merge(data, aux, on = ['fecha_operacion'], how='left')

    # create the variable of number of flights in destiny scheduled by date and airline scheduled
    aux = data.groupby(['fecha_operacion','Emp-I'])['Ori-I'].count().reset_index().sort_values(by='Ori-I')
    aux.rename(columns={'Ori-I':'Nvuelos-Fecha-Emp-I'}, inplace=True)
    data = pd.merge(data, aux, on = ['fecha_operacion','Emp-I'], how='left')

    # create the variable of number of flights in by date and airline who operated it
    aux = data.groupby(['fecha_operacion','Emp-O'])['Ori-O'].count().reset_index().sort_values(by='Ori-O')
    aux.rename(columns={'Ori-O':'Nvuelos-Fecha-Emp-O'}, inplace=True)
    data = pd.merge(data, aux, on = ['fecha_operacion','Emp-O'], how='left')

    # create the variable of number of flights in operated destiny by date of operation and airline who operated it
    aux = data.groupby(['fecha_operacion','Emp-O','Des-O'])['Ori-O'].count().reset_index().sort_values(by='Ori-O')
    aux.rename(columns={'Ori-O':'Nvuelos-Fecha-Emp-Des-O'}, inplace=True)
    data = pd.merge(data, aux, on = ['fecha_operacion','Emp-O','Des-O'], how='left')

    # create the variable of number of flights in destiny scheduled by date scheduled and airline scheduled
    aux = data.groupby(['fecha_operacion','Emp-I','Des-I'])['Ori-I'].count().reset_index().sort_values(by='Ori-I')
    aux.rename(columns={'Ori-I':'Nvuelos-Fecha-Emp-Des-I'}, inplace=True)
    data = pd.merge(data, aux, on = ['fecha_operacion','Emp-I','Des-I'], how='left')

    # Change in the code of the flight
    data[['Vlo-I_cambio']] = 0
    data.loc[data[data['Vlo-I'] != data['Vlo-O']].index,'Vlo-I_cambio'] = 1
    
    # Change type of data from number to string
    data['Vlo-O'] = data['Vlo-O'].apply(lambda x : str(x))
    data['Vlo-I'] = data['Vlo-I'].apply(lambda x : str(x))
    
    # Create the variable that merge airlines code with flight number
    data['Emp-Vlo-O'] = data['Emp-O'] + data['Vlo-O'] 
    data['Emp-Vlo-I'] = data['Emp-I'] + data['Vlo-I']
    
    return data


def scale_data(x,y,scale):
    if(scale=='min-max'):
        scaler = MinMaxScaler()
        scaler .fit(x,y)
        return scaler 
    elif( scale == 'robustscaler'):
        scaler = RobustScaler()
        scaler .fit(x,y)
        return scaler 
    elif(scale=='normalize'):
        scaler = Normalizer()
        scaler .fit(x,y)
        return scaler 
    else:
        raise Exception('You did not introduce a valid parameter for scale')
        
def traine(clf, X_train, y_train,X_test,y_test, treshhold = 0.5):
    scores = cross_val_score(estimator = clf, X = X_train,y = y_train,cv=5, scoring = 'f1_weighted')
    print("El F1-weighted en entrenamiento con validación cruzada es es: "+ str(scores.mean()))
    clf.fit(X_train, y_train)
    y_pred_probs = clf.predict_proba(X_test)[:,1]
    y_pred = y_pred_probs>=treshhold
    
    return y_pred_probs, y_pred

def test(y_test, y_pred_probs, y_pred):
    ras = roc_auc_score(y_test, y_pred_probs)
    report = classification_report(y_test, y_pred)
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual class'], colnames=['Predicted class'],margins=True)
    print("EL ROC AUC Score es: " + str(ras))
    print(report)
    return cm
       
    
def get_na(df):
    qsna=df.shape[0]-df.isnull().sum(axis=0)
    qna=df.isnull().sum(axis=0)
    ppna=round(100*(df.isnull().sum(axis=0)/df.shape[0]),2)
    aux= {'datos sin NAs en q': qsna, 'Na en q': qna ,'Na en %': ppna}
    na=pd.DataFrame(data=aux)
    return na.sort_values(by='Na en %',ascending=False)

def plot_pie(y):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct='%1.1f%%')
    ax.axis('equal')
    
def get_cdf_pdf(x):
    # getting data of the histogram
    count, bins_count = np.histogram(x, bins=10)
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    # plotting PDF and CDF
    plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()
    
def get_boxplot_target_by_var(x,var,target):
    print(x.groupby(var)[target].describe())
    sns.boxplot(x=var, y=target,data=x)
    plt.title("Boxplot of " + target + " by " + var);
    
def scatterplot_target_by_var(x,var1, var2,target):
    sns.scatterplot(data=x, y=target, x=var1, hue=var2 , legend=False)
    plt.title("Scatterplot of found"+ target + " and " + var1 + "respect to " + var2)
    plt.show();

def get_location(df):
    ind = []
    locations = []
    for index,row in df.iterrows():
        try:
            ind.append(index)
            location = geolocator.reverse([row.Lat,row.Long])
            locations.append(location)
        except:
            time.sleep(180) # cuando falla por  esperando le digo que espere
            ind.append(index)
            location = geolocator.reverse([row.Lat,row.Long])
            locations.append(location)
    return ind,locations

def get_distances(df):
    ind = []
    distances = []
    for index , row in df.iterrows():
        try:
            ind.append(index)
            distances.append(geodesic((row.Lat,row.Long), (-33.3930016,-70.7857971)).km)
        except:
            time.sleep(180) # cuando falla por  esperando le digo que espere
            ind.append(index)
            distances.append(geodesic((row.Lat,row.Long), (-33.3930016,-70.7857971)).km)
    return ind , distances
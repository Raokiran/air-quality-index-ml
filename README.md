# air-quality-index-ml
This is my ml project to emulate the prediction of air quality based on humidty and wind speed.
import streamlit as stt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import requests

from streamlit_lottie import st_lottie
def stream():
    def load_lottie(url):
        r=requests.get(url)
        if r.status_code !=200:
            return None
        return r.json()
    lottie_coding2=load_lottie("https://lottie.host/427ef65f-1211-4012-9764-d7964cd2a7bd/2HMh1ske5c.json")
    with stt.container():
        leftc,rightc=stt.columns(2)
        with leftc:
            stt.header("AIR QUALITY INDEX PREDICTION")
            stt.write("Predict the air quality of your city")
        with rightc:
                st_lottie(lottie_coding2,height=300,key="code")  
   
    with stt.container():
        stt.write("enter the PM2.5")
        a=stt.slider( "",min_value=0.0,  max_value=500.0,  value=5.0,  step=0.1   )
        stt.write("enter the PM10")
        b=stt.slider(" ",min_value=0.0,  max_value=500.0,  value=5.0,  step=0.1  )
        stt.write("enter the NO")
        c=stt.slider("  ",min_value=0.0,  max_value=500.0,  value=5.0,  step=0.1  )
        stt.write("enter the NO2")
        d=stt.slider("   ",min_value=0.0,  max_value=500.0,  value=5.0,  step=0.1  )
        stt.write("enter the NOx")
        e=stt.slider("      ",min_value=0.0,  max_value=500.0,  value=5.0,  step=0.1  )

        stt.write("enter the CO")
        g=stt.slider("          ",min_value=0.0,  max_value=500.0,  value=5.0,  step=0.1  )
        stt.write("enter the O3")
        i=stt.slider("                ",min_value=0.0,  max_value=500.0,  value=5.0,  step=0.1  )

        stt.write("enter the temperature")
        m=stt.slider("                               ",min_value=0.0,  max_value=60.0,  value=5.0,  step=0.1  )
        stt.write("enter the humidity")
        n=stt.slider("                                    ",min_value=10.0,  max_value=100.0,  value=5.0,  step=0.1  )
        stt.write("enter the wind speed")
        o=stt.slider("                                            ",min_value=1.0,  max_value=20.0,  value=5.0,  step=0.1  )

    return a,b,c,d,e,g,i,m,n,o
   
   
def stream_print(x):
    with stt.container():
        result=x[0]
        if result<50:
             cat='GOOD'
        elif result>50 and result<100 :  
             cat='MODERATE'
        elif result>100 and result<150 :  
             cat='Unhealthy for Sensitive Groups'
        elif result>150 and result<200 :  
             cat='Unhealthy'
        elif result>200 and result<250 :  
             cat='VERY Unhealthy '
        else :  
             cat='HAZARDOUS'
              
    stt.write(f"The estimated AQI is={int(result)}")
    stt.write(f"The category of AQI is={cat}")


def main():
    import pandas as pd
    data=pd.read_csv('air_quality.csv')
    data=data.drop('Date',axis=1)
    data9=data.drop('AQI',axis=1)
    data10=data['AQI']
    from sklearn.preprocessing import StandardScaler
    scale=StandardScaler()
    X_scaled=scale.fit_transform(data9)
    from sklearn.model_selection import train_test_split
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X_scaled,data10,test_size=0.20,random_state=42)
    rf=RandomForestRegressor(n_estimators=1,random_state=42)
    rf.fit(Xtrain,Ytrain)
    a,b,c,d,e,g,i,m,n,o=stream()
    test=[]
    test.append(a)
    test.append(b)
    test.append(c)
    test.append(d)
    test.append(e)
    test.append(g)
    test.append(i)
    test.append(m)
    test.append(n)
    test.append(o)
   
    user_input=[]
    user_input.append(test)
    x=rf.predict(user_input)
    result=stt.button("Predict")
    if result==True:
          stream_print(x)
if __name__ == "__main__":
        stt.set_page_config(page_title="Air Quality Index Prediction",layout="wide")
        main()


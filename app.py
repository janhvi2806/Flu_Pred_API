import pickle
with open("model_flu.pkl", "rb") as f:
    model = pickle.load(f)

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
import numpy as np
import uvicorn

class DataType(BaseModel):
    h1n1_concern: int
    h1n1_knowledge: int
    behavioral_antiviral_meds: int
    behavioral_avoidance: int
    behavioral_face_mask: int
    behavioral_wash_hands: int
    behavioral_large_gatherings: int
    behavioral_outside_home: int
    behavioral_touch_face: int
    doctor_recc_h1n1: int
    doctor_recc_seasonal: int
    chronic_med_condition: int
    child_under_6_months: int
    health_worker: int
    health_insurance: int
    opinion_h1n1_vacc_effective: int
    opinion_h1n1_risk: int
    opinion_h1n1_sick_from_vacc: int
    opinion_seas_vacc_effective: int
    opinion_seas_risk: int
    opinion_seas_sick_from_vacc: int
    age_group: str
    education: str
    race: str
    sex: str
    income_poverty: str
    marital_status: str
    rent_or_own: str
    employment_status: str
    hhs_geo_region: str
    census_msa: str
    household_adults: int
    household_children: int
    employment_industry: str
    employment_occupation: str

app = FastAPI()

"""
Sample JSON Input:- 
{
  "h1n1_concern": 1,
  "h1n1_knowledge": 0,
  "behavioral_antiviral_meds": 0,
  "behavioral_avoidance": 0,
  "behavioral_face_mask": 0,
  "behavioral_wash_hands": 0,
  "behavioral_large_gatherings": 0,
  "behavioral_outside_home": 1,
  "behavioral_touch_face": 1,
  "doctor_recc_h1n1": 0,
  "doctor_recc_seasonal": 0,
  "chronic_med_condition": 0,
  "child_under_6_months": 0,
  "health_worker": 0,
  "health_insurance": 1,
  "opinion_h1n1_vacc_effective": 3,
  "opinion_h1n1_risk": 1,
  "opinion_h1n1_sick_from_vacc": 2,
  "opinion_seas_vacc_effective": 2,
  "opinion_seas_risk": 1,
  "opinion_seas_sick_from_vacc": 2,
  "age_group": "55 - 64 Years",
  "education": "< 12 Years",
  "race": "White",
  "sex": "Female",
  "income_poverty": "Below Poverty",
  "marital_status": "Not Married",
  "rent_or_own": "Own",
  "employment_status": "Not in Labor Force",
  "hhs_geo_region": "oxchjgsf",
  "census_msa": "Non-MSA",
  "household_adults": 0,
  "household_children": 0,
  "employment_industry": null,
  "employment_occupation": null
}

"""


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df= pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns    
    

def Preprocessing(data):
    print(data)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    print(type(scaled_data))
    return scaled_data

with open("model_flu.pkl", "rb") as f:
    model = pickle.load(f)

def scale_data(Test_X):

    Test_X.drop(['health_insurance','employment_industry','employment_occupation'],axis=1,inplace = True)

    # All the NULL values are populated with the mode
            
    d = Test_X
    str_cols = d.select_dtypes(include = 'object').columns

    ### LabelEcoding all categorical types #####
    for col in Test_X.columns:
        if Test_X[col].isnull().sum() and Test_X[col].dtypes != 'object':
            Test_X[col].loc[(Test_X[col].isnull())] = Test_X[col].median()
    for col in Test_X.columns:
        if Test_X[col].isnull().sum() and Test_X[col].dtypes == 'object':
            Test_X[col].loc[(Test_X[col].isnull())] = Test_X[col].mode().max()
    LE = LabelEncoder()
    for col in str_cols:
        Test_X[col] = LE.fit_transform(Test_X[col]) # Converts to int64
            
    data = d
    ### Synthesizing two new features cleanliness level of the individual and opinion of vaccine ####
    data['opinion'] = data['opinion_h1n1_vacc_effective'] + data['opinion_h1n1_risk']+\
                  data['opinion_h1n1_sick_from_vacc'] + data['opinion_seas_vacc_effective']+\
                  data['opinion_seas_risk'] + data['opinion_seas_sick_from_vacc']
    data['cleanliness'] =  data['behavioral_antiviral_meds']+ data['behavioral_avoidance']+\
                        data['behavioral_face_mask']+data['behavioral_wash_hands']+\
                       data['behavioral_large_gatherings'] + data['behavioral_outside_home']+\
                       data['behavioral_touch_face']
    data['opinion_h1n1'] = data['opinion_h1n1_vacc_effective'] + data['opinion_h1n1_risk']-\
                      data['opinion_h1n1_sick_from_vacc'] 
    data['opinion_seasonal'] = data['opinion_seas_vacc_effective']+\
                      data['opinion_seas_risk'] - data['opinion_seas_sick_from_vacc']

    data['concern>=2'] = np.where(data['h1n1_concern']>=2,1,0)
    data['good_opinion_vacc'] = np.where(data['opinion_seas_vacc_effective'] == 3,1,0) # 5 before
    data['good_knowledge'] = np.where(data['h1n1_knowledge'] == 2,1,0)
    data['risk'] = np.where(data['opinion_h1n1_risk']>=4,1,0)
    data['concern_knowledge'] = data['h1n1_concern']+data['h1n1_knowledge']
    data['a^2'] = data['age_group']*data['age_group']
    ###### Dropping other features #########
    data.drop(['race','child_under_6_months','opinion_h1n1_sick_from_vacc','opinion_seas_sick_from_vacc','household_adults','behavioral_antiviral_meds','behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_antiviral_meds','marital_status',
           'behavioral_avoidance','behavioral_face_mask','income_poverty','hhs_geo_region','employment_status','education','census_msa'],axis=1,inplace = True)
    print(data.shape)
    Test_X = data
    return Test_X

@app.post("/predict")
async def predict(item: DataType):
    print("hi")
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    # print(df)
    data = scale_data(df)
    print(data)

    # data,temp = one_hot_encoder(df, nan_as_category=True)
    # print(data)
    # print(data.columns)
    # data.to_csv('datafile.csv')
    ans1 = model.predict(data)
    print(ans1)
    # print(ans1)

    # ans1 = list(ans1)
    # if ans1[0] == 0:
    #     return "Benign Prostatic Hyperplasia (BPH)"
    # else:
    #     return "Malignant Prostate Cancer (MPC)"
    return "done"
@app.get("/")
async def root():
    return {"message": "This API Only Has Get Method as of now"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
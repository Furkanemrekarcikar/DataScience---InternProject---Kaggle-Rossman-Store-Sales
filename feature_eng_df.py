from datasets.EDA import df
import pandas as pd

old_df = df.copy()

old_df = df.dropna(subset= ["CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"])

old_df['Promo2SinceWeek'] = old_df['Promo2SinceWeek'].fillna(0)
old_df['Promo2SinceYear'] = old_df['Promo2SinceYear'].fillna(0)

old_df.isnull().sum()

old_df.head(3)

old_df.drop(columns='Date', inplace=True)


str_type_dummies = pd.get_dummies(old_df['StoreType'], prefix='StoreType')
old_df = pd.concat([old_df, str_type_dummies], axis=1)

st_hol_dummies = pd.get_dummies(old_df['StateHoliday'], prefix='State')
old_df = pd.concat([old_df, st_hol_dummies], axis=1)

assortdummies = pd.get_dummies(old_df['Assortment'], prefix='Assort')
old_df = pd.concat([old_df, assortdummies], axis=1)

old_df.drop(columns=['StoreType', 'StateHoliday', 'Assortment'], inplace=True)
old_df.head(3)
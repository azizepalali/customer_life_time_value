##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma Models
##############################################################

# 1. Data Preprocessing
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (The number of multiple shopper / Total Customers)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Creation of Segments

##########################
# importing libaries
##########################

import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width',200)

# For Outliers
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


#########################
# Get the data about 2010-2011 years
#########################
df= pd.read_excel('hafta_3\online_retail.xlsx', sheet_name="Year 2010-2011")
df.head()

##############################################################
# Data Preperation
##############################################################

df.dropna(inplace=True)
# df = df[df["Country"] == "United Kingdom"]
df = df[df["Country"].str.contains("United Kingdom", na=False)]
df["Country"].nunique()
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]
df.describe().T

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)
df.head()

#############################################
# Creating the Data Structure for Life Time
#############################################

# recency: difference between first and last purchase. Weekly (for every single user)
# T: the age of the client in the company. Weekly (for every single user)
# frequency: total number of repeat purchases. Weekly (frequency>1)(for every single user)
# monetary_value: average earnings per purchase.Weekly (for every single user)


cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.head()   # index issue
cltv_df.columns = cltv_df.columns.droplevel(1)  # fixing index issue

# Defining columns names
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df.head()

#############################################
# Calculations
#############################################

# Average Order Value (average_order_value = total_price / total_transaction)
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
# cltv_df = cltv_df[cltv_df["monetary"] > 0]

# bir alışverişten fazla olması gerekiyor
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# hafftalık değerine ihtiyacımız var kendi içinde
cltv_df["recency"] = cltv_df["recency"] / 7

# hafftalık değerine ihtiyacımız var kendi içinde
cltv_df["T"] = cltv_df["T"] / 7

# cltv_df["frequency"] = cltv_df["frequency"].astype(int)

cltv_df.head()

##############################################################
# BG-NBD Model  (Expected number of transaction)
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

################################################################
# Who are the 10 customers we expect the most to purchase in a week?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])


################################################################
# Who are the 10 customers we expect the most to purchase in a month?
################################################################

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_1_month", ascending=False).head(20)

##############################################################
# GAMMA-GAMMA Modelinin Kurulması (expected average profit)
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])


ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

# top 10 most profitable customers
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)

##############################################################
# Calculation CLTV with BG-NBD and GAMMA GAMMA Models
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,   # 6 month
                                   freq="W", # Weekly
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by= "clv", ascending=False).head()

# 50 most valuable customers in a 6-month
cltv.sort_values(by="clv", ascending=False).head(50)

################################################################
# Analyze the top 10 people at 1 month CLTV and the 10 highest at 12 months.
################################################################

a= ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 month
                                   freq="W",  # Weekly
                                   discount_rate=0.01)

b= ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

a.head()
b.head()

a.reset_index().head(10).sort_values(by="clv",ascending=False)
b.reset_index().head(10).sort_values(by="clv",ascending=False)

################################################################
# Addition to the CLTV dataset.
################################################################

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Crossing customer information
cltv_final[cltv_final["Customer ID"] == 14298.00000]
cltv_final[cltv_final["Customer ID"] == 14096.00000]

# apply them scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
cltv_final.head()

# CLTV
cltv_final.sort_values(by="scaled_clv", ascending=False).head()

##############################################################
# Creating Segments by CLTV
##############################################################

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)

# segment-based customer descriptive statistics
cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


##############################################################
# Functionalization of Work
##############################################################

def create_cltv_p(dataframe, month=6):
    # 1. Preperation
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. Calculation CLTV with BG-NBD and GAMMA GAMMA Models.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 month
                                       freq="W",  # Weekly
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

cltv_final2 = create_cltv_p(df)


##############################################################
# Sonuçların Veri Tabanına Gönderilmesi
##############################################################
# credentials.
creds = {'user': 'group_04',
         'passwd': 'hayatguzelkodlarucuyor',
         'host': '34.88.156.118',
         'port': 3306,
         'db': 'group_04'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))
import mysql.connector

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)


cltv_final2.head()

cltv_final2["Customer ID"] = cltv_final["Customer ID"].astype(int)

cltv_final.to_sql(name='azize_sultan_palali', con=conn, if_exists='replace', index=False)

pd.read_sql_query("show tables", conn)

pd.read_sql_query("select * from azize_sultan_palali limit 10", conn)

# conn.close()

########################
# A grubu Segment
########################

# A grubu segmentine ait müşteriler; expected average profiti en yüksek müşteriler
# Bu müşteriler clv değeri de en yüksek segment

# İlk 3 ay içerisinde alınabilecek aksiyonlar;

# Özel olduklarını ve bizimle uzun zamandır bir arada olduklarını hatırlatmalıyız.
# Anket çalışması yapıp geri dönüşlerine göre süreçlerimizi iyileştirebiiriz.
# Yeni bir ürün çıktığında haberdar edebiliriz.
# Müşteri sadakat programı oluştuturup kendilerine özel ayrıcaklıklar tanımlanabilir.


# Sonraki 3 ay içerisinde alınabilecek aksiyonlar;

# İlk 3 ay içerisinde alınan aksiyonlar, çıktılarına göre devam ettirilebilir
# Kişiselleştirilmiş mail atılabilir, telefonla özel olarak arama yapılabilir.
# cross sellde yüksek indirim tanımlanabilir.
# En kaliteli ve pahalı ürünler bu gruba önerilebilir.


########################
# D grubu Segment
########################
# eski müşterilerimiz olmasına rağmen clv değerleri çok düşük
# Bu segmentinde kendi içerisinde segmentasyona ihtiyacı olabilir.

# İlk 3 ay içerisinde alınabilecek aksiyonlar;

# Alışveriş yapma isteklerini oluşturacak aksiyonlar almalıyız.
# Kendimizi hatırlatmalıyız, özel günlerde mesaj atabiliriz.
# Mobil ya da web uygulaması üzerinden pop uplar çıkarılabilir.
# Özel günlerde indirim sağlanabilir.


# Sonraki 3 ay içerisinde alınabilecek aksiyonlar;

# İlk 3 ay içerisinde alınan aksiyonlar, çıktılarına göre devam ettirilebilir
# Alışveriş ihtiyacını oluşturduktan sonra indirimler tanımlanabilir.
# Yeni ve mevcut ürünleri düşük indirimlerle önerebiliriz.
# Anket çalışması yapıp geri dönüşlerine göre süreçlerimizi iyileştirebiiriz.
# Özel günlerde indirim sağlanabilir.
# Kişiselleştirilmiş mail atılabilir, telefonla özel olarak arama yapılabilir.
# ML_4G_service_prediction

Please see the coding file attached or reach the link below:

https://colab.research.google.com/drive/1LRZm6U5MVzR6KA0nK2qaWbN23kW9uoES#scrollTo=962e9ee5

## I. Introduction

### 1. Business question

A telecommunications corporation wants to build a model to predict which customers will use its 4G service in the next month.

### 2. Dataset

The dataset records the usage characteristics of 200.000 customers for 10 consecutive months. Each customer is labeled as using 4G or not.
Dataset include these following main fields:
- CUOC_GOC_GPRS_220601: Customer data revenue generated when using exceeding the allowed traffic
- HA_TANG_220601: The infrastructure used by customer. 2G-3G is an old technology, 4G is a new technology in the future
- IS_SIM_4G_220601: Customers who already have a 4G SIM (Necessary condition for TB to be able to move to 4G infrastructure)
- LL_THOAI_220601: Voice traffic customers usage
- NOD_PSLL_THOAI_220601: Number of days using voice calls in that month
- NOD_PSLL_DATA_220601: Number of days using data in that month
- SO_LAN_NAP_THE_220601: Number of times to top up the card (top up the account for consumption)
- SO_LAN_NAP_TOPUP_220601: TOP up is a modern form of top-up via banks or digital tools, customers with no value ie top up in the traditional way.
- SO_NGAY_SU_DUNG_220601: Number of days using one of their services: Voice, Data, SMS.
- THIET_BI_220601: Device customers are using: 2G devices, 3G devices, 4G devices (Only customers with 4G devices can convert customers to 4G)
- thuc_4g_220601: This field is the field that needs to be used to check whether the customer is actually 4G or not ==> I need to forecast that the group of customers with a value of 0 has the potential to convert to 1 in the next month.
- TONG_CUOC_GOC_DATA_4_HUONG_22060: Spending of customers for Data services (VND)
- TONG_CUOC_GOC_FN_220601: Customer's spending for all services provided on customer's phone number (Voice, SMS, Data, Vas) (VND)
- TONG_CUOC_GOC_THOAI_220601: Spending of customers on voice services (VND)
- TONG_LL_GPRS_220601: Customer's data usage behavior (Unit GB) - like the type of customer's unit of measure when watching youtube/going to facebook.
- TONG_TIEN_NAP_THE_220601: Amount of money deposited into the account by the customer to spend
- TONG_TIEN_NAP_TOPUP_220601: Amount deposited into account by modern methods (Bank, digital payment)
- TUOI_KHACH_HANG_CUT_LEVEL_220601: Age of customer
- user_id: subscriber ID (This is the encoding of the phone number)

![image](https://github.com/thuhuongphan11/Python_Cohort_Analysis/assets/141643891/1789ee39-277f-42bb-9fe3-0cb16784541c)

### 3. Method
Supervised learning with Scikit-learn on Python
- Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately.
- As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process.
- Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

## II. Proccess
### 1. Cleaning and transforming dataset
- The maximum number of null values in each column is low: 137/200000 values => Remove null values from dataset
- Values in 'user_id' column was changed from original data for security, so some duplicated values appear => Remove those duplicate values
- Original data is organized as wide form, each field is devided into 10 columns to show infomation of 10 months => Convert dataframe form wide to long form for better manipulation. Then calculate the mean, mode of each column group by 'user_id' to combine 10 months into a single row

**Dataset after transforming**

![image](https://github.com/thuhuongphan11/Python_Cohort_Analysis/assets/141643891/7f001184-3913-4d25-8898-5ad6ff100b40)
### 2. EDA and select features of the model
#### EDA
![image](https://github.com/thuhuongphan11/Python_Cohort_Analysis/assets/141643891/11ea5be8-1ba8-4c81-8922-de397a98db39)

- The heatmap shows that 'HA_TANG_', 'THIET_BI_', 'NOD_PSLL_DATA_', 'SO_NGAY_SU_DUNG_' have a high correlation with the target column 'thuc_4g_'. But 'HA_TANG_' and 'THIET_BI_' are correlated, -> we choose 1 of these 2 features.   
  => Choose 'HA_TANG_', 'NOD_PSLL_DATA_' and 'SO_NGAY_SU_DUNG_' to apply to the model.
- There are some columns that we think may affect the target "thuc_4g" ('NOD_PSLL_THOAI_', 'TUOI_KH_', 'IS_DCOM_') => EDA to check the relation with the "thuc_4g" column.
#### Features selection
After EDA we will keep the below columns to the model:
- 'HA_TANG_': The infrastructure used by customer. 2G-3G is an old technology, 4G is a new technology in the future
- 'NOD_PSLL_DATA_': Number of days using data in that month
- 'NOD_PSLL_THOAI_': Number of days using voice calls in that month
- 'SO_NGAY_SU_DUNG_': Number of days using one of their services: Voice, Data, SMS
- 'thuc_4g_': target column
## III. Model training and evaluation
- Encoding & Normalizing dataset
- Apply to models: Logistic Regression, K Nearest Neighbors (KNN), Decision Tree and Random Forest

## IV. Conclussion
Comparing the balance accuracy of 4 models, we can see that Logistic Regression has the highest test set's F1-score (0.929). Logistic Regression also has the least difference between the F1-score of the test set and the train set (0.929 and 0.93, respectively).
=> Choose **Logistic Regression** as the final model used to predict 4G service customers for this corporation.

***Confusion matrix of Logistic Regression model***
![image](https://github.com/thuhuongphan11/Python_Cohort_Analysis/assets/141643891/de737434-9ba1-4bc2-8bb2-f62028c9f1a2)

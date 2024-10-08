import numpy as np
np.float_ = np.float64
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score
def mean_absolute_percentage_error(y_true,y_pred):
    y_true,y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
pd.set_option('display.max_columns',20)
df = pd.read_excel('count_model.xlsx', index_col=[0], parse_dates=[0])
st.set_page_config('Prophet Model')
st.title('Predictions for number of items picked by the IDC using the Facebook Prophet Model')
st.write('Ibrahim Quaizar')
st.write('August 13th, 2024')
st.header('Abstract')
st.write('''Pick Lines are used by the IDC to know how many items must be picked from the shelves on a daily
         basis. Knowing how many picks they may have to do beforehand can be beneficial to them and their
         resources. We can use a Facebook prophet model in order to forecast this data for them. With a
         rough accuracy of 94.8%, and compensations coming into mind, this model can prepare the IDC 2 years
         in advance, allowing them to more effictively allocate items and workers.''')
st.header('Brief summary')
st.write('''Pick lines are the amount of items that the IDC(A storage facility) has to pick from the shelves each day. This may be to ship out to sales, or for whatever other reason.
            The dataset being manipulated is one that shows the number of pick lines on a daily basis. The goal of this project is to try to predict how many picks the IDC might have to do in the
            future so that they are more readily able to allocate workers and items. Based on research, the Facebook Prophet model may be the best to do this because it forecasts data working with
            time very well.''')
         
#print(df.head())
#Graphing just to see
#plt.plot(df.index,df['pick_lines'])
#plt.show()

#Train/Test split
train_size = int(len(df) * 0.7)
train_data, test_data = df[0:train_size], df[train_size:len(df)]
#print(train_data.head())
#print(test_data.head())

train_data_proph = train_data.reset_index().rename(columns={'demand_date':'ds','pick_lines':'y'})
test_data_proph = test_data.reset_index().rename(columns={'demand_date':'ds','pick_lines':'y'})


#print(train_data_proph.head())
#print(test_data_proph.head())
 
mod = Prophet()
mod.add_country_holidays(country_name= 'US')
mod.fit(train_data_proph)
y_pred = mod.predict(test_data_proph)
new_pred = y_pred[['ds','yhat']]
new_pred = new_pred.rename(columns={'yhat':'y'})
#pd.to_datetime(test_data_proph['ds'])
#pd.to_datetime(new_pred['ds'])
x = test_data_proph['ds'].values.astype('float64')
y = new_pred['ds'].values.astype('float64')
new_pred['ds'] = y
test_data_proph['ds'] = x
#print(new_pred.dtypes)
#print(test_data_proph.dtypes)
accuracy = r2_score(test_data_proph,new_pred)


test_data_proph['ds'] = pd.to_datetime(test_data_proph['ds'])
#print(y_pred.head())
revTable = y_pred[['ds','yhat','yhat_lower','yhat_upper']]
revTable = pd.merge(test_data_proph,revTable, on = 'ds')

revTable = revTable.rename(columns = {'ds':'Date', 'y':'actual_pick_lines', 'yhat':'predicted_pick_lines',
                                      'yhat_lower':'lower_bound','yhat_upper':'upper_bound'})

rangee = []
for index, row in revTable.iterrows():
    if row['actual_pick_lines'] >= row['lower_bound'] and row['actual_pick_lines'] <= row['upper_bound']:
        rangee.append('within_range')
    elif  row['actual_pick_lines'] > row['upper_bound']:
        rangee.append('over_range')
    elif row['actual_pick_lines'] < row['lower_bound']:
        rangee.append('under_range')
revTable['Range_Status'] = rangee
#print(revTable.head())

fig, ax = plt.subplots(figsize = (10,5))
fig1 = mod.plot(y_pred,ax=ax, xlabel = 'Date',ylabel = 'Pick Lines')
plt.title(label = 'Training & Testing Data')
#plt.show()
fig2 = mod.plot_components(y_pred)
#plt.show()
st.header('Table visualization')
st.write('''Table showing predicted and actual pick lines from the testing dataset, along with a lower and upper bound. Last column
         shows an analysis if the actual pick lines are within the upper and lower bounds or not. This helps
         with figuring out how our model worked, or some workarounds we may need to take into consideration
         when trusting these predictions.''')
st.dataframe(revTable)

st.header('Model Visualization through graphs')
st.write('''Graph showing testing and training data done with the model to achieve prediction results. 
         Upper and Lower bound shades shown. As indicated, the model does predict the training data
         well, with a few outliers stemming from unforseen events. When checking the accuracy using an 
         accuracy model, we got an accuracy score of 94.8%, showing a strong prediction set.''')
st.pyplot(fig1)
st.write('''Components of the above graph, showing the overall trend, weekly tend, and trends near the holidays.
         From the overall trend, we can gather that the model is fitting to an upwards trend, which becomes
         important when predicting future years. As for the days of the week, We can see that no production
         happens on weekends, and that production tends to be highest on Thursdays.''')
st.pyplot(fig2)
f,ax = plt.subplots(figsize = (15,5))
ax.scatter(test_data.index,test_data['pick_lines'],color = 'r')
fig3 = mod.plot(y_pred,ax=ax)
plt.xlim(pd.to_datetime('2023-10-21'),pd.to_datetime('2024-07-28'))
plt.title('Testing Data Predictions')
plt.xlabel('Dates')
plt.ylabel('Pick Lines')
#plt.show()

st.write('''Zoomed in graph to show just the predicted quantities. Date Range: 10/21/2023 - 07/28/2024.
         Here, we can see a more concrete trendline of the model fitting, and see how many of the predictions
         were accurate. The red dots represent the actual quantities while the blue line represents the
         predicted quantities.''')
st.pyplot(fig3)
st.header('Model error justification & explanation')
range_count = revTable['Range_Status'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(x = range_count.index, y = range_count.values, alpha = 0.8, palette = ['red','blue','green'])
plt.title('Range Status Count')
plt.xlabel('Range Status')
plt.ylabel('Count in dataset')
st.write('''Bar Graph showing how many of the actual values are either within, above, or below the
         predicted range. As we can see, majority of the values predicted by the model were within range.
         The under range values may be ignored because there is no issue with picking more items than we need to. 
         The over range values are where the main problem stems from, as it is an issue if
         we pick less items than we have to. However, this can be justified by the 
         fact that the extra supplies from the under range values may compensate for them in the future.''')
st.pyplot(plt)
st.write('''Within Range: 160 instances(~58%)''')
st.write('''Under Range: 80 instances(~28%)''')
st.write('''Over Range: 41 instances(~14%)''')
st.write('''If we take within range and under range as both acceptable metrics, we have an overall accuracy rate of roughly 86%, but can still be discounted
            if we include the fact that the over range will compensate for the under range, so it is acceptable as well.''')

st.header('Future Predictions')
future = mod.make_future_dataframe(periods = 800, freq = 'd', include_history= False)
future_fcst = mod.predict(future)
#print(future_fcst.tail())

fig, ax = plt.subplots(figsize = (10,5))
fig = mod.plot(future_fcst, ax=ax)
plt.title('Future Predictions')
plt.xlabel('Date')
plt.ylabel('Pick Lines')
#plt.show()
st.write('''Predictions for the dataset until the end of 2025 shown below. Here we can see that the prophet model 
         fitted to an upward curve shown from the years 2021-2023. This made it so that the amount of items 
         predicted would increase. On a company level, this means that in the future, more items may be needed
         to be picked by the IDC. This would possibily require us to allocate more space for items, and improve 
         funding and costs, while also increasing staffing.''')
st.write('''Looking at the increasing model, we can actually predict how many more workers we need. We can see that on average,
        the amount of pick lines increased by 25% than the current lines. Because of this, we would need 25-30% more workers in 
        order to ensure proper workflow. Using the base line of 40 picks per hour for one person, and 40 hours a week, 30% more workers should compensate
        for the extra items being picked''')
st.pyplot(fig)
st.write('''From analyzing the components of the future predictions, we can say with more confidence
         that our future predictions work. Noticing the trend line and weekly trend, it maps just about the
         same as the previously predicted test model. Like before, there was an upward general trend, along with
         Thursdays being the day that the most production occured. ''')
fig2 = mod.plot_components(future_fcst)
st.pyplot(fig2)
f,ax = plt.subplots(figsize = (15,5))
fig3 = mod.plot(future_fcst,ax=ax)
st.write('''Below is a zoomed in graph of just the future data, ranging from August 15th 2024 to
         December 1st 2025. There are no red dots like the last one due to the fact that this is future
         data, and there was no existing dataset to compare this to. This graph allows us to grasp a further
         understanding of the trend shown by this model in the next year or so.''')
plt.xlim(pd.to_datetime('2024-08-15'),pd.to_datetime('2025-12-01'))
plt.title('Future Data Points')
plt.xlabel('Dates')
plt.ylabel('Pick Lines')
st.pyplot(fig3)

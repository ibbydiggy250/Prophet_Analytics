import numpy as np
np.float_ = np.float64
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score
def mean_absolute_percentage_error(y_true,y_pred):
    y_true,y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
pd.set_option('display.max_columns',20)
df = pd.read_excel('count_model.xlsx', index_col=[0], parse_dates=[0])
st.set_page_config('Prophet Model')
st.title('Predictions for number of items vended using the Facebook Prophet Model')
st.write('Ibrahim Quaizar')
st.write('August 13th, 2024')
         
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
print(train_data_proph.head())
print(test_data_proph.head())
 
mod = Prophet()
mod.add_country_holidays(country_name= 'US')
mod.fit(train_data_proph)
y_pred = mod.predict(test_data_proph)


print(y_pred.head())
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
print(revTable.head())

fig, ax = plt.subplots(figsize = (10,5))
fig1 = mod.plot(y_pred,ax=ax, xlabel = 'Date',ylabel = 'Pick Lines')
plt.title(label = 'Training & Testing Data')
plt.show()
fig2 = mod.plot_components(y_pred)
plt.show()
st.header('Table visualization')
st.write('''Table showing predicted and actual pick lines, along with a lower and upper bound. Last column
         shows an analysis if the actual pick lines are within the upper and lower bounds or not. This helps
         with figuring out how our model worked, or some workarounds we may need to take into consideration
         when trusting these predictions.''')
st.dataframe(revTable)

st.header('Model Visualization through graphs')
st.write('''Graph showing testing and training data done with the model to achieve prediction results. 
         Upper and Lower bound shades shown. As indicated, the model does predict the training data
         well, with a few outliers stemming from unforseen events.''')
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
plt.show()

st.write('''Zoomed in graph to show just the predicted quantities. Date Range: 10/21/2023 - 07/28/2024.
         Here, we can see a more concrete trendline of the model fitting, and see how many of the predictions
         were accurate. The red dots represent the actual quantities while the blue line represents the
         predicted quantities.''')
st.pyplot(fig3)
st.header('Model error justification & explanation')
range_count = revTable['Range_Status'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(x = range_count.index, y = range_count.values, alpha = 0.8, color = 'red',)
plt.title('Range Status Count')
plt.xlabel('Range Status')
plt.ylabel('Count in dataset')
st.write('''Bar Graph showing how many of the actual values are either within, above, or below the
         predicted range. As we can see, majority of the values predicted by the model were within range.
         The under range values may be ignored because there is no issue with selling vendors 
         more items. The over range values are where the main problem stems from, but can be justified by the 
         fact that the extra supplies from the under range values may compensate for them in the future.''')
st.pyplot(plt)

st.header('Future Predictions')
future = mod.make_future_dataframe(periods = 800, freq = 'd', include_history= False)
future_fcst = mod.predict(future)
print(future_fcst.tail())

fig, ax = plt.subplots(figsize = (10,5))
fig = mod.plot(future_fcst, ax=ax)
plt.title('Future Predictions')
plt.xlabel('Date')
plt.ylabel('Pick Lines')
plt.show()
st.write('''Predictions for the dataset until the end of 2025 shown below. Here we can see that the prophet model 
         fitted to an upward curve shown from the years 2021-2023. This made it so that the amount of items 
         predicted would increase. On a company level, this means that in the future, more items may be needed
         to be vended. This would possibily require us to allocate more space for items, and improve 
         funding and costs, while also increasing staffing.''')
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

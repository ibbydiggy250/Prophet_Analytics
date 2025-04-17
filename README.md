This app is streamlit based.
In order to access the website please click on this link:
https://prophetprediction.streamlit.app/
The proph.ipynb file is based in the jupyter notebook. Both the proph.py and proph.ipynb file are included in the repository for reference.

The count_model.xlsx file is an excel based document, and just contains the original dataset used in training and testing this prophet model.

The intention of this project was to create a prediction model for Northwell Health using the Facebook Prophet Model. The data being predicted was the number of items being picked from a storage facility across the next 2 years for the company. This allows the company to more efficiently allocate items, resources, and workers based on how busy the model predicts the storage facility will be. Using the time series data provided, the model uses the number of picked items per month in order to generate its predictions. Part of the predicted portion of the model is set for the past 2 months in order to test accuracy, which yielded 94.8%. 

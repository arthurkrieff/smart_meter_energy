# Smart Meter by BCM Energy

This project was developed in the context of our Machine Learning II class at Ecole Polytechnique. It is extracted from the ENS Data Challenge website (https://challengedata.ens.fr/participants/challenges/29/). The goal is to predict the energy consumption of four home appliances (washing machine, fridge/freezer, TV and kettle) at a specific timestamp. 

# Dataset
The dataset contains an observation every minute and the main features are the following:
- Overall consumption
- Temperature
- Wind
- Other weather information.. 

# How to use ?
This repository contains four main files, one for each home appliance. In each python file there is a class containing at least: 
- a **transform** method: applies transformations to the dataframe before feeding it to the model
- a **fit** method: fit the regression model
- a **predict** method: predict the energy consumption of the appliance

# Credits
This project was made with the contribution of Aicha BOKBOT and Arthur KRIEFF. 

# West Nile Virus (WNV) in Chicago
![alt text](https://images.medicinenet.com//images/slideshow/west-nile-virus-s1a-photo-of-tem-and-culex-mosquito.jpg "West Nile")
West Nile Virus has been in Chicago for years. How can we help the city? Chicago has collected data on the following:
* Mosquito traps
* Weather
* Pesticide spraying
According to research, the pesticide spraying had little effect on the mosquitoes. So we dropped the spray data. After cleaning the weather data, we combined it with the mosquito trap data.
(Dataset source: https://www.kaggle.com/c/predict-west-nile-virus/data)

The team created a model based on a neural network to help field scientists test mosquito traps. Currently, the field scientist tests every mosquito trap. With our model, the field scientist would only have to **test a trap for WNV 27% of time**, instead of 100%. This enables Chicago with more resources to combat WNV.

The neural network is a feed-forward deep neural network. The hidden layer has three layers:
1. Twice the number of neurons as the input layer with 25% dropout, RELU activation
2. Same as the previous layer
3. Ten neurons  with 25% dropout, RELU activation
4. Output is a single neuron with a sigmoid activation function to predict the probability of WNV presence in the the trap

With a threshold level of 0.42 set for the prediction, the model produced the following confusion matrix:

                        Predicted 'No WNV Present'    Predicted 'WNV Present'
    WNV was not Present           1532                          460
    WNV was Present               0                             110

The results focused on zero false negatives. In other words, nothing should slip through the cracks.

If the model predicts no WNV, then we have a good reason to believe that WNV is not present. If the model says "WNV is Present", there's a 19% chance that WNV is actually present.

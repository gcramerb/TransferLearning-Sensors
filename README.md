# TransferLearning for Sensor data

There is a lot of heterogeneity in the sensor data due to the enormous ways to acquire data. 
This fact creates a scenario that the train an application data may be very different. 
To face this issue, transfer Learning techniques are essential. 

This repository applies some transfer learning tools in the context of sensor data.
We use the Sinkhorn divergence as a distance metric between the source data probability and the target one. 
It was used as a loss function of the Deep model. This loss aims to force the model to learn the same discriminative features in both source and target datasets.
We achieve some important and consistent results, but these techniques do not generalize well.

We also use a soft-label technique. That uses the predicted labels of the target domain to train the model iteratively.
This archive a very solid results. 
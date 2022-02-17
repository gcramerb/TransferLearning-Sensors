# TransferLearning for Sensor data

There is a lot of heterogeneity in the sensor data due to the enoumerous ways to aquire data. 
This fact creates a scnario that tha train an appication data may be very different. 
To face this issue, transfer Learning techniques are essencial. 

This repository apply some transfer learning tools in the context of sensor data.
We use the Sinkhorn divergence as distance metric between the source data probability and the target one. 
It was used as loss function of the Deep model. This loss aims to force the model to learn the same discriminative features in both source and target datasets.
We achive some important and consistent results, but this techniques do not generalize well.

We also use a soft-label technique. That uses the predicted labels of the target domain to train the model in a iterative way.
This achive a very solid results. 
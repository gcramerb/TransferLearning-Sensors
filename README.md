
# TransferLearning for Sensor data

This repository tackles the unsupervised domain task. A really important task in the machine learning context. 
We apply our technique in the wearable sensor data. There is a lot of heterogeneity in the sensor data due to the enormous ways to acquire data. 
This fact creates a scenario that the train an application data may be very different. 
To face this issue, transfer Learning techniques are essential. 

This repository applies some transfer learning tools in the context of sensor data.
We use the divergence metric between the source data 
probability and the target one. 
It was used as a loss function of the Deep model. 
This loss aims to force the model to learn the same
discriminative features in both source and target datasets.
We achieve some important and consistent results,
but these techniques do not generalize well.

We also implemented a Soft-label technique. We predicted a 
probable label of the target data and use it as a label in the way 
of supervised learning. We develop a novel method to choose 
the samples which the predicted labels are the most reliable. 

We also develop a solid classifier that works well in the four datasets
we had chosen: [USC-HAD](https://sipi.usc.edu/had/), [PAMAP2](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)
, [DSADS](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities) and [UCI-HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). 

### Discrepancy domain.

We use a divergence measurement of probability distributions as a loss function. 
The deep model uses the classical Cross entropy loss for the source data
and, for the target data, we use the [Sinkhorn](https://www.kernel-operations.io/geomloss/api/pytorch-api.html) divergence
as the measurement between source latent data and target latent data. 

Those two losses are responsible for the updating of the parameters. 
We achieve good results but it was not satisfactory, 
because in some changeling scenarios this metric do not 
transfer much relevant information from source to target. 

### Soft-Label.

We also use a soft-label technique. That uses the predicted labels of the target domain to train the model iteratively.
This archive a very solid results. We develop a novel 
method to select the most important target samples.
This method is based on kernel density estimation. 

# tensorboard_cifar10
Using tensorboard to visualize results of a simple CNN model trained on a CIFAR-10 dataset  

## Requirements

- Python 2.7
- Tensorflow > 0.12
- Numpy
- Keras

## Model and Training
The CNN model we will be using to classify the CIFAR-10 images is as follows:
Input → Conv(3,3,48) + relu → Conv(3,3,48) + relu → Max-pool(2,2) → Dropout(0.25)→ Fully Connected Layer(512) → Fully Connected Layer(256) → Scores → Predictions

We train the model using 3 different learning rates [1e-2, 1e-3, 1e-4].  

## Visualizing results in Tensorboard 
The summaries we keep track of are :
- Loss
- Accuracy
- Weights and their gradients 

Since we train the model for different learning rates, we write the summaries in each run to a different folder (named based on the learning rate) within runs/summaries. So if we point the tensorboard log directory to summaries, we should be able to compare summaries across these runs. 

Use the following command in the terminal:

```bash
tensorboard --logdir runs/summaries
```




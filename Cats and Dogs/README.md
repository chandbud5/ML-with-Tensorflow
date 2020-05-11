# Cats v/s Dogs Classifier

This binary classifier uses real world dataset from **Kaggle**.
It contains 25000 images of cats and dogs, for their classification I have made a 
model which uses ImageDataGenerator for Augmentation and Convolutional Neural Network<br><br>

## To see the effect of Augmentation we will see the plots in both the cases

**Before Augmentation**

<div id="banner">
    <div class="inline-block">
        <img src="https://github.com/chandbud5/ML-with-Tensorflow/blob/master/Cats%20and%20Dogs/Before_Aug_Acc-v-epo.png" width="400">
        <img src="https://github.com/chandbud5/ML-with-Tensorflow/blob/master/Cats%20and%20Dogs/Before_Aug_loss-v-epo.png" width="400">
    </div>
</div>

**After Augmentation**

<div id="banner">
    <div class="inline-block">
        <img src="https://github.com/chandbud5/ML-with-Tensorflow/blob/master/Cats%20and%20Dogs/Acc-v-Epochs.png" width="400">
        <img src="https://github.com/chandbud5/ML-with-Tensorflow/blob/master/Cats%20and%20Dogs/Loss-v-Epochs.png" width="400">
    </div>
</div>

### On the basis of above figure we can infer that
* Previously, our model was overfitted as training accuracy was around 100% and validation accuracy was quite low.
* While after applying Augmentation it saved our model from overfitting as due to augmentation our model gets variation in dataset.
* Accuracy was increasing slowly in augmentation as compared to previous one.

### Downloading dataset
* To get Dataset visit the given link 👉👉 [Kaggle Cats v/s Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)<br>
* To use smaller version of dataset(2000 training images) use the wget command used in this colab 👉👉 [Colab Link](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/b71119aba42aae8922d00124b8fbe5b7d71f4ec3/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)

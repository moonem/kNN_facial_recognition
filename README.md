# kNN & Facial Recognition

## Goal of this project:

This project will - 
  - discuss and develop functions to find Euclidean Distance between data points to find nearest neighbors; 
  - identify the applicability, assumptions, and limitations of the k-Nearest Neighbors (k-NN) algorithm;
  - build a Facial Recognition system using the k-NN algorithm;
  - Implement loss functions to compute the accuracy of an algorithm


> **_Content Credit_**: Most of the contents are a part of the course "Problem-solving with Machine Learning" taught by Professor Kilian Weinberger from the Cornell University. This project is summarized key contents and intended for learning purpose only.

## Topics and Tools:

Supervised Machine Learning, Linear Algebra, Vector and Matrix opearations, Python, NumPy, Jupyter Notebook.

## Supervised Machine Learning:

Supervised learning is when we have a specific data set and we know exactly what we want to predict. So we want to teach the algorithm to predict a very specific thing.
Supervised machine learning uses inputs, referred to as *features* (`X`), and their corresponding outputs, referred to as *labels* (`y`), to train a function (`h`)
that can predict the label of a previously unseen data point.

$$X -> h -> y$$

For example, our *features* could describe a patient in a hospital (e.g., gender, age, body temperature, various symptoms) and the *label* could be if the patient is sick or healthy. You can use data from past medical records to learn a function (`h`) that is able to determine a future patient's diagnosis based on their symptoms.

For an incoming patient, when we observe features (`X`), we can apply the function (`h`) to predict whether this new patient is sick or healthy (`y`).

The initial stage where we use existing medical records to learn a function is called the **training** stage, and the latter where we apply the function to a new patient is called the **testing** stage.

### Hypothesis or Function:

The function, often referred to as hypothesis and denoted as `h`, is the program that is learned from the data.

### Features or `X`:

- Features are the relevant characteristics or attributes that we believe may contribute to the outcome of an event.
- In this project, we assume that features are stored as a `d`-dimensional vector of feature values.
- How a data instance is encoded into a vector and what data the vector contains will usually influence the outcome of the machine learning process.
- The examples of feature vectors provided are bag-of-words features, pixel features, and heterogeneous features.

#### Examples of feautres: Heterogeneous

Patient data that could include the patient's age in years, blood pressure, and height in centimeters. For instance, blood pressure and height are of different scales; a blood pressure reading that is 10 units higher than the average and a height measurement of 10 units higher than the average have very different implications. This difference in scale should be taken into account.

#### Bag-of-words features 

Text documents are often stored as bag-of-words features.This is a method to convert a text document with any number of words into a feature vector of fixed dimensionality. Before learning begins, one agrees on a finite set of possible words of interest, such as the =100,000 most common words in the English language. The text document is then scanned for these words and represented as a vector of word counts.

#### Pixel features 

Images are typically stored as pixels. These can be represented as a vector by simply “vectorizing” the image in one long chain of numbers. If an image has six megapixels, and each pixel has three numbers (one for red, green, blue) this yields an 18 million dimensional vector.

#### Examples of feature vectors

We call $x_i$ a feature vector and $d$ the dimensions of that feature vector that describe the $i^{th}$ sample.
We call $x_i$ a feature vector and $d$ the dimensions of that feature vector that describe the $i^{th}$ sample. For example, if we consider patient data in a hospital as follows,

  1) First patient, Mysha Ahmed ($i^{th}$ patient, here $i$ = 1, $[x_1]$), female (first dimension $d=1$; encoded $[x_1]_1 = 0$ for female, $1$ for male), height 165 cm (second dimension $d = 2$; encoded as $[x_1]_2 = 165$), 23 years old (third dimension $d = 3$; encoded $[x_1]_3 = 23$), label healthy ($y_i = y_1 = -1$ for healthy, and $+1$ for sick).
  
        $$x_1 = [1, 165, 23],  y_1 = -1$$
        
  2) Second patient, Raif Jamil ($i^{th}$ patient, here $i$ = 2, $[x_2]$), male (first dimension $d = 1$; encoded $[x_2]_1 = 0$ for male), height 167 cm (second dimension $d = 2$; encoded as $[x_2]_2 = 165$), 65 years old (third dimension $d = 3$; encoded $[x_2]_3 = 65$), label healthy ($y_i = y_2 = +1$ for sick).
  
        $$x_2 = [0, 167, 65],  y_1 = +1$$
        
A feature vector is called **dense** if $x_i$ has large number of non-zero components (or coordinates), and $x_i$ is called **sparse** if it consists mostly of zeros.

### Labels or `y`:

The label (`y`) is what we want to predict for a given data instance. Labels can come in many different forms such as,

#### Binary

There are only two possible label values. For example, with spam email classification, an email is either spam or not spam. Spam could be mapped to "+1" and email not considered spam to "-1."

#### Multiclass

There are multiple distinct label values. For example, in a facial recognition application, we would distinguish each individual as a separate class, such as:
- class1 = ”Bill Gates”
- class2 = ”Steve Jobs”
- class3 = ”Linus Torvalds”

### Loss Function:

- There are typically two steps in learning a hypothesis function: selecting the algorithm and finding the best function within the class of possible functions
- A loss function evaluates a hypothesis on our data and tells us how good or bad it is, helping us choose the best function

Three examples of loss function are *zero-one, squared,* and *absolute* losses.

### Zero-One Loss

The simplest loss function is the zero–one loss. It literally counts how many mistakes a hypothesis function makes on a particular data set. For every single example that is predicted incorrectly, it suffers a loss of 1. The normalized zero–one loss returns the fraction of misclassified training samples, also referred to as the *training error*.

$$L_{0,1}(h) = \frac{1}{n} \sum_{i=1}^n \delta_{h(x_i) \neq y_i}$$

where, $$\delta_{h(x_i)\neq y_i = 1, if h(x_i) \neq y_i, \\
                  y_i = 0, otherwise. $$

The zero-one loss is often used to evaluate classifiers in *multiclass/binary classification* settings but rarely useful to guide optimization procedures because the
function is non-differentiable and non-continuous.

### Squared Loss

The squared loss function is typically **used in regression** settings. It iterates over all training samples and suffers the loss $(h(x_i) - y_i)^2 $. The
squaring has two effects:
- The loss suffered is always non-negative,
- The loss suffered grows quadratically with the absolute mispredicted amount; can be **problematic for data with noisy labels**.

$$L_{sq}(h) = \frac{1}{n} \sum_{i=1}^n (h(x_i)-y_i)^2$$

### Absolute Loss

Similar to the squared loss, the absolute loss function is also typically used in regression settings. Because the suffered loss grows linearly with mispredictions, it is more **suitable for noisy data**.

$$L_{abs}(h) = \frac{1}{n} \sum_{i=1}^n |h(x_i)-y_i|$$

#### Which Loss Function should we use?

Choosing the right loss function is a critical step in solving a problem with machine learning. Considerations include,
  - first and foremost, determining whether the problem is a classification or regression problem;
  - then, if the labels in the training data are accurate or if the data is noisy;
  - if Regression, it is additionally worth considering if it is better to make many tiny but few large mistakes (**squared loss**);
  - or if it is OK to tolerate some larger mistakes yet drive most errors down to zero when possible (**absolute loss**).

### Generalization not Memorization: Split data into Training-Validation-Test

In machine learning, we incentivize an algorithm to do as well as possible on one particular data set. So what they can do is they can just memorize that data set. Essentially, what they're learning is a database, but they just look up the samples and say, "Oh, for that sample, the label is the following." We want to avoid
this from happening. To avoid this we split the data into two parts: **training** and **test**. Usually, it's an 80-20 split. We only let it train on the 80% training data. and we never show the test data. And once we are convinced it's a good algorithm, then we evaluate it; not on the training data, but on the test
data.

the test data is really the kind of data that you may see during deployment. So we can't just have all data of one kind going training, and all data of another kind during testing. We need to just shuffle the data uniformly at random, and then we just split it 80-20 into train and test.

But shuffling may not be appropriate if the data is **temporal** type, that means data was collected over time, and it changes over time such as, email spam. So in this case, the training data has to be past data; for example, data collected on Monday through Thursday. And then test data has to be future data; for example, Friday through Sunday. Another setting may be where we have medical data. Here, we want to make sure that we *split on patients* so that the training data consists of all measurements made by one set of patients. But then the test data is actually totally different patients because we want to make sure that our algorithm
then works on new patients that come into the hospital. We need to make sure that we don't predict the past from the future data, always future from the past.

there is another caveat when we do the splitting to training and test, we are only allowed to look at the test data once. If we look at the test data multiple
times and tweak our algorithm to do well on test data it will overfit the function on this particular dataset and not necessarily improving its accuracy on new data. To avoid overfitting to the training set, we should usually split the data into three mutually exclusive subsets: 

      - Training Data, $D_{TR}$ - 80%
      - Validation Data, $D_{VA}$ - 10%
      - Test Data, $D_{TE}$ - 10%

We then choose a function based on the training data, improve it using the validation data, and evaluate it on the test data. The validation data set is a proxy for the test set.

## Training the Function

We choose a function $h$ to minimize the training loss:

$$h^*(.) = argmin_{h\epsilon \mathcal{H}} \epsilon_{TR} (h),$$

$$\epsilon_{TR} (h) = \frac{1}{|D_{TR}|} \sum_{(\mathbf{x},y)\epsilon D_{TR}} \mathcal{l}(\mathbf{X},y|h(.))$$

Here, $\mathcal{H}$ is the hypothetical class (i.e., the set of all possible classifiers $h(.)$). In other words, we are trying to find a hypothesis $h$ which would
have resulted in the lowest possible value ($argmin$) of the loss function ($\mathcal{l}$) averaged over all the points in the training set which we will call $\epsilon_{TR}$.

## Evaluating the Function

Evaluate the function through the testing loss:

$$\epsilon_{TE}(h) = \frac{1}{|D_{TE}|} \sum_{(\mathbf{X},y) \epsilon D_{TE}} \mathcal{l}(\mathbf{X},y|h*(.))$$

The function $ \epsilon_ {TE} $ is the average of the loss function over all points in the
test set.

## Generalization

The testing loss is an unbiased estimator (i.e., an approximation) of the **generalization loss** — the loss over unseen data. Minimizing the generalization loss is our true objective but cannot be done directly:

$$\epsilon (h) = \mathbb{E}_{(\mathbf{X},y) ~ \mathcal{P}} [\mathcal{l} (\mathbf{X}, y|h(.))]$$

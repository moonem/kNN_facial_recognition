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

<div style="text-align: justify">
~~~
**X** &rarr; **h** &rarr; **y**
~~~
</div>

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

We call `$x_i` a feature vector and `d` the dimensions of that feature vector that describe the $i^{th}` sample. For example, if we consider patient data in a hospital as follows,

  1) Mysha Ahmed (`$i^{th}` patient, here `$i$ = 1`), female (first dimension `$d = 1$`; encoded `$[x_i]_1 = 0` for female, $1$ for male), height 165 cm (second dimension `$d = 2$`; encoded as `$[x_i]_2 = 165'), 23 years old (third dimension `$d = 3$`; encoded `$[x_i]_3 = 165'), .
  $$ x_1 = [1, 165, 23]
  3) 

### Labels or `y`:

The label (`y`) is what we want to predict for a given data instance. Labels can come in many different forms such as,

#### Binary

There are only two possible label values. For example, with spam email classification, an email is either spam or not spam. Spam could be mapped to "+1" and email not considered spam to "-1."

#### Multiclass

There are multiple distinct label values. For example, in a facial recognition application, we would distinguish each individual as a separate class, such as:
- class1 = ”Bill Gates”
- class2 = ”Steve Jobs”
- class3 = ”Linus Torvalds”




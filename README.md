# Generating adversarial examples for leNet5 type classifiers

This repository contains code to train a variant of the leNet5 classifier on the MNIST dataset and generate some adversarial examples using PGD. 
It also contains code to train a different CNN on the same dataset (performing the same classification task), in order to demonstrate a phenemenon called the "transferability of adversarial examples", which takes place when adversarial examples crafted to fool one model also succeed in fooling another model with a potentially different architecture.

This contains 
- An implementation of a variant of LeNet and code to train it on the MNIST dataset.
- An implementation of a different CNN and code to train it on the same dataset.
- Some code to generate adversarial examples (based on https://adversarial-ml-tutorial.org/introduction/) for the leNet5 model and test them on the other CNN.
- Sample weights for both models 
- Some test photos of digits

How to use this: 
 - Download the notebook `adversarial_examples_notebook` and open it on Google Colab. 
 - Read through and enjoy.

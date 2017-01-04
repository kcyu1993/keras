"""
ilsvrc1000.py
ILSVRC 1000 Image Net Challenge

This file implements basic ImageNet VGG model training.

Base model:
    VGG 19

Second-layer structure:
    Cov-branch: O2Transform (Non-Parametric and Parametric layers)
    Following the same mode in CIFAR
        Mode 0: Base-line model
        Mode 1: Without top layer, but only the


"""
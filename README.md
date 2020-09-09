# CaptchaRecognition
Captcha Recognition on Pytorch with CRNN + CTC loss

# Introduction
Everyone hates CAPTCHAs. Isn’t it? Those annoying characters in the image you have to type in every time you log in a website. But CAPTCHAs are designed to prevent any program from logging in to the system by verifying you’re a real person. But with the rise of Deep learning and Computer Vision, they can now be defeated. 

# Previous Works on Captcha Recognition:
Earlier, the AI community developers use individual character detection and recognition and then finally append the recognized characters to form the actual text. This approach was in practice till the advent of CTC loss.

![alt text]

![alt text]

# Let’s use CRNN + CTC loss to solve the problem in one-go:
We’ll see how we can recognize the entire text in one go using Convolutional Recurrent Neural Net architecture with Connectionist Temporal Classification loss function.

# Overview of the architecture:
![alt text]

# DATA
Captcha images were downloaded from a repository: https://github.com/AakashKumarNain/CaptchaCracker

# LOSS - CTC:
For better understanding of how CTC sequence modelling works, please refer:
https://distill.pub/2017/ctc/

# What Next?
As next steps, we’ll finetune the model with different combinations of hyperparameters to improve the prediction accuracy. This model is trained to always predict sequence length of 5, whereas in real time, CAPTCHA can be of any length. Next step is to train model with dynamic length of sequences.

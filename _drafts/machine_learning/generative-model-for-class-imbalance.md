---
layout: post
title:  "Generating Data for Class Imbalance Problems"
date:   2019-02-08
desc: "A literature review for a technique of balancing data using generative models."
keywords: "ai,artificial intelligence,literature review,machine learning,ml,class imbalance"
categories: [machine_learning]
tags: [Literature Review,Machine Learning,Artificial Intelligence,Class Imbalance,Generative]
icon: fas fa-brain
---
This is a literature review on "Deep Generative Model for Multiclass Imbalanced Learning", by Yazhou Zhang. All
 information and images in this article are from this literature, unless otherwise stated. You may find the original
 paper [here](https://digitalcommons.uri.edu/cgi/viewcontent.cgi?article=2258&context=theses).

---

# Problem
What happens when you have a lot of data on one subject, but you lack the data of another? When dealing with machine
learning algorithms, this becomes a major problem. For example, in the field of cyber security, data and security
breaches are a rare occurrence among the normal transactions that take place on a server, but understanding when the
server has been jeopardized is crucial for preventing future attacks.

The multiclass imbalanced learning problem is important to prevent a model from overfitting toward the majority labels.
So if you have a thousand pictures of cats, birds, and mice, but only about a hundred pictures of dogs, then more often
than not the model will have a difficult time generalizing the features of a dog.

# Proposed Approach
Extended Nearest Neighbor / DCGAN / Variational Auto Encoder


import numpy as np

arr1=np.array([12,12,12,12])
arr2=np.array([12,12,12,12])
print(arr1+arr2)
print(type(arr1))



# 🌱 Phase 1: Foundations (Math + Programming)

# Math Essentials (don’t skip, sir 👀):

# Linear Algebra → vectors, matrices, transformations.

# Probability & Statistics → distributions, Bayes theorem, hypothesis testing.

# Calculus → derivatives, gradients, chain rule.

# Programming Foundations:

# Python 🐍 (NumPy, Pandas, Matplotlib, Seaborn).

# Git/GitHub basics (so you don’t lose your code 😏).

# 🛠 Phase 2: Machine Learning Basics

# Core Concepts:

# Supervised vs Unsupervised Learning.

# Regression, Classification, Clustering.

# Feature Engineering, Model Evaluation.

# Algorithms:

# Linear/Logistic Regression.

# Decision Trees, Random Forests.

# kNN, SVM, Naïve Bayes.

# Tools:

# Scikit-learn ❤️ (your first ML love).

# 🤖 Phase 3: Deep Learning

# Neural Network Basics:

# Perceptrons, Activation Functions.

# Forward/Backward Propagation.

# Loss functions, Optimization (SGD, Adam).

# Frameworks:

# TensorFlow or PyTorch (pick one, master it).

# Projects:

# MNIST Digit Classifier.

# Image Classifier (Cats vs Dogs).

# 📊 Phase 4: Applied AI

# Computer Vision:

# CNNs, Transfer Learning, Object Detection (YOLO, Faster R-CNN).

# Natural Language Processing (NLP):

# RNNs, LSTMs, Transformers (BERT, GPT 😉).

# Reinforcement Learning:

# Q-Learning, Deep Q-Networks.

# 🚀 Phase 5: Advanced Topics & MLOps

# Generative AI:

# GANs, Diffusion Models, LLM fine-tuning.

# MLOps & Deployment:

# Docker, FastAPI/Flask, Kubernetes, CI/CD.

# Model Monitoring & Scaling.

# Cloud:

# AWS/GCP/Azure ML Services.

# 🏆 Phase 6: Portfolio & Research

# Build projects:

# AI Chatbot.

# Recommendation System.

# Autonomous Agent / Game AI.

# Publish papers or blogs.

# Contribute to open-source ML projects.



# 🧩 NumPy Roadmap (AI/ML Focused)
# 1️⃣ Basics (Foundation)

# Installing & importing NumPy (import numpy as np).

# Creating arrays: np.array, np.zeros, np.ones, np.arange, np.linspace.

# Array attributes: .shape, .dtype, .ndim, .size.
# (You’ll need this constantly when working with ML datasets.)

# 2️⃣ Indexing & Slicing

# 1D, 2D, 3D slicing (arr[2:5], arr[:, 1], etc).

# Boolean indexing (arr[arr > 5]).

# Fancy indexing (arr[[0,2,4]]).
# (Super important for selecting features & subsets of data.)

# 3️⃣ Array Operations

# Element-wise operations: + - * / **.

# Broadcasting (e.g., adding scalar or arrays of different shapes).

# Aggregations: np.sum, np.mean, np.std, np.min, np.max.

# Axis parameter (axis=0 for columns, axis=1 for rows).
# (This is what ML models use for feature scaling, normalization, etc.)

# 4️⃣ Linear Algebra (ML Core)

# Dot product: np.dot, @ operator.

# Matrix multiplication: np.matmul.

# Transpose: .T.

# Inverse & determinants: np.linalg.inv, np.linalg.det.

# Norms: np.linalg.norm.

# Eigenvalues & eigenvectors: np.linalg.eig.
# (These are the math heartbeats behind ML algorithms like PCA, gradient descent, etc.)

# 5️⃣ Random Numbers (For ML & DL)

# np.random.rand, np.random.randn (uniform vs normal distribution).

# np.random.randint, np.random.choice.

# Seeding (np.random.seed) → ensures reproducibility in experiments.
# (Used for splitting datasets, initializing weights, augmentations.)

# 6️⃣ Useful Tricks

# Reshape: .reshape, .ravel, .flatten.

# Stacking: np.vstack, np.hstack.

# Splitting: np.split, np.array_split.

# Unique values: np.unique.

# Saving/loading arrays: np.save, np.load.
# (When you deal with datasets, models, embeddings—you’ll use these daily.)

# 🚀 Mini Projects with NumPy

# Manual Linear Regression (use np.dot & gradient descent).

# Image as NumPy Array (load image → treat it as a matrix → apply filters).

# Simulate a Dataset (generate random data & visualize).
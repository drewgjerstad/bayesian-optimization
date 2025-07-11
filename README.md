# Bayesian Optimization
This repo contains my work with Bayesian optimization and related topics. The
primary goal of this repo is to develop and curate a set of resources that
myself and others can use to better understand (and utilize) Bayesian
optimization.

> [!NOTE]
> When accessing the notebooks linked below, no te that some of the internal
> links––primarily used for navigation––may not work on all platforms. While
> they are functional in VS Code, I have noticed issues with their functionality
> in GitHub Preview and on popular rendering sites such as _nbviewer_. If you
> have any suggestions or know of a workaround, feel free to
> [raise an issue](https://github.com/drewgjerstad/bayesian-optimization/issues/new/choose)
> and let me know.

## 📁 `docs`
This directory contains notebooks consisting of notes and tutorials for Bayesian
optimization and related topics.

* [**Introduction to Bayesian Optimization**](docs/01_introduction.ipynb)
* **Gaussian Processes**
* **Covariance Functions (Kernels)**
* [**Decision Theory**](docs/04_decision_theory.ipynb)
* **Utility Functions**
* **Acquisition Functions**
* **Implementation**
* **Theoretical Analysis**
* **GP Regression**
* **GP Classification**


## 📁 `examples`
This directory contains notebooks exploring various examples of Bayesian
optimization and its applications.
 * **Bayesian Statistics**
 * [**Introduction to GPyTorch and GAUCHE**](examples/gpytorch_and_gauche.ipynb)  
    _This notebook synthesizes information from GPyTorch and GAUCHE's_
    _documentation regarding Gaussian processes for machine learning and how to_
    _apply them to irregular-structured input representations (i.e., molecular,_
    _graph, etc.)._

## 📁 `src`
This directory contains from-scratch implementations of Bayesian optimization
methods and the methods of related topics.


## 📚 References
Below is a list of reference texts, papers, and other sources on Bayesian
optimization and related topics. The BibTeX entries can be found in the
[**`bibliography.bib`**](bibliography.bib) file.

 * _Bayesian Optimization_ by Roman Garnett (2023)
 * _Bayesian Optimization: Theory and Practice Using Python_ by Peng Liu (2023)
 * _Gaussian Processes for Machine Learning_ by Rasmussen & Williams (2019)


<!---
Topics
 - Introduction to Bayesian optimization
    - Garnett Chapter 1, Liu Chapter 1, R+W Chapter 1, Shahriari Paper
 - Gaussian Processes
    - Garnett Chapter 2/3, Liu Chapter 2, R+W Appendix B
 - Covariance Functions
    - R+W Chapter 4
 - Decision Theory
    - Garnett Chapter 5, Liu Chapter 3
 - Utility Functions
    - Garnett Chapter 6
 - Acquisition Functions
    - Garnett Chapter 7/8
 - Implementation
    - Garnett Chapter 9
 - Theoretical Analysis
    - Garnett Chapter 10
 - GP Regression
    - R+W Chapter 2
 - GP Classification
    - R+W Chapter 3
--->
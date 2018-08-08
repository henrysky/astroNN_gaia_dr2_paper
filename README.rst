
Abstract
===========

To be written (づ｡◕‿‿◕｡)づ.

But you can learn more about how astroNN is applied on APOGEE spectra to infer stellar parameters and abundances in this
repository: https://github.com/henrysky/astroNN_spectra_paper_figures

| Basically,
| **Step 1**: Find [APOGEE/LAMOST]-Gaia overlap
| **Step 2**: Convert Gaia parallax with K-band extinction corrected apparent magnitude to luminosity
| **Step 3**: Shows pairs of APOGEE/LAMOST spectra and luminosity to neural network such that NN can learn something
| **Step 4**: We now have a neural network predicts luminosity just by looking at APOGEE/LAMOST spectra
| **Step 5**: Apply it on the whole APOGEE DR14/LAMOST DR5 or whatever, convert luminosity back to distances and we can map MW

| Fancy Stuff:
| - How can neural network so awesome?
| - Take unknown Gaia DR2 zero-point offset via generative adversarial training

Getting Started
=================

This repository is to make sure all figures and results are reproducible by anyone easily for this paper. Python 3.6 or
above and reasonable computational resource is required.

If Github has issue (or too slow) to load the Jupyter Notebooks, you can go
http://nbviewer.jupyter.org/github/henrysky/astroNN_gaia_dr2_paper/tree/master/

To get started, this paper uses `astroNN`_ developed by us and tested with **astroNN 1.1.0 (Not yet released)**.
Extensive documentation at http://astroNN.readthedocs.io and quick start guide at
http://astronn.readthedocs.io/en/latest/quick_start.html

Some notebooks make use of `milkyway_plot`_ to plot on milkyway.

.. _astroNN: https://github.com/henrysky/astroNN
.. _milkyway_plot: https://github.com/henrysky/milkyway_plot

To continuum normalize arbitrary APOGEE spectrum, see:
http://astronn.readthedocs.io/en/latest/tools_apogee.html#pseudo-continuum-normalization-of-apogee-spectra

Jupyter Notebook
------------------

Incomplete list of notebook

-   | `Datasets_Data_Reduction.ipynb`_
    | You should check out this notebook first as it describes how to reproduce the **exactly** same datasets used in the paper
-   | `Parallax_Offset_PolyModels.ipynb`_
    | To show how does offset in parallax space (aka Gaia observations) affect a simple regress model
-   | `Inference.ipynb`_
    | It describes inference, NN performance and NN uncertainty
-   | `Jacobian.ipynb`_
    | It describes jacobian analysis.
-   | `APOGEE_RC_N_Distance.ipynb`_
    | It describes the evaluation of NNs on APOGEE DR14 Red Clumps catalog and BPG distances

.. _Datasets_Data_Reduction.ipynb: Datasets_Data_Reduction.ipynb
.. _Parallax_Offset_PolyModels.ipynb: Parallax_Offset_PolyModels.ipynb
.. _Inference.ipynb: Inference.ipynb
.. _Jacobian.ipynb: Jacobian.ipynb
.. _APOGEE_RC_N_Distance.ipynb: APOGEE_RC_N_Distance.ipynb

Authors
=========
-  | **Henry Leung** - henrysky_
   | Student, Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] mail.utoronto.ca

-  | **Jo Bovy** - jobovy_
   | Professor, Department of Astronomy and Astrophysics, University of Toronto

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

License
---------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE

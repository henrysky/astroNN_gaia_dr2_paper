
Abstract
===========

Gaia measures the five astrometric parameters for stars in the Milky Way, but only four of them (positions and proper
motion, but not parallax) are well measured beyond a few kpc from the Sun. Modern spectroscopic surveys such as APOGEE
cover a large area of the Milky Way disk and we can use the relation between spectra and luminosity to determine distances
to stars beyond Gaia's parallax reach. Here, we design a deep neural network trained on stars in common between Gaia
and APOGEE that determines spectro-photometric distances to APOGEE stars, while including a flexible model to calibrate
parallax zero-point biases in Gaia DR2. We determine the zero-point offset to be -52.3 +/- 2.0uas when modeling
it as a global constant, but also train a multivariate zero-point offset model that depends on G, G_BP - G_RP color,
and T_eff and that can be applied to all 139 million stars in Gaia DR2 within APOGEE's color--magnitude range.
Our spectro-photometric distances are more precise than Gaia at distances ≈2kpc from the Sun.
We release a catalog of spectro-photometric distances for the entire APOGEE DR14 data set which covers Galactocentric radii
2kpc<≈R<≈19kpc; ≈150,000 stars have <10% uncertainty, making this a
powerful sample to study the chemo-dynamical structure of the disk. We use this sample to map the mean [Fe/H] and 15
abundance ratios [X/Fe] from the Galactic center to the edge of the disk. Among many interesting trends, we find that
the bulge and bar region at R<≈5kpc clearly stands out in [Fe/H] and most abundance ratios.

.. contents:: **Table of Contents**
    :depth: 3

Getting Started
=================

This repository is to make sure all figures and results are reproducible by anyone easily for this paper.

If Github has issue (or too slow) to load the Jupyter Notebooks, you can go
http://nbviewer.jupyter.org/github/henrysky/astroNN_gaia_dr2_paper/tree/master/

To get started, this paper uses `astroNN`_ developed by us and tested with **astroNN 1.1.0 (Not yet released)**.
Python 3.6 or above and reasonable computational resource is required.
Extensive documentation at http://astroNN.readthedocs.io and quick start guide at
http://astronn.readthedocs.io/en/latest/quick_start.html

astroNN Apogee DR14 Distance data is available as `apogee_dr14_nn_dist.fits`_

Some notebooks make use of `milkyway_plot`_ to plot on milkyway and `gaia_tools`_ to do query.

Some notebooks make use of data from
**Deep learning of multi-element abundances from high-resolution spectroscopic data** [`arXiv:1804.08622`_][`ADS`_] and its \
data product available at https://github.com/henrysky/astroNN_spectra_paper_figures

.. _arXiv:1804.08622: https://arxiv.org/abs/1808.04428
.. _ADS: https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.2079L/abstract

.. _astroNN: https://github.com/henrysky/astroNN
.. _milkyway_plot: https://github.com/henrysky/milkyway_plot
.. _gaia_tools: https://github.com/jobovy/gaia_tools

To continuum normalize arbitrary APOGEE spectrum, see:
http://astronn.readthedocs.io/en/latest/tools_apogee.html#pseudo-continuum-normalization-of-apogee-spectra

A legacy version of data file available as `apogee_dr14_nn_dist_0562.fits`_ in which 56.2uas offset is applied directly to train,
and its data model will not be provided.

Docker Image
----------------

If you have `Docker`_ installed, you can use the `Dockerfile`_ to build a Docker image upon Tensorflow container from `NVIDIA NGC Catalog`_ with all dependencies installed and data files downloaded.

.. _NVIDIA NGC Catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow
.. _Dockerfile: Dockerfile
.. _Docker: https://www.docker.com/

To build the Docker image called ``astroNN_gaia_dr2_paper``, run the following command in the root directory of this repository:

.. code-block:: bash

    docker build -t astroNN_gaia_dr2_paper .

To run the Docker container with all GPU available to the container named ``testing123``, run the following command:

.. code-block:: bash
    
    docker run --gpus all --name testing123 -it -e SHELL=/bin/bash --entrypoint bash astroNN_gaia_dr2_paper

Then you can attach to the container by running:

.. code-block:: bash

    docker exec -it testing123 bash

Now you can run all notebooks or training script inside the container

Jupyter Notebook
------------------
-   | `Datasets_Data_Reduction.ipynb`_
    | You should check out this notebook first as it describes how to reproduce the **exactly** same datasets used in the paper
    | You can also download the pre-compiled dataset used in this paper on `Zenodo`_ and place the files under the root directory of this repository.
-   | `Training.ipynb`_
    | It provides the code used to train ``astroNN_no_offset_model``, ``astroNN_constlant_model`` and ``astroNN_multivariate_model``
    | It provides a minimal model code (in pure Tensorflow and pure PyTorch) to help you understand what the core logic of the model is
-   | `Offset_Gaia.ipynb`_
    | It describes the result of Gaia offset
-   | `Inference.ipynb`_
    | It describes inference, NN performance and NN uncertainty. And how to generate distances for the whole APOGEE DR14.
-   | `Jacobian.ipynb`_
    | It describes jacobian analysis.
-   | `MW_Science.ipynb`_
    | It describes some MilkyWay Science plots
-   | `nn_figure1_draw_io`_
    | Source for Figure 1 in paper for the NN model, can be opened and edited by draw.io

.. _Zenodo: https://zenodo.org/records/13308879
.. _Datasets_Data_Reduction.ipynb: Datasets_Data_Reduction.ipynb
.. _Training.ipynb: Training.ipynb
.. _Offset_Gaia.ipynb: Offset_Gaia.ipynb
.. _Inference.ipynb: Inference.ipynb
.. _Jacobian.ipynb: Jacobian.ipynb
.. _MW_Science.ipynb: MW_Science.ipynb
.. _nn_figure1_draw_io: https://github.com/henrysky/astroNN_gaia_dr2_paper/raw/master/nn_figure1_draw_io

Neural Net Models and Quantity Conversion
-----------------------------------------------

It is recommended to use model ends with ``_reduced`` for example, using ``astroNN_constant_model_reduced`` instead of ``astroNN_constant_model``

- ``astroNN_no_offset_model`` is a astroNN's `ApogeeBCNN()`_ class model to infer Ks-Band `fakemag`_ without offset model.

- ``astroNN_constant_model`` is a astroNN's `ApogeeDR14GaiaDR2BCNN()`_ class model to infer Ks-Band `fakemag`_ with a constant offset model

- ``astroNN_constant_model_reduced`` is a astroNN's `ApogeeBCNN()`_ class model extracted from ``astroNN_constant_model``

- ``astroNN_multivariate_model`` is a astroNN's `ApogeeDR14GaiaDR2BCNN()`_ class model to infer Ks-Band `fakemag`_ with a multivariate offset model

- ``astroNN_multivariate_model_reduced`` is a astroNN's `ApogeeBCNN()`_ class model extracted from ``astroNN_multivariate_model``

.. _ApogeeBCNN(): http://astronn.readthedocs.io/en/latest/neuralnets/apogee_bcnn.html
.. _ApogeeDR14GaiaDR2BCNN(): https://astronn.readthedocs.io/en/latest/neuralnets/apogeedr14_gaiadr2_bcnn.html
.. _fakemag: https://astronn.readthedocs.io/en/latest/tools_gaia.html#fakemag-dummy-scale

To load the model, open python outside ``your_astroNN_model``

.. code-block:: python

    from astroNN.models import load_folder

    # replace the name of the NN folder you want to open
    neuralnet = load_folder('astroNN_model')
    # neuralnet is an astroNN neural network object, to learn more;
    # http://astronn.readthedocs.io/en/latest/neuralnets/basic_usage.html

    # To get what the output neurones are representing
    print(neuralnet.targetname)

To convert NN Ks-band fakemag (a pseudo luminosity scale) and its uncertainty to astrometric quantities, you can

.. code-block:: python

    from astroNN.gaia import fakemag_to_pc, fakemag_to_parallax

    # outputs carry astropy unit
    parsec, parsec_uncertainty = fakemag_to_pc(nn_fakemag, ks_magnitude, nn_fakemag_uncertainty)
    # outputs carry astropy unit
    parallax, parallax_uncertainty = fakemag_to_parallax(nn_fakemag, ks_magnitude, nn_fakemag_uncertainty)

    # OR you can provide input without uncertainty
    # output carries astropy unit
    parsec = fakemag_to_pc(fakemag, ks_magnitude)
    # output carries astropy unit
    parallax = fakemag_to_parallax(fakemag, ks_magnitude)

To convert NN Ks-band fakemag (a pseudo luminosity scale) to log10 solar luminosity, you can

.. code-block:: python

    from astroNN.gaia import fakemag_to_logsol

    logsol = fakemag_to_logsol(fakemag, band='Ks')

astroNN Apogee DR14 Distance & Data Model
-------------------------------------------

`apogee_dr14_nn_dist.fits`_ is compiled prediction with ``astroNN_constant_model_reduced`` on the whole Apogee DR14.
The code used to generate this file is described in `Inference.ipynb`_

.. _apogee_dr14_nn_dist.fits: https://github.com/henrysky/astroNN_gaia_dr2_paper/raw/master/apogee_dr14_nn_dist.fits
.. _apogee_dr14_nn_dist_0562.fits: https://github.com/henrysky/astroNN_gaia_dr2_paper/raw/master/apogee_dr14_nn_dist_0562.fits

To load it with python and to initialize orbit with `galpy`_ (requires galpy>=1.4 and astropy>3)

.. _galpy: https://github.com/jobovy/galpy

.. code-block:: python

    from astropy.io import fits

    # read the data file
    f = fits.getdata("apogee_dr14_nn_dist.fits")

    # ========= see our paper for the most accurate descriptive data model ========= #

    # APOGEE and NN data, contains -9999. for unknown/bad data
    apogee_id = f['apogee_id']  # APOGEE's apogee id
    location_id = f['location_id']  # APOGEE DR14 location id
    ra_apogee = f['ra_apogee']  # J2000 RA
    dec_apogee = f['dec_apogee']  # J2000 DEC
    fakemag = f['fakemag']  # NN Ks-band pseudo luminosity prediction
    fakemag_error = f['fakemag_error']  # NN Ks-band pseudo luminosity uncertainty
    nn_parsec = f['dist']  # NN inverse parallax in parsec
    nn_parsec_uncertainty = f['dist_error']  # NN inverse parallax total uncertainty in parsec
    nn_parsec_model_uncertainty = f['dist_model_error']  # NN inverse parallax model uncertainty in parsec
    nn_plx = f['nn_parallax']  # NN parallax in mas
    nn_plx_uncertainty = f['nn_parallax_error']  # NN parallax uncertainty in mas
    nn_plx_model_uncertainty = f['nn_parallax_model_error']  # NN parallax model uncertainty in mas
    weighted_dist = f['weighted_dist']  # inv var weighted NN & Gaia distance in parsec
    weighted_dist_uncertainty = f['weighted_dist_error']  # inv var weighted NN & Gaia distance uncertainty in parsec

    # Gaia DR2 Data, contains -9999. for unknown/bad data
    ra = f['ra']  # RA J2015.5
    dec = f['dec']  # DEC J2015.5
    pmra = f['pmra']  # RA proper motion
    pmra_error = f['pmra_error']  # RA proper motion error
    pmdec = f['pmdec']  # DEC proper motion
    pmdec_error = f['pmdec_error']  # DEC proper motion error
    pmdec = f['pmdec']  # DEC proper motion
    phot_g_mean_mag = f['phot_g_mean_mag']  # g-band magnitude
    bp_rp = f['bp_rp']  # bp_rp colour


Moreover, you can use galpy (>=1.5) to setup ``Orbit`` to easily do unit conversion or integrating orbits

.. code-block:: python

    # To convert to 3D position and 3D velocity
    from astroNN.apogee import allstar
    from galpy.orbit import Orbit
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import CartesianDifferential

    f_allstardr14 = fits.getdata(allstar(dr=14))

    # because the catalog contains -9999.
    non_n9999_idx = ((pmra !=-9999.) & (pmdec !=-9999.) & (nn_parsec !=-9999.))
    c = coord.SkyCoord(ra=ra[non_n9999_idx]*u.degree,
                       dec=dec[non_n9999_idx]*u.degree,
                       distance=nn_parsec[non_n9999_idx]*u.pc,
                       pm_ra_cosdec=pmra[non_n9999_idx]*u.mas/u.yr,
                       pm_dec=pmdec[non_n9999_idx]*u.mas/u.yr,
                       radial_velocity=f_allstardr14['VHELIO_AVG'][non_n9999_idx]*u.km/u.s,
                       galcen_distance=8.125*u.kpc, # https://arxiv.org/abs/1807.09409 (GRAVITY Collaboration 2018)
                       z_sun=20.8*u.pc, # https://arxiv.org/abs/1809.03507 (Bennett & Bovy 2018)
                       galcen_v_sun=CartesianDifferential([11.1, 245.7, 7.25]*u.km/u.s))

    # galpy Orbit object, need galpy >= 1.5
    os = Orbit(c)
    x, y, z = os.x(), os.y(), os.z()    # 3D position
    vx, vy, vz = os.vx(), os.vy(), os.vz()    # 3D velocity

Using Neural Net on arbitrary APOGEE spectra
-----------------------------------------------

To do inference on an arbitrary APOGEE spectrum to get distance,

1. Open python under the repository folder but outside the neural net folder
2. Copy and paste the following code to do inference with neural net in this paper on ``2M19060637+4717296``

.. code-block:: python

    from astropy.io import fits
    from astroNN.apogee import visit_spectra, apogee_continuum
    from astroNN.gaia import extinction_correction, fakemag_to_pc
    from astroNN.models import load_folder

    # arbitrary spectrum
    f = fits.open(visit_spectra(dr=14, apogee='2M19060637+4717296'))
    spectrum = f[1].data
    spectrum_err = f[2].data
    spectrum_bitmask = f[3].data

    # using default continuum and bitmask values to continuum normalize
    norm_spec, norm_spec_err = apogee_continuum(spectrum, spectrum_err,
                                                bitmask=spectrum_bitmask, dr=14)

    # load neural net, it is recommend to use model ends with _reduced
    # for example, using astroNN_constant_model_reduced instead of astroNN_constant_model
    neuralnet = load_folder('astroNN_constant_model_reduced')

    # inference, if there are multiple visits, then you should use the globally
    # weighted combined spectra (i.e. the second row)
    pred, pred_err = neuralnet.test(norm_spec)

    # correct for extinction
    K = extinction_correction(f[0].header['K'], f[0].header['AKTARG'])

    # convert prediction in fakemag to distance
    pc, pc_error = fakemag_to_pc(pred[:, 0], K, pred_err['total'][:, 0])
    print(f"Distance: {pc} +/- {pc_error}")

Authors
=========
-  | **Henry Leung** - henrysky_
   | Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] utoronto.ca

-  | **Jo Bovy** - jobovy_
   | Department of Astronomy and Astrophysics, University of Toronto

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

License
---------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE

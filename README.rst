
Abstract
===========

To be written (づ｡◕‿‿◕｡)づ.

But you can learn more about how astroNN is applied on APOGEE spectra to infer stellar parameters and abundances in this
repository: https://github.com/henrysky/astroNN_spectra_paper_figures

Getting Started
=================

This repository is to make sure all figures and results are reproducible by anyone easily for this paper.

If Github has issue (or too slow) to load the Jupyter Notebooks, you can go
http://nbviewer.jupyter.org/github/henrysky/astroNN_gaia_dr2_paper/tree/master/

To get started, this paper uses `astroNN`_ developed by us and tested with **astroNN 1.1.0 (Not yet released)**.
Python 3.6 or above and reasonable computational resource is required.
Extensive documentation at http://astroNN.readthedocs.io and quick start guide at
http://astronn.readthedocs.io/en/latest/quick_start.html

astroNN Apogee DR14 Distance data is available as `apogee_dr14_nn_dist.fits`_.

Some notebooks make use of `milkyway_plot`_ to plot on milkyway. Some notebooks make use of data from
**Deep learning of multi-element abundances from high-resolution spectroscopic data** [`arXiv:1804.08622`_][`ADS`_] and its\
data product avaliable at https://github.com/henrysky/astroNN_spectra_paper_figures

.. _arXiv:1804.08622: https://arxiv.org/abs/1808.04428
.. _ADS: https://ui.adsabs.harvard.edu/#abs/2018arXiv180804428L/

.. _astroNN: https://github.com/henrysky/astroNN
.. _milkyway_plot: https://github.com/henrysky/milkyway_plot

To continuum normalize arbitrary APOGEE spectrum, see:
http://astronn.readthedocs.io/en/latest/tools_apogee.html#pseudo-continuum-normalization-of-apogee-spectra

Jupyter Notebook
------------------

Incomplete list of notebook

-   | `Datasets_Data_Reduction.ipynb`_
    | You should check out this notebook first as it describes how to reproduce the **exactly** same datasets used in the paper
-   | `Trainnig.ipynb`_
    | It provided the code used to train ``astroNN_no_offset_model``, ``astroNN_constant_model`` and ``astroNN_multivariate_model``
-   | `Offset_Gaia.ipynb`_
    | It describes the result of Gaia offset
-   | `Inference.ipynb`_
    | It describes inference, NN performance and NN uncertainty
-   | `Jacobian.ipynb`_
    | It describes jacobian analysis.
-   | `APOGEE_RC_N_Distance.ipynb`_
    | It describes the evaluation of NNs on APOGEE DR14 Red Clumps catalog and BPG distances
-   | `APOGEE_DR14_NN_Catalog.ipynb`_
    | It describes how to generate distances for the whole APOGEE DR14

.. _Datasets_Data_Reduction.ipynb: Datasets_Data_Reduction.ipynb
.. _Offset_Gaia.ipynb: Offset_Gaia.ipynb
.. _Trainnig.ipynb: Trainnig.ipynb
.. _Inference.ipynb: Inference.ipynb
.. _Jacobian.ipynb: Jacobian.ipynb
.. _APOGEE_RC_N_Distance.ipynb: APOGEE_RC_N_Distance.ipynb
.. _APOGEE_DR14_NN_Catalog.ipynb: APOGEE_DR14_NN_Catalog.ipynb

Neural Net Models and Quantity Conversion
-----------------------------------------------

It is recommend to use model ends with ``_reduced`` for example, using ``astroNN_constant_model_reduced`` instead of ``astroNN_constant_model``

- ``astroNN_apogee14_gaia2_no_offset`` is a astroNN's `ApogeeBCNN()`_ class model to infer Ks-Band `fakemag`_ without offset model.

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

To convert NN Ks-band fakemag and fakemag uncertainty to astrometric quantities, you can

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

To convert NN Ks-band fakemag to log10 solar luminosity, you can

.. code-block:: python

    from astroNN.gaia import fakemag_to_logsol

    logsol = fakemag_to_logsol(fakemag, band='Ks')

astroNN Apogee DR14 Distance & Data Model
-------------------------------------------

`apogee_dr14_nn_dist.fits`_ is compiled prediction with ``astroNN_constant_model_reduced`` on the whole Apogee DR14.
The code used to generate this file is described in `Inference.ipynb`_

.. _apogee_dr14_nn_dist.fits: apogee_dr14_nn_dist.fits

To load it with python and to initialize orbit with `galpy`_ (requires galpy>=1.4 and astropy>3)

.. _galpy: https://github.com/jobovy/galpy

.. code-block:: python

    from astropy.io import fits

    # read the data file
    f = fits.getdata("apogee_dr14_nn_dist.fits")

    # APOGEE and NN data, contains -9999. for unknown/bad data
    apogee_id = f['APOGEE_ID']  # APOGEE's apogee id
    location_id = f['LOCATION_ID']  # APOGEE DR14 location id
    ra = f['RA']  # J2000 RA
    dec = f['DEC']  # J2000 DEC
    fakemag = f['fakemag']  # NN Ks-band fakemag prediction
    fakemag_error = f['fakemag_error']  # NN Ks-band fakemag uncertainty
    nn_parsec = f['pc']  # NN inverse parallax in parsec
    nn_parsec_uncertainty = f['pc_error']  # NN inverse parallax total uncertainty in parsec
    nn_parsec_model_uncertainty = f['pc_model_error']  # NN inverse parallax model uncertainty in parsec
    nn_plx = f['nn_parallax']  # NN parallax in mas
    nn_plx_uncertainty = f['nn_parallax_error']  # NN parallax uncertainty in mas
    nn_plx_model_uncertainty = f['nn_parallax_model_error']  # NN parallax model uncertainty in mas
    weighted_plx = f['weighted_parallax']  # inv var weighted NN & Gaia parallax in mas
    weighted_plx_uncertainty = f['weighted_parallax_error']  # inv var weighted NN & Gaia parallax uncertainty in mas

    # Gaia DR2 Data, contains -9999. for unknown/bad data
    ra_j2015_5 = f['RA_J2015.5']  # RA J2015.5
    dec_j2015_5 = f['DEC_J2015.5']  # DEC J2015.5
    pmra = f['pmra']  # RA proper motion
    pmra_error = f['pmra_error']  # RA proper motion error
    pmdec = f['pmdec']  # DEC proper motion
    pmdec_error = f['pmdec_error']  # DEC proper motion error
    pmdec = f['pmdec']  # DEC proper motion
    phot_g_mean_mag = f['phot_g_mean_mag']  # g-band magnitude
    bp_rp = f['bp_rp']  # bp_rp colour


In addition, you can use galpy to convert to useful quantity with the following code

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
    c = coord.SkyCoord(ra=ra_j2015_5[non_n9999_idx]*u.degree,
                       dec=dec_j2015_5[non_n9999_idx]*u.degree,
                       distance=nn_parsec[non_n9999_idx]*u.pc,
                       pm_ra_cosdec=pmra[non_n9999_idx]*u.mas/u.yr,
                       pm_dec=pmdec[non_n9999_idx]*u.mas/u.yr,
                       radial_velocity=f_allstardr14['VHELIO_AVG'][non_n9999_idx]*u.km/u.s,
                       galcen_distance=8.125*u.kpc, # https://arxiv.org/abs/1807.09409 (GRAVITY Collaboration 2018)
                       z_sun=20.8*u.pc, # https://arxiv.org/abs/1809.03507 (Bennett & Bovy 2018)
                       galcen_v_sun=CartesianDifferential([11.1, 245.7, 7.25]*u.km/u.s))

    # galpy Orbit object
    o = Orbit(c)
    x, y, z = o.x(), o.y(), o.z()    # 3D position
    vx, vy, vz = o.vx(), o.vy(), o.vz()    # 3D velocity

Or you can use an experimental feature of galpy to setup `Orbits` class which allow you to integrate orbit in parallel

.. code-block:: python

    # To convert to 3D position and 3D velocity
    from astroNN.apogee import allstar
    from galpy.orbit import Orbits
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import CartesianDifferential

    f_allstardr14 = fits.getdata(allstar(dr=14))

    # because the catalog contains -9999.
    non_n9999_idx = ((pmra !=-9999.) & (pmdec !=-9999.) & (nn_parsec !=-9999.))
    c = coord.SkyCoord(ra=ra_j2015_5[non_n9999_idx]*u.degree,
                       dec=dec_j2015_5[non_n9999_idx]*u.degree,
                       distance=nn_parsec[non_n9999_idx]*u.pc,
                       pm_ra_cosdec=pmra[non_n9999_idx]*u.mas/u.yr,
                       pm_dec=pmdec[non_n9999_idx]*u.mas/u.yr,
                       radial_velocity=f_allstardr14['VHELIO_AVG'][non_n9999_idx]*u.km/u.s,
                       galcen_distance=8.125*u.kpc, # https://arxiv.org/abs/1807.09409 (GRAVITY Collaboration 2018)
                       z_sun=20.8*u.pc, # https://arxiv.org/abs/1809.03507 (Bennett & Bovy 2018)
                       galcen_v_sun=CartesianDifferential([11.1, 245.7, 7.25]*u.km/u.s))

    # galpy Orbits object
    os = Orbits(c)
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

    # arbitary spectrum
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

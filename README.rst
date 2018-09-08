
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
-   | `GaiaAdversarial.ipynb`_
    | To show the adversarial training process
-   | `Inference.ipynb`_
    | It describes inference, NN performance and NN uncertainty
-   | `Jacobian.ipynb`_
    | It describes jacobian analysis.
-   | `APOGEE_RC_N_Distance.ipynb`_
    | It describes the evaluation of NNs on APOGEE DR14 Red Clumps catalog and BPG distances

.. _Datasets_Data_Reduction.ipynb: Datasets_Data_Reduction.ipynb
.. _Parallax_Offset_PolyModels.ipynb: Parallax_Offset_PolyModels.ipynb
.. _GaiaAdversarial.ipynb: GaiaAdversarial.ipynb
.. _Inference.ipynb: Inference.ipynb
.. _Jacobian.ipynb: Jacobian.ipynb
.. _APOGEE_RC_N_Distance.ipynb: APOGEE_RC_N_Distance.ipynb

Neural Net Models and Quantity Conversion
-----------------------------------------------
- ``astroNN_Ks_fakemag`` is a trained astroNN's `ApogeeBCNN()`_ class model to infer Ks-Band `fakemag`_.

- ``astroNN_Ks_fakemag_adversial`` is an adversarially trained astroNN's `ApogeeBCNN()`_ class model to infer Ks-Band `fakemag`_.

.. _ApogeeBCNN(): http://astronn.readthedocs.io/en/latest/neuralnets/apogee_bcnn.html
.. _fakemag: https://astronn.readthedocs.io/en/latest/tools_gaia.html#fakemag-dummy-scale

To load the model, open python outside ``astroNN_Ks_fakemag`` or ``astroNN_Ks_fakemag_adversial``

.. code-block:: python

    from astroNN.models import load_folder

    # replace the name of the NN folder you want to open
    neuralnet = load_folder('astroNN_Ks_fakemag_adversial')
    # neuralnet is an astroNN neural network object, to learn more;
    # http://astronn.readthedocs.io/en/latest/neuralnets/basic_usage.html

    # To get what the output neurones are representing
    print(neuralnet.targetname)

To converse NN Ks-band fakemag and fakemag uncertainty to astrometric quantities, you can

.. code-block:: python

    from astroNN.gaia import fakemag_to_pc, fakemag_to_parallax

    # outputs carry astropy unit
    parsec, parsec_uncertainty = fakemag_to_pc(nn_fakemag, ks_magnitude, nn_fakemag_uncertainty)
    # outputs carry astropy unit
    parallax, parallax_uncertainty = fakemag_to_parallax(nn_fakemag, ks_magnitude, nn_fakemag_uncertainty)

    # OR you can provide input without uncertainty
    # output carries astropy unit
    parsec = fakemag_to_pc(nn_fakemag, ks_magnitude)
    # output carries astropy unit
    parallax = fakemag_to_parallax(nn_fakemag, ks_magnitude)

To converse NN Ks-band fakemag and fakemag uncertainty to log solar luminosity, you can

.. code-block:: python

    from astroNN.gaia import fakemag_to_logsol

    logsol = fakemag_to_logsol(nn_fakemag, band='Ks')

astroNN Apogee DR14 Distance and initialize as galpy Orbits
-------------------------------------------------------------
``apogee_dr14_nn_dist_0562.fits`` is compiled prediction with ``astroNN_Ks_fakemag_adversial`` on the whole Apogee DR14.

To load it with python and to initialize orbit with `galpy`_

.. _galpy: https://github.com/jobovy/galpy

.. code-block:: python

    from astropy.io import fits

    f = fits.getdata("apogee_dr14_nn_dist_0562.fits")
    apogee_id = f['APOGEE_ID']  # APOGEE's apogee id
    location_id = f['LOCATION_ID']  # APOGEE DR14 location id
    ra = f['RA']  # J2000 RA
    dec = f['DEC']  # J2000 DEC
    fakemag = f['fakemag']  # neural network Ks-band fakemag prediction
    fakemag_error = f['fakemag_error']  # neural network Ks-band fakemag uncertainty
    nn_prediction = f['pc']  # distance in parsec
    nn_uncertainty = f['pc_error']  # distance uncertainty in parsec
    
    # Gaia DR2 Data
    ra_j2015_5 = f['RA_J2015.5']  # RA J2015.5
    dec_j2015_5 = f['DEC_J2015.5']  # DEC J2015.5
    pmra = f['pmra']  # RA proper motion
    pmra_error = f['pmra_error']  # RA proper motion error
    pmdec = f['pmdec']  # DEC proper motion
    pmdec_error = f['pmdec_error']  # DEC proper motion error
    pmdec = f['pmdec']  # DEC proper motion
    phot_g_mean_mag = f['phot_g_mean_mag']  # g-band magnitude
    bp_rp = f['bp_rp']  # bp_rp colour

    # To convert to 3D position and 3D velocity
    import astropy.units as u
    import astropy.coordinates as coord
    from astroNN.apogee import allstar
    from galpy.orbit import Orbit
    f_allstardr14 = fits.getdata(allstar(dr=14))

    # because the catalog contains -9999.
    non_n9999_idx = [(pmra !=-9999.) & (pmdec !=-9999.) & (nn_prediction !=-9999.)]
    c = coord.SkyCoord(ra=ra_j2015_5[non_n9999_idx]*u.degree,
                       dec=dec_j2015_5[non_n9999_idx]*u.degree,
                       distance=nn_prediction[non_n9999_idx]*u.pc,
                       pm_ra_cosdec=pmra[non_n9999_idx]*u.mas/u.yr,
                       pm_dec=pmdec[non_n9999_idx]*u.mas/u.yr,
                       radial_velocity=f_allstardr14['VHELIO_AVG'][non_n9999_idx]*u.km/u.s)

    # galpy Orbit object
    o = Orbit(c)
    x, y, z = o.x(), o.y(), o.z()    # 3D position
    vx, vy, vz = o.vx(), o.vy(), o.vz()    # 3D velocity

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

    #using default continuum and bitmask values to continuum normalize
    norm_spec, norm_spec_err = apogee_continuum(spectrum, spectrum_err,
                                                bitmask=spectrum_bitmask, dr=14)

    #load neural net
    neuralnet = load_folder('astroNN_Ks_fakemag_adversial')

    # inference, if there are multiple visits, then you should use the globally
    # weighted combined spectra (i.e. the second row)
    pred, pred_err = neuralnet.test(norm_spec)

    # correct for extinction
    K = extinction_correction(f[0].header['K'], f[0].header['AKTARG'])

    # convert prediction in distance
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

# Lifted

Lifted is a multithreaded cpp library for performing lifted wavelet transforms.

It makes use of google-highway to create vectorized code for many different platforms.

There is also a cython based wrapper for python that executes wavelet transforms up to 10x faster in single-threaded mode than pywavelets.

Currently it supports only predefined wavelets.

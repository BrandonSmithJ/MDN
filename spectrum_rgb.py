"""
Based on 
  http://www.physics.sfasu.edu/astro/color/spectra.html
  RGB VALUES FOR VISIBLE WAVELENGTHS   by Dan Bruton (astro@tamu.edu)
"""

import numpy as np
select=np.select
power=np.power
transpose=np.transpose
arange=np.arange

def factor(wl):
    return select(
        [ wl > 700.,
          wl < 420.,
          True ],
        [ .3+.7*(780.-wl)/(780.-700.),
          .3+.7*(wl-380.)/(420.-380.),
          1.0 ] )

def raw_r(wl):
    return select(
        [ wl >= 580.,
          wl >= 510.,
          wl >= 440.,
          wl >= 380.,
          True ],
        [ 1.0,
          (wl-510.)/(580.-510.),
          0.0,
          (wl-440.)/(380.-440.),
          0.0 ] )

def raw_g(wl):
    return select(
        [ wl >= 645.,
          wl >= 580.,
          wl >= 490.,
          wl >= 440.,
          True ],
        [ 0.0,
          (wl-645.)/(580.-645.),
          1.0,
          (wl-440.)/(490.-440.),
          0.0 ] )

def raw_b(wl):
    return select(
        [ wl >= 510.,
          wl >= 490.,
          wl >= 380.,
          True ],
        [ 0.0,
          (wl-510.)/(490.-510.),
          1.0,
          0.0 ] )

gamma = 0.80
def correct_r(wl):
    return power(factor(wl)*raw_r(wl),gamma)
def correct_g(wl):
    return power(factor(wl)*raw_g(wl),gamma)
def correct_b(wl):
    return power(factor(wl)*raw_b(wl),gamma)

ww=arange(380.,781.)
def get_spectrum_cmap():
    ''' use like: 
      cmap = get_spectrum_cmap()
      plot(..., color=[cmap.to_rgba(nm) for nm in wavelengths])
    '''
    from matplotlib import colors, cm
    cmap = transpose([correct_r(ww),correct_g(ww),correct_b(ww)])
    return cm.ScalarMappable(norm=colors.Normalize(vmin=380, vmax=780),
                         cmap=colors.ListedColormap(cmap))
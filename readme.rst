A little bridge that allows you to perform 1d curve fits using 
scipy minimize. 

Basically I needed this kind of code because there are no parameter bounds 
in ``scipy.curve_fit`` (see `this SO question <http://stackoverflow.com/q/16760788/7918>`__), 
so here is a small utility that can: 

1. Enables you to do curve fiffing using very good ``scipy.optimize.minimize`` 
   function (and for example use parameter bounds, or use different method 
   than Lavenberg-Marquard). 
2. Allows you to transparently "disable" parameters --- that is fix it at 
   initial value. 
    
For examples see `example notebook <http://nbviewer.ipython.org/github/jbzdak/scipy-curve-minimizer/blob/master/examples/fit_with_bounds.ipynb>`__. 






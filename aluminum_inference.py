import matplotlib
matplotlib.use('Agg')
import numpy as np
import schwimmbad
import shutil
import glob
import os
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal, norm
from scipy.integrate import quad, solve_ivp
import pdb
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import pickle
import time

alHalfLife = 717.0e3
feHalfLife  = 2.6e6


distPc = [0, 0.625,  1.875,  3.125,  4.375,  5.625,  6.875,  8.125,  9.375, 10.625, 11.875, 13.125, 14.375, 15.625, 16.875, 18.125, 19.375, 20.625, 21.875, 23.125, 24.375, 25.625, 26.875, 28.125, 29.375, 30.625, 31.875, 33.125, 34.375, 35.625, 36.875, 38.125, 39.375, 40.625, 41.875, 43.125, 44.375, 45.625, 46.875, 48.125, 49.375]
distCdf = [0, 2.41287787e-04, 2.13712040e-03, 7.85908793e-03, 1.81310537e-02, 3.72961980e-02, 7.01113371e-02, 1.44634794e-01, 2.03577953e-01, 2.52180208e-01, 3.00644583e-01, 3.49040019e-01, 4.02054393e-01, 4.69890731e-01, 5.23146393e-01, 5.70990314e-01, 6.15628555e-01, 6.52338768e-01, 6.88842163e-01, 7.22829272e-01, 7.51301231e-01, 7.78704629e-01, 8.04453483e-01, 8.27410293e-01, 8.48678088e-01, 8.68256868e-01, 8.85526180e-01, 9.00417083e-01, 9.15101169e-01, 9.28337527e-01, 9.40436386e-01, 9.50984109e-01, 9.59187894e-01, 9.67184861e-01, 9.74526904e-01, 9.80352280e-01, 9.85695081e-01, 9.89865913e-01, 9.93967805e-01, 9.97070077e-01, 1.00000000e+00]

fDist = interp1d( distCdf, distPc )


class star:
    def __init__(self, fn):
        # First let's parse the filename.
        Mind = fn.find('M')
        Zind = fn.find('Z')
        self.mass = float(fn[Mind+1:Zind].replace('p','.'))
        Vind = fn.find('V')
        self.Z = float(fn[Zind+1:Vind])/100.0 # uh, not sure this works in general...
        self.V = float(fn[Vind+1]) # not sure what this is - in retrospect this is velocity as a fraction of breakup - it seems there are two options: 0 and 0.4, (where 0.4 is represented here as just 4)
        self.totalOverLifetime = None

        # excellent! Now let's read in the data array.
        # These are from this website: https://obswww.unige.ch/Research/evol/tables_grids2011/
        # Which is accesed by clicking "tracks" under "Grids of Stellar Models with Rotation" here:
        #   https://www.unige.ch/sciences/astro/evolution/en/database/
        # which was linked (though the link was broken) https://ui.adsabs.harvard.edu/abs/2012A%26A...537A.146E/abstract
        # and cited by Dwarkadas+ as his source for 26Al production
        # Cited in this paper
        # These columns are:
        #  (our index 0) 2 - age (yr)
        #  (our index 1) 3 - Mass (msun)
        #  (our index 2) 4 - log10 Luminosity (Lsun)
        #  (our index 3) 5 - log10 Teff (K)
        #  (our index 4) 16 - 26Al mass fraction @ the surface
        #  (our index 5) 19 - log10 Mdot (Msun/yr)
        #  (our index 6) 36 - correction factor for Mdot due to rotation (not sure how this is defined)
        self.arr = np.genfromtxt(fn, skip_header=3, usecols=(2-1, 3-1, 4-1, 5-1, 16-1, 19-1) )

    def aluminum26_vs_time(self):
        return self.arr[:,4] * np.power(10.0, self.arr[:,5])
    def time(self):
        return self.arr[:,0]
    def aluminum26_at_age(self, age, scale):
        # age is the actual age
        # scale is the ratio of lifetimes. Used to just pass age*scale into this function, but that's pretty sketch because alHalfLife should not be scaled!
        dts = (self.time()[1:] - self.time()[:-1])*scale
        avg26Al = (self.aluminum26_vs_time()[1:] + self.aluminum26_vs_time()[:-1])/2.0
        avgTime = ((self.time()[1:] + self.time()[:-1])/2.0) *scale
        include = avgTime < age
        integrand = avg26Al[include] * dts[include] * np.power(2.0, - (age - avgTime[include]) / alHalfLife ) 
        return np.sum(integrand)
    def lifetime(self):
        return np.max(self.time())
    def total_aluminum_over_lifetime(self, scale):
        if self.totalOverLifetime is None:
            dts = (self.time()[1:] - self.time()[:-1])*scale
            avg26Al = (self.aluminum26_vs_time()[1:] + self.aluminum26_vs_time()[:-1])/2.0
            avgTime = ((self.time()[1:] + self.time()[:-1])/2.0)*scale
            #include = avgTime < 2*self.lifetime() # should be all true
            include = np.ones( len(dts), dtype=bool )
            integrand = avg26Al[include] * dts[include] 
            self.totalOverLifetime = np.sum(integrand)
        return self.totalOverLifetime

    def lifetime_fraction(self):
        return self.time()/np.max(self.time())
    def logProdRate(self, grid):
        # Given a grid of values of 1-Age/Lifetime, return 26Al production rate per log time.
        f = interp1d( np.log10(1.0 - self.lifetime_fraction()), np.log10(self.aluminum26_vs_time() * (1.0-self.lifetime_fraction())) , bounds_error=False)
        return f(np.log10(grid))


class setOfModels:
    def __init__(self, list_of_stars):
        #self.list_of_stars = list_of_stars
        self.masses = np.array([star.mass for star in list_of_stars])
        to_sort = np.argsort(self.masses)
        array_of_stars = np.array(list_of_stars)
        self.list_of_stars = array_of_stars[to_sort]
        self.masses = self.masses[to_sort]
        self.est_lifetime = interp1d( np.log10(self.masses), np.log10(np.array([star.lifetime() for star in self.list_of_stars])), kind='quadratic', bounds_error=False )

        Z96Masses = [9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0]
        Z96Yields = [5.51e-06, 5.62e-06, 6.32e-06, 6.89e-06, 6.62e-06, 7.39e-06, 8.92e-06, 9.94e-06, 1.12e-05, 1.1e-05, 7.93e-06, 9.74e-06, 1.22e-05 ]
        Z96YieldsFe = [2.63e-05, 2.28e-05, 1.83e-05, 2.47e-05, 1.75e-05, 1.93e-05, 2.49e-05, 2.2e-05, 3.07e-05, 1.29e-06, 2.94e-07, 1.06e-05, 2.33e-05]
        
        N20Masses = [ 12.25, 12.5, 12.75, 13., 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14. , 14.1, 14.2, 14.3, 14.4, 14.5, 14.6,14.7, 14.8, 14.9, 15.2, 15.7, 15.8, 15.9, 16., 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17., 17.1, 17.3, 17.4, 17.5, 17.6, 17.7, 17.9, 18., 18.1, 18.2, 18.3, 18.4, 18.5, 18.7, 18.8, 18.9, 19., 19.1,   19.2,   19.3,   19.4,   19.7, 19.8,   20.1,   20.2,   20.3,   20.4,   20.5,   20.6,   20.8,   21. ,   21.1, 21.2,   21.5,   21.6,   21.7,   25.2,   25.3,   25.4,   25.5,   25.6,   25.7, 25.8,   25.9,   26. ,   26.1,   26.2,   26.3,   26.4,   26.5,   26.6,   26.7, 26.8,   26.9,   27. ,   27.1,   27.2,   27.3,   27.4,   29. ,   29.1,   29.2, 29.6,   60. ,   80. ,  100. ,  120.  ]
        N20Yields = [1.48e-05, 1.53e-05, 1.59e-05, 1.72e-05, 2.37e-05, 2.44e-05, 2.48e-05, 2.53e-05,2.56e-05, 2.60e-05, 2.43e-05, 3.00e-05, 2.49e-05, 3.12e-05, 3.17e-05, 3.57e-05,3.21e-05, 3.16e-05, 2.54e-05, 3.09e-05, 2.38e-05, 3.53e-05, 3.14e-05, 2.36e-05,2.49e-05, 2.48e-05, 2.38e-05, 2.54e-05, 2.50e-05, 2.62e-05, 2.64e-05, 2.64e-05,2.68e-05, 2.68e-05, 2.71e-05, 2.76e-05, 2.77e-05, 2.81e-05, 2.87e-05, 3.58e-05,3.62e-05, 3.37e-05, 3.54e-05, 3.46e-05, 6.14e-05, 3.14e-05, 6.11e-05, 6.18e-05,6.03e-05, 6.12e-05, 4.15e-05, 4.13e-05, 2.76e-05, 2.74e-05, 2.80e-05, 2.88e-05,2.89e-05, 2.72e-05, 2.65e-05, 3.68e-05, 2.41e-05, 7.79e-05, 3.51e-05, 6.08e-05,3.96e-05, 7.31e-05, 3.18e-05, 3.41e-05, 3.19e-05, 3.22e-05, 3.76e-05, 3.01e-05,3.16e-05, 3.07e-05, 7.80e-05, 1.02e-04, 8.27e-05, 7.83e-05, 1.06e-04, 7.35e-05,7.19e-05, 7.38e-05, 7.20e-05, 7.14e-05, 7.08e-05, 6.60e-05, 6.37e-05, 6.72e-05,5.55e-05, 5.48e-05, 5.22e-05, 5.50e-05, 4.90e-05, 4.78e-05, 5.01e-05, 4.85e-05,4.72e-05, 5.48e-05, 4.66e-05, 5.15e-05, 4.44e-05, 1.53e-05, 2.69e-05, 5.21e-05,6.05e-05]
        N20YieldsFe =  [9.19e-05, 1.01e-04, 9.41e-05, 1.04e-04, 5.95e-05, 5.98e-05, 6.37e-05, 6.77e-05, 6.97e-05, 9.97e-05, 9.26e-05, 8.28e-05, 9.24e-05, 7.69e-05, 7.09e-05, 6.16e-05, 6.98e-05, 6.44e-05, 4.70e-05, 5.93e-05, 4.67e-05, 4.95e-05, 4.85e-05, 4.17e-05, 4.42e-05, 4.03e-05, 3.36e-05, 4.94e-05, 4.47e-05, 4.66e-05, 4.53e-05, 4.94e-05, 5.11e-05, 5.25e-05, 5.63e-05, 5.38e-05, 5.57e-05, 5.96e-05, 5.95e-05, 4.49e-05, 4.37e-05, 4.12e-05, 4.26e-05, 3.61e-05, 4.44e-05, 3.34e-05, 4.70e-05, 4.68e-05, 4.93e-05, 5.06e-05, 5.36e-05, 4.97e-05, 8.95e-05, 8.31e-05, 8.89e-05, 8.81e-05, 7.27e-05, 4.81e-05, 3.69e-05, 4.11e-06, 2.06e-05, 6.20e-05, 8.93e-06, 1.90e-05, 4.76e-06, 1.03e-05, 2.86e-05, 4.29e-05, 5.11e-05, 5.62e-05, 6.00e-05, 1.03e-04, 1.07e-04, 1.33e-04, 3.57e-05, 6.07e-05, 7.07e-05, 7.33e-05, 3.32e-05, 2.51e-05, 2.44e-05, 2.68e-05, 3.09e-05, 3.30e-05, 3.13e-05, 3.67e-05, 3.65e-05, 3.75e-05, 4.12e-05, 4.49e-05, 4.45e-05, 4.80e-05, 5.23e-05, 5.67e-05, 5.25e-05, 5.70e-05, 5.68e-05, 1.11e-04, 1.15e-04, 1.17e-04, 1.21e-04, 1.86e-04, 7.40e-05, 2.66e-05, 3.50e-05]
        to_interp_mass = np.array( Z96Masses+N20Masses )
        to_interp_yield = np.array(Z96Yields+N20Yields)
        to_interp_yield_fe = np.array(Z96YieldsFe+N20YieldsFe)
        #self.snyield_interp = interp1d(np.log10(N20Masses), np.log10(N20Yields), kind='linear', bounds_error=False, fill_value="extrapolate")
        self.snyield_interp = interp1d(np.log10(to_interp_mass), np.log10(to_interp_yield), kind='linear', bounds_error=False, fill_value="extrapolate")
        self.snyield_interp_fe = interp1d(np.log10(to_interp_mass), np.log10(to_interp_yield_fe), kind='linear', bounds_error=False, fill_value="extrapolate")



    def get_star_with_closest_mass(self, mass, log=True):
        if not log:
            idx = get_closest( self.masses, mass )
            return self.list_of_stars[idx]
        else:
            idx = get_closest( np.log10(self.masses), np.log10(mass) )
            return self.list_of_stars[idx]
#        if log:
#            absdiff = np.abs(np.log10(self.masses / mass))
#        else:
#            absdiff = np.abs( self.masses - mass )
#        i = np.argmin(absdiff)
#        return self.list_of_stars[i]
    def estimate_aluminum_prod_rate( self, logFractionalAgeRemaining, logMass ):
        ### This is the simplest algorithm - just get the nearest one.
        nearestStars = self.get_star_with_closest_mass( np.power(10.0,logMass) )
        
        #return  nearestStar.logProdRate( np.power(10.0, logFractionalAgeRemaining) )
        results = np.zeros( np.shape(nearestStars) )
        it = np.nditer( nearestStars, flags=['multi_index','refs_ok'])
        while not it.finished:
            results[it.multi_index] = nearestStars[it.multi_index].logProdRate( np.power(10.0, logFractionalAgeRemaining[it.multi_index] ))
            it.iternext()
        return results
    def effective_lifetime(self, mass):
        #starmodel = self.get_star_with_closest_mass(np.array([mass]))[0]
        #est_lifetime = starmodel.lifetime() * np.power(mass/starmodel.mass, -1.2 )
        #est_lifetime = interp1d( np.log10(self.masses), np.log10(np.array([star.lifetime() for star in self.list_of_stars])), kind='quadratic', bounds_error=False )
        return np.power(10.0, self.est_lifetime([np.log10(mass)]))
    def totalAluminumAliveNow(self, mass, age, at20, at120, SNfac, SNflag):
        if mass<8.0:
            return 0.0,0.0
        starmodel = self.get_star_with_closest_mass(np.array([mass]))[0]
        est_lifetime = self.effective_lifetime(mass)
        al_from_wr=0.0
        if mass>=20.0:
            a,b = getPowerlaw(at20, at120)
            totalAlMassOfNearestModel = starmodel.total_aluminum_over_lifetime( est_lifetime/starmodel.lifetime() )
            renorm = a*np.power(mass,b) / totalAlMassOfNearestModel # we assume we can use the same shape of Al production as a function of time, but renormalized so that the TOTAL mass the star would produce over its whole lifetime fits into this powerlaw thing we're assuming.
            al_from_wr = starmodel.aluminum26_at_age(age,  est_lifetime/starmodel.lifetime() ) * renorm
            if np.isnan(al_from_wr) or not np.isfinite(al_from_wr):
                al_from_wr=0.0
        al_from_sn = 0.0
        if age> est_lifetime and mass > 8.0 and SNflag>0.5:
            al_from_sn += 10.0**self.snyield_interp(np.log10(mass)) * SNfac * np.power(2.0, - (age-est_lifetime) / alHalfLife  )
        if np.isnan(al_from_wr) or np.isnan(al_from_sn):
            pdb.set_trace()
        return  al_from_wr,  al_from_sn
    def totalIronAliveNow( self, mass, age, SNfac, SNflag):
        starmodel = self.get_star_with_closest_mass(np.array([mass]))[0]
        est_lifetime = self.effective_lifetime(mass)
        fe_from_wr = 0.0
        fe_from_sn = 0.0
        if age> est_lifetime and mass > 8.0 and SNflag>0.5:
            fe_from_sn += 10.0**self.snyield_interp_fe(np.log10(mass)) * SNfac * np.power(2.0, - (age-est_lifetime) / feHalfLife  )
        if np.isnan(fe_from_wr) or np.isnan(fe_from_sn):
            pdb.set_trace()
        return  fe_from_wr,  fe_from_sn


    #####z = aluminum_prod_rate( x, y ) # log10 (1-t/lifetime),  log10(M)

def read_stellar_models():
    fns = sorted(glob.glob('ekstrom_geneva/*.dat'))
    if len(fns)==0:
        print("Please download the Geneva stellar tracks from https://obswww.unige.ch/Research/evol/tables_grids2011/tablesZ014.tgz and place the untarred .dat files in a subdirectory called ekstrom_geneva, or modify the preceding line accordingly")
        assert False

    stars = []
    for fn in fns:
        stars.append( star(fn) )

    return stars

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_closest(array, values):
    #make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    return idxs

def lnIMF(mass):
    return -2.3*np.log(mass)

class alModel:
    def __init__(self, N, collection_of_stars, ageSpread, prior_mean_age, prior_age_uncertainty, prior_max_agespread, prior_minimum_mass):
        self.N = N
        self.collection = collection_of_stars
        mux,muy,cov = getNormalization()
        self.rv = multivariate_normal( [mux,muy], cov) #[[sigmax*sigmax, cov],[cov, sigmay*sigmay]] )
        self.ageSpread = ageSpread
        #ax[2].contour(np.power(10.0,x),np.power(10.0,y), rv.pdf(pos))


        self.prior_mean_age = prior_mean_age
        self.prior_age_uncertainty  = prior_age_uncertainty
        self.prior_max_agespread = prior_max_agespread # age spread 0-3 Myr uniform
        self.prior_minimum_mass  = prior_minimum_mass




        # N20 parameters:
        self.N20terms = np.array([ 105.80398061, -229.33701929, 34.16262753, -3.13691431, -144.39514404, 5.59962549,  -86.30050184,  -23.34290822,  -51.66020525,  -23.39004776, -27.40780923, 9.45836501,  -21.13176386, -219.53325422,  289.60862842, -112.49960168,  -11.05483657,  190.96947806, -272.34408252,  121.28741863])
       #W18:
        self.meansLogM =  np.array([1.08813609, 1.13987909, 1.17318627, 1.20411998, 1.23299611, 1.26007139, 1.28555731, 1.30963017, 1.33243846, 1.35218252, 1.372912, 1.39269695, 1.41161971, 1.42975228, 1.44715803, 1.46389299, 1.50514998, 2.07918125])
        self.sigmasLogM = np.array([0.051743,   0.03330718, 0.03093371, 0.02887613, 0.02707528, 0.02548592, 0.02407286, 0.02280829, 0.01974406, 0.02072948, 0.01978495, 0.01892275, 0.01813257, 0.01740575, 0.01673496, 0.04125699, 0.1, 0.1 ])
        self.meanX  = 1.3445926679420546
        self.W18terms = np.array([  77.67807251, -166.47719149,  6.29659338, 9.78236367, -108.33942684, 4.94219171, -63.99255647, -13.28033301, -46.23101207, -4.33600064, -38.68875372,   50.95756196,  -99.61714546, 46.81276916, 134.40683222, 27.8007384, 282.50158369, -247.65637725, -221.34330643, 44.39241135])


    def explosionProbability(self, masses, zeta):
        thisSetOfParams = self.W18terms + zeta*(self.N20terms-self.W18terms)
        logm = np.log10(masses)
        lor = np.zeros(len(masses))
        lor += thisSetOfParams[0]
        lor += thisSetOfParams[1] * (logm - self.meanX)
        for k in range(len(thisSetOfParams)-2):
            lor += thisSetOfParams[2+k] * np.exp( -0.5 * ( self.meansLogM[k] - logm ) * (self.meansLogM[k] - logm) / (self.sigmasLogM[k]*self.sigmasLogM[k]) )
        return 1.0/(1.0 + np.exp(-lor))

    def lnlik(self, theta):
        ret = 0
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        masses = theta[7:7+self.N]
        ageOffsets = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        assert len(masses) == len(ageOffsets)
        aluminumMassThis = 0
        for i in range(self.N):
            #aluminumMassWR, aluminumMassSN = self.collection.totalAluminumAliveNow( theta[i+4], age, at20, at120, SNfac )
            aluminumMassWR, aluminumMassSN = self.collection.totalAluminumAliveNow( masses[i], age+ageOffsets[i], at20, at120, SNfac, SNflags[i])
            aluminumMassThis +=  aluminumMassWR + aluminumMassSN
            if masses[i]<8.0:
                break
        indAges = ageOffsets + age
        lifetimes = self.collection.effective_lifetime(masses).flatten()
        alive =  lifetimes > indAges
        exploded = np.logical_and( np.logical_not(alive), SNflags)
        includeZeta = True 
        if includeZeta:
            if np.any(exploded):

                explosionTime = 1.78e6
                explosionTimeUncertainty = 0.21e6
                #print('dbg includeZeta: ', np.shape(indAges), np.shape(exploded), np.shape(lifetimes))
                ii = np.argmin( np.abs( indAges[exploded] - lifetimes[exploded] - explosionTime )) #### this isn't necessarily right - may need to think deeply about how to include multiple SNe
                ret += -0.5*np.log(2*np.pi*explosionTimeUncertainty*explosionTimeUncertainty) - 0.5 * ( indAges[exploded][ii] - lifetimes[exploded][ii] - explosionTime )**2 / (explosionTimeUncertainty * explosionTimeUncertainty)

# this is even more sketch since I don't think we believe the mass of zeta.
                #mostMassiveLiving = masses[alive][0]    
                #ret += -0.5*np.log(2*np.pi*1.0*1.0) -0.5 * (mostMassiveLiving - 20.0)**2 / 1.0**2
            else:
                ret = -np.inf




        aluminumMassMeasured, aluminumMassUncertainty = 1.1e-4, 1.1e-4 * np.sqrt(1.0**2+1.2**2)/6.1 # from Diehl 2010
        ret += -0.5*np.log(2*np.pi*aluminumMassUncertainty*aluminumMassUncertainty) -0.5 * (aluminumMassThis - aluminumMassMeasured)**2 / aluminumMassUncertainty**2
        if not aluminumMassThis > 0:
            ret = -np.inf
        if np.isnan(ret):
            ret = -np.inf
        if hasattr(ret, '__len__'):
            assert len(ret)==1
            ret=ret[0]
        if np.random.random() < 0.00001:
            print("returning a likelihood corresponding to a total aluminum mass of ", aluminumMassThis)
        return ret

    def sample_from_prior(self):
        if not self.ageSpread:
            assert False

        return self.prior_transform( np.random.random(size=7 + 3*self.N) )


    def prior_transform(self,u):
        # The parameters are: age, at20, at120, SNfac, SNprobfac, ageSpread, N x masses, N x ageOffsets, N x explosion flags
        assert len(u) == 7 + 3*self.N
        theta = np.zeros( np.shape(u) )

        theta[0] = norm.ppf(u[0], loc=self.prior_mean_age, scale=self.prior_age_uncertainty )# age ~ N(10 Myr, 3 Myr) - units of years.
        sigma1 = np.sqrt(self.rv.cov[0,0])
        sigma2 = np.sqrt(self.rv.cov[1,1])
        corr = self.rv.cov[1,0]/(sigma1*sigma2)
        theta[1] = np.power(10.0, norm.ppf(u[1], loc=self.rv.mean[0], scale=sigma1) ) # Yield At20 solar masses. 2d gaussian -> marginal is just the 1d gaussian
        theta[2] = np.power(10.0, norm.ppf(u[2], loc=self.rv.mean[1] + sigma2/sigma1 * corr * (np.log10(theta[1]) - self.rv.mean[0]), scale=np.sqrt(1.0-corr*corr)*sigma2))
        theta[3] = np.power(10.0, norm.ppf(u[3], scale=np.log10(2.0))) # factor of 2 uncertainty
        theta[4] = u[4]*2-0.5 #### ????? SNprobfac "zeta" elsewhere in the code
        theta[5] = u[5]*self.prior_max_agespread # age spread 0-3 Myr uniform
        alpha = 2.3 # slope of the IMF
        #Mmin = 5.0 # solar masses
        assert self.prior_minimum_mass < 8.0
        #theta[6] = u[6]*(8.0-self.prior_minimum_mass) + self.prior_minimum_mass # may require deeper thought
        #Mmin = theta[6]
        Mmin = self.prior_minimum_mass
        theta[7] = (1.0-u[7]**(1.0/float(self.N)))**(1.0/(1.0-alpha)) * Mmin  # maximum mass star (solar masses)
        for i in range(self.N-1):
            j = float( self.N - (i+1) )
            theta[8+i] = Mmin * (1.0 - u[8+i]**(1.0/j) *(1.0 - (theta[7+i]/Mmin)**(1.0-alpha)) )**(1.0/(1.0-alpha))

        # age offsets
        for i in range(self.N):
            theta[7+self.N+i] = norm.ppf(u[7+self.N+i], loc=0, scale=theta[5])


        # explosion flags

        masses = theta[7:7+self.N]
        probs = self.explosionProbability(masses, theta[4])

        theta[7+2*self.N:] = np.sign( u[7+2*self.N:] - 1.0 + probs)
        return theta
    def fractionsFe(self, theta ):
# k is an index of possible multiple datasets are being plotted.
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        feDist = np.zeros( (self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]

        for i in range(self.N):
            feMassWR, feMassSN = self.collection.totalIronAliveNow( masses[i], age+ageAdjustments[i], SNfac, SNflags[i] )
            if np.isnan(feMassWR) or np.isnan(feMassSN):
                pdb.set_trace()
            feDist[i,0] = feMassWR
            feDist[i,1] = feMassSN
        return feDist


    def fractions(self, theta, axIn=None, plot=True, k=0, weights=None):
# k is an index of possible multiple datasets are being plotted.
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        alDist = np.zeros( (self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        if weights is None:
            weights = 1.0

        for i in range(self.N):
            alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[i], age+ageAdjustments[i], at20, at120, SNfac, SNflags[i] )
            if np.isnan(alMassWR) or np.isnan(alMassSN):
                pdb.set_trace()
            alDist[i,0] = alMassWR
            alDist[i,1] = alMassSN

        if plot:
            if axIn is None:
                fig,ax = plt.subplots()
            else:
                ax = axIn
            ax.scatter( masses, alDist[:,0], c='b', lw=0, s=40/(k+1), alpha=weights*0.4/(k+1) )
            ax.scatter( masses, alDist[:,1], c='r', lw=0, s=40/(k+1), alpha=weights*0.4/(k+1) )
            #print("Plotting minstar, maxstar, minal, maxal:", np.min(theta[4:]), np.max(theta[4:]), np.min(alDist[:,0]), np.max(alDist[:,0]), np.min(alDist[:,1]), np.max(alDist[:,1]))

            if axIn is None:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel(r'Star Mass (M$_\odot$)')
                ax.set_ylabel(r'Mass of Living $^{26}$Al from one star (M$_\odot$)')
                plt.savefig('chain_aluminum_frac.pdf')
                plt.close(fig)
        return alDist

    def coreAccumulateFast(self, theta, minTime=0.7, maxTime=1.3, chi=1.0, rcore=1.0e4,  u=None, tcore=1.0e6, preEnriched=True, drawFrom3D=False ):
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        times = np.linspace( 0 + age*minTime, age*maxTime, 30 )
        alDist = np.zeros( (len(times), self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
# Likely irrelevant since no dynamics are assumed in this version.
        tff = 177912.377 * np.sqrt(chi) * np.power(rcore/1.0e4,1.5) # years = sqrt(3*pi / (32*G*rho)) where rho = 1 Msun uniformly distributed in a sphere of radius 10^4 AU
        alMass = 0.0
        accumTerm = 0.0
        preTerm = 0.0
        accumTermFixedDist = 0.0
        preTermFixedDist = 0.0

        t = age


        for k in range(self.N):
# these are just scalars in solar masses
            alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[k], t+ageAdjustments[k], at20, at120, SNfac, SNflags[k] ) 
            alMass += alMassWR + alMassSN
            starmodel = self.collection.get_star_with_closest_mass(np.array([masses[k]]))[0]
            est_lifetime = self.collection.effective_lifetime(masses[k])

            di = fDist( np.random.random() ) # draw from the 3D distance distribution
            tCoreForm = t - tcore
# this condition should be "the star died before the core formed" The star is dead when est_lifetime < t+ageAdjustments[k], and t=tCoreForm. So! est_lifetime < tCoreForm+ageAdjustments[k]
            if est_lifetime < tCoreForm + ageAdjustments[k]:
                accumTerm  += (alMassWR + alMassSN) / (di*di*di)
                preTerm += (alMassWR + alMassSN) / (di*di*di)
                accumTermFixedDist  += (alMassWR + alMassSN) 
                preTermFixedDist += (alMassWR + alMassSN) 
# this condition is that the star dies during the core's existence
# so, est_lifetime < t+ageAdjustments[k] and est_lifetime>tCoreform+ageAdjustments[k]
            #elif tCoreForm <= t+ageAdjustments[k] < tCoreForm + tcore:
            elif est_lifetime < t+ageAdjustments[k] and est_lifetime>tCoreForm + ageAdjustments[k]:
# this prefactor should be the fraction of tcore during which the star has been dead. The star dies at t=est_lifetime -ageAdjustments[k]
# therefore it has been dead for t - (est_lifetime-ageAdjustments[k])
                #accumTerm += (tCoreForm+tcore - (t+ageAdjustments[k]))/tcore *  (alMassWR + alMassSN) / (di*di*di)
                assert (t - est_lifetime+ageAdjustments[k])/tcore <= 1.0 and (t - est_lifetime+ageAdjustments[k])/tcore>=0.0 # this should be true. If not either this fraction or the conditions for the elif statement are messed up
                accumTerm += (t - est_lifetime+ageAdjustments[k])/tcore *  (alMassWR + alMassSN) / (di*di*di)
                accumTermFixedDist += (t - est_lifetime+ageAdjustments[k])/tcore *  (alMassWR + alMassSN) 
# This remaining case is that the star is still alive and well at the present time, in which case we consider it to be contributing relatively little
            #elif t+ageAdjustments[k] >= tCoreForm + tcore:
            else:
                pass

        accumTerm *= .05 * (rcore/1.0e4)*(rcore/1.0e4) * (tcore/1.0e6) * 0.00180283152 # = (10^4 AU)^2 * 1e6 yrs * 1 km/s * 3/4
        preTerm *= chi * (rcore/1.0e4)**3 * 0.000113952695 # = (10^4 AU)^3 in pc^3

        accumTermFixedDist *= .05 * (rcore/1.0e4)*(rcore/1.0e4) * (tcore/1.0e6) * 0.00180283152 # = (10^4 AU)^2 * 1e6 yrs * 1 km/s * 3/4
        preTermFixedDist *= chi * (rcore/1.0e4)**3 * 0.000113952695 # = (10^4 AU)^3 in pc^3

        return accumTerm, preTerm, accumTermFixedDist, preTermFixedDist

    def AlMassAtTime(self, theta, t):
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        alMass = 0.0
        for i in range(self.N):
            if t+ageAdjustments[i]>0:
                alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[i], t+ageAdjustments[i], at20, at120, SNfac, SNflags[i] )
                alMass += alMassWR+alMassSN
        return alMass

    def atTimeFe(self, theta, theTime, axIn=None, reverse=False, k=0, weight=1.0):
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]

        feDist = np.zeros( ( self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        for i in range(self.N):
            thisTime = theTime+ageAdjustments[i]
            if reverse:
                thisTime = age - thisTime
            feMassWR, feMassSN = self.collection.totalIronAliveNow( masses[i], thisTime, SNfac, SNflags[i] )
            feDist[i,:] = (feMassWR, feMassSN)
        return np.sum(feDist[:,0]), np.sum(feDist[:,1]), np.sum(feDist)




    def atTime(self, theta, theTime, axIn=None, reverse=False, k=0, weight=1.0):
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]

        alDist = np.zeros( ( self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        for i in range(self.N):
            thisTime = theTime+ageAdjustments[i]
            if reverse:
                thisTime = age - thisTime
            alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[i], thisTime, at20, at120, SNfac, SNflags[i] )
            alDist[i,:] = (alMassWR, alMassSN)
        return np.sum(alDist[:,0]), np.sum(alDist[:,1]), np.sum(alDist)



    def vsTimeFe(self, theta, axIn=None, reverse=False, minTime=0, maxTime=1, k=0, weight=1.0):
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        times = np.linspace( 0 + age*minTime, age*maxTime, 200 )
        alDist = np.zeros( (len(times), self.N, 2) )
        feDist = np.zeros( (len(times), self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        for j in range(len(times)):
            for i in range(self.N):
                alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[i], times[j]+ageAdjustments[i], at20, at120, SNfac, SNflags[i] )
                feMassWR, feMassSN = self.collection.totalIronAliveNow( masses[i], times[j]+ageAdjustments[i], SNfac, SNflags[i] )
                alDist[j,i,:] = (alMassWR, alMassSN)
                feDist[j,i,:] = (feMassWR, feMassSN)

        if axIn is None:
            fig,ax = plt.subplots()
        else:
            ax = axIn

        linestyles = ['-','-.','--',':']*10


        ind = np.argmin( np.abs(age - times))
        rat = np.sum(alDist[ind,:,0]) / ( np.sum(alDist[ind,:,0])+ np.sum(alDist[ind,:,1]) )
        if rat<0.2:
            c='r'
        elif rat>0.8:
            c='b'
        else:
            c='purple'


        if not reverse:
# before obs
            ax.plot( times/1.0e6, np.sum(np.sum( feDist[ :, :, :], axis=2 ),axis=1), c=c, alpha=0.5*weight, lw=1, ls=linestyles[k])
            ax.scatter( times[ind]/1.0e6, np.sum(np.sum( feDist[ :, :, :], axis=2 ),axis=1)[ind], c='k', alpha=1.0*weight, lw=0, zorder=10)
        if reverse:
            ax.plot( (age-times)/1.0e6, np.sum(np.sum( feDist[ :, :, :], axis=2 ),axis=1), c=c, alpha=0.5*weight, lw=1, ls=linestyles[k])
            ax.scatter( (age-times)[ind]/1.0e6, np.sum(np.sum( feDist[ :, :, :], axis=2 ),axis=1)[ind], c='k', alpha=1.0*weight, lw=0, zorder=10)

        if axIn is None:
            ax.set_yscale('log')
            if not reverse:
                ax.set_xlabel(r'Time (yr)')
            if reverse:
                ax.set_xlabel(r'Lookback Time (yr)')
            ax.set_ylabel(r'Living $^{60}$Fe (M$_\odot$) ')
            plt.savefig('chain_iron_vsTime.pdf')
            plt.close(fig)




    def vsTimeFeAl(self, theta, axIn=None, reverse=False, minTime=0, maxTime=1, k=0, weight=1.0):
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        times = np.linspace( 0 + age*minTime, age*maxTime, 200 )
        alDist = np.zeros( (len(times), self.N, 2) )
        feDist = np.zeros( (len(times), self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        for j in range(len(times)):
            for i in range(self.N):
                alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[i], times[j]+ageAdjustments[i], at20, at120, SNfac, SNflags[i] )
                feMassWR, feMassSN = self.collection.totalIronAliveNow( masses[i], times[j]+ageAdjustments[i], SNfac, SNflags[i] )
                alDist[j,i,:] = (alMassWR, alMassSN)
                feDist[j,i,:] = (feMassWR, feMassSN)

        if axIn is None:
            fig,ax = plt.subplots()
        else:
            ax = axIn

        linestyles = ['-','-.','--',':']*10


        ind = np.argmin( np.abs(age - times))
        rat = np.sum(alDist[ind,:,0]) / ( np.sum(alDist[ind,:,0])+ np.sum(alDist[ind,:,1]) )
        if rat<0.2:
            c='r'
        elif rat>0.8:
            c='b'
        else:
            c='purple'


        if not reverse:
# before obs
            #ax.plot( times[:ind]/1.0e6, np.sum( alDist[ :, :, 0], axis=1 )[:ind], c='lightblue', alpha=0.2*weight, ls=linestyles[k])
            #ax.plot( times[:ind]/1.0e6, np.sum( alDist[ :, :, 1], axis=1 )[:ind], c='pink', alpha=0.2*weight, ls=linestyles[k])
            #ax.plot( times[:ind]/1.0e6, np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1)[:ind], c='gray', alpha=0.8*weight, lw=2, ls=linestyles[k])

#after obs
            #ax.plot( times[ind:]/1.0e6, np.sum( alDist[ :, :, 0], axis=1 )[ind:], c='darkblue', alpha=0.2*weight, ls=linestyles[k])
            #ax.plot( times[ind:]/1.0e6, np.sum( alDist[ :, :, 1], axis=1 )[ind:], c='maroon', alpha=0.2*weight, ls=linestyles[k])
            #ax.plot( times[ind:]/1.0e6, np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1)[ind:], c='k', alpha=0.8*weight, lw=2, ls=linestyles[k])
            #ax.plot( times/1.0e6, np.sum( alDist[ :, :, 0], axis=1 ), c='blue', alpha=0.2*weight, ls=linestyles[k])
            #ax.plot( times/1.0e6, np.sum( alDist[ :, :, 1], axis=1 ), c='red', alpha=0.2*weight, ls=linestyles[k])
            ax.plot( times/1.0e6, np.sum(np.sum( feDist[ :, :, :], axis=2 ),axis=1)/np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1), c=c, alpha=0.5*weight, lw=1, ls=linestyles[k])
            ax.scatter( times[ind]/1.0e6, np.sum(np.sum( feDist[ :, :, :], axis=2 ),axis=1)[ind]/ np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1)[ind], c='k', alpha=1.0*weight, lw=0, zorder=10)
        if reverse:
            #ax.plot( (age-times)/1.0e6, np.sum( alDist[ :, :, 0], axis=1 ), c='b', alpha=0.2*weight, ls=linestyles[k])
            #ax.plot( (age-times)/1.0e6, np.sum( alDist[ :, :, 1], axis=1 ), c='r', alpha=0.2*weight, ls=linestyles[k])
            ax.plot( (age-times)/1.0e6, np.sum(np.sum( feDist[ :, :, :], axis=2 ),axis=1)/ np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1), c=c, alpha=0.5*weight, lw=1, ls=linestyles[k])
            ax.scatter( (age-times)[ind]/1.0e6, np.sum(np.sum( feDist[ :, :, :], axis=2 ),axis=1)[ind]/ np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1)[ind], c='k', alpha=1.0*weight, lw=0, zorder=10)

        #ax.scatter( theta[4:], alDist[:,0], c='b', lw=0, s=20, alpha=0.2 )
        #ax.scatter( theta[4:], alDist[:,1], c='r', lw=0, s=20, alpha=0.2 )

        #print("Plotting minstar, maxstar, minal, maxal:", np.min(theta[4:]), np.max(theta[4:]), np.min(alDist[:,0]), np.max(alDist[:,0]), np.min(alDist[:,1]), np.max(alDist[:,1]))

        if axIn is None:
            #ax.set_xscale('log')
            ax.set_yscale('log')
            if not reverse:
                ax.set_xlabel(r'Time (yr)')
            if reverse:
                ax.set_xlabel(r'Lookback Time (yr)')
            ax.set_ylabel(r'Ratio of Living $^{60}$Fe/ $^{26}$Al ')
            plt.savefig('chain_ironaluminum_vsTime.pdf')
            plt.close(fig)


    def vsTime(self, theta, axIn=None, reverse=False, minTime=0, maxTime=1, k=0, weight=1.0):
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        times = np.linspace( 0 + age*minTime, age*maxTime, 200 )
        alDist = np.zeros( (len(times), self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        for j in range(len(times)):
            for i in range(self.N):
                alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[i], times[j]+ageAdjustments[i], at20, at120, SNfac, SNflags[i] )
                alDist[j,i,:] = (alMassWR, alMassSN)

        if axIn is None:
            fig,ax = plt.subplots()
        else:
            ax = axIn

        linestyles = ['-','-.','--',':']*10


        ind = np.argmin( np.abs(age - times))
        rat = np.sum(alDist[ind,:,0]) / ( np.sum(alDist[ind,:,0])+ np.sum(alDist[ind,:,1]) )
        if rat<0.2:
            c='r'
        elif rat>0.8:
            c='b'
        else:
            c='purple'


        if not reverse:
# before obs
            ax.plot( times/1.0e6, np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1), c=c, alpha=0.5*weight, lw=1, ls=linestyles[k])
            ax.scatter( times[ind]/1.0e6, np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1)[ind], c='k', alpha=1.0*weight, lw=0, zorder=10)
        if reverse:
            ax.plot( (age-times)/1.0e6, np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1), c=c, alpha=0.5*weight, lw=1, ls=linestyles[k])
            #ax.scatter( (age-times)[ind]/1.0e6, np.sum(np.sum( alDist[ :, :, :], axis=2 ),axis=1)[ind], c='k', alpha=1.0*weight, lw=0, zorder=10)


        if axIn is None:
            #ax.set_xscale('log')
            ax.set_yscale('log')
            if not reverse:
                ax.set_xlabel(r'Time (yr)')
            if reverse:
                ax.set_xlabel(r'Lookback Time (yr)')
            ax.set_ylabel(r'Mass of Living $^{26}$Al (M$_\odot$)')
            plt.savefig('chain_aluminum_vsTime.pdf')
            plt.close(fig)
            
    def vsTimeFrac(self, theta, axIn=None, minAge=0.0, maxAge=1.0, reverse=False, k=0, weight=1.0):
        age = theta[0]
        at20 = theta[1]
        at120 = theta[2]
        SNfac = theta[3]
        times = np.linspace( 0 + age*minAge, age*maxAge, 200 )
        alDist = np.zeros( (len(times), self.N, 2) )
        masses = np.zeros(self.N)
        ageAdjustments = np.zeros(self.N)
        masses = theta[7:7+self.N]
        ageAdjustments = theta[7+self.N:7+2*self.N]
        SNflags = theta[7+2*self.N:]
        for j in range(len(times)):
            for i in range(self.N):
                #alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[i], age + ageAdjustments[i] - times[j], at20, at120, SNfac ) 
                if ageAdjustments[i] + times[j]>0:
                    alMassWR, alMassSN = self.collection.totalAluminumAliveNow( masses[i], ageAdjustments[i] + times[j], at20, at120, SNfac, SNflags[i] ) 
                    alDist[j,i,0] = alMassWR
                    alDist[j,i,1] = alMassSN

        if axIn is None:
            fig,ax = plt.subplots()
        else:
            ax = axIn

        list_of_linestyles = ['-','-.','--',':']*10
        y = np.sum( alDist[ :, :, 0], axis=1 )/np.sum(np.sum( alDist[ :, :, :], axis=2 ), axis=1)
        if reverse:
            ax.plot( (age - times[::-1])/1.0e6, y[::-1] , c='k', alpha=0.02)
        else:
            #ind = int( len(times) * maxAge )
            #ind = len(times) * age * (maxAge-minAge)
            ind = np.argmin( np.abs(times-age))
            ax.plot( times[:ind]/1.0e6, y[:ind] , c='green', alpha=0.1*weight, ls=list_of_linestyles[k])
            ax.plot( times[ind:]/1.0e6, y[ind:] , c='orange', alpha=0.1*weight, ls=list_of_linestyles[k])
            ax.scatter( [times[ind]/1.0e6], [y[ind]], c='k', alpha=1.0*weight, s=30, lw=0) # mark the point where the measurement actually occurs according to the model

        #ax.scatter( theta[4:], alDist[:,0], c='b', lw=0, s=20, alpha=0.2 )
        #ax.scatter( theta[4:], alDist[:,1], c='r', lw=0, s=20, alpha=0.2 )

        #print("Plotting minstar, maxstar, minal, maxal:", np.min(theta[4:]), np.max(theta[4:]), np.min(alDist[:,0]), np.max(alDist[:,0]), np.min(alDist[:,1]), np.max(alDist[:,1]))

        if axIn is None:
            #ax.set_xscale('log')
            ax.set_yscale('log')
            if reverse:
                ax.set_xlabel(r'Lookback Time (Myr)')
            else:
                ax.set_xlabel(r'Time (Myr)')
            ax.set_ylabel(r'Fraction of Living $^{26}$Al from WR')
            plt.savefig('chain_aluminum_vsTimeFrac.pdf')
            plt.close(fig)

def lnlik(theta, theModel):
    ret = theModel.lnlik(theta)
    return ret

def ptransform(u, theModel):
    theta = theModel.prior_transform(u)
    assert np.shape(theta)==np.shape(u)
    return theta

def lnprob(theta, theModel, runEmceeOnPrior):
    prior = theModel.lnprior(theta)
    if not np.isfinite(prior) or runEmceeOnPrior:
        return prior, -np.inf
    else:
        return prior + theModel.lnlik(theta), prior


def analyze_sample( samples, bn, theModel, more_datasets=[], weights=None):
    print( "BEGINNING TO ANALYZE SAMPLE WITH SHAPE: ", np.shape(samples), 'NAMED',bn, len(more_datasets) )
    ndatasets = 1+len(more_datasets)
    if ndatasets>1:
        for k in range(ndatasets-1):
            assert np.shape(samples) == np.shape(more_datasets[k])

    nsamples = np.shape(samples)[0]
    wtnorm = copy.deepcopy(weights)
    if weights is None:
        wtnorm = np.ones(nsamples)
    else:
        wtnorm = weights/np.max(weights)
        assert len(weights)==nsamples

    N = theModel.N
    ages = np.zeros((nsamples, ndatasets))
    ageSpreads = np.zeros((nsamples, ndatasets))
    zetas = np.zeros((nsamples, ndatasets))
    mmins = np.zeros((nsamples, ndatasets))
    at20s= np.zeros((nsamples, ndatasets))
    at120s= np.zeros((nsamples, ndatasets))
    fSNs= np.zeros((nsamples, ndatasets))
    Ngtr20 = np.zeros((nsamples, ndatasets))
    Ngtr8living = np.zeros((nsamples, ndatasets))
    fractions = np.zeros((nsamples, ndatasets))
    alMasses = np.zeros((nsamples, N, ndatasets))
    feMasses = np.zeros((nsamples, N, ndatasets))
    masses = np.zeros((nsamples, N, ndatasets) )
    indAges = np.zeros((nsamples, N, ndatasets)) 
    alive = np.zeros((nsamples, N, ndatasets), dtype=bool) 
    exploded = np.zeros((nsamples, N, ndatasets), dtype=bool) 
    timesOfExplosion = np.zeros((nsamples, N, ndatasets)) 
    mostRecentExplosion = np.zeros((nsamples, ndatasets))

    massOfLargestAluminumSource = np.zeros((nsamples, ndatasets))
    ageOfLargestAluminumSource = np.zeros((nsamples, ndatasets))
    timeSinceDeathOfLargestAluminumSource = np.zeros((nsamples,ndatasets))
    numberOfStarsToReach90percentAluminum = np.zeros((nsamples,ndatasets))

    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            alDistThis = theModel.fractions( samplesThis[i,:], axIn=None, k=k, weights=wtnorm[i] ,plot=False) 
            if np.any(np.isnan(alDistThis)):
                pdb.set_trace()
            feDistThis = theModel.fractionsFe( samplesThis[i,:]) 
            ages[i, k] = samplesThis[i,0]
            ageSpreads[i, k] = samplesThis[i,5] 
            zetas[i, k] = samplesThis[i,4]  
            mmins[i, k] = samplesThis[i,6]  
            at20s[i,k] = samplesThis[i,1] 
            at120s[i,k] = samplesThis[i,2]
            fSNs[i,k] = samplesThis[i,3]
            if np.sum(alDistThis)>0:
                fractions[i,k] =  np.sum(alDistThis[:,0])/np.sum(alDistThis) 
            alMasses[i,:, k] = np.sum(alDistThis, axis=1) 
            feMasses[i,:, k] = np.sum(feDistThis, axis=1)
            masses[i,:,k] = samplesThis[i,7:7+N]
            indAges[i,:,k] = samplesThis[i,7+N:7+2*N] + samplesThis[i,0]
                ### theseLifetimes[i] = theModel.collection.effective_lifetime(mass)
            alive[i,:,k] = theModel.collection.effective_lifetime(masses[i,:,k]) > indAges[i,:,k]
            exploded[i,:,k] = np.logical_and( theModel.collection.effective_lifetime(masses[i,:,k]) < indAges[i,:,k], samplesThis[i,7+2*N:]>0.5 )
            timesSinceExplosion = (indAges[i,:,k] - theModel.collection.effective_lifetime(masses[i,:,k])).flatten()

            jj = np.argmax( alMasses[i,:,k] )
            massOfLargestAluminumSource[i,k] = masses[i,jj,k]
            ageOfLargestAluminumSource[i,k] = indAges[i,jj,k]
            timeSinceDeathOfLargestAluminumSource[i,k] =  indAges[i,jj,k] - theModel.collection.effective_lifetime(masses[i,jj,k]).flatten()
            if np.any(np.isnan(alMasses[i,:,k])):
                pdb.set_trace()
            if np.any(alMasses[i,:,k]>0):
                cu = sorted(alMasses[i,:,k])
                cu = cu[::-1]
                cu = np.cumsum( cu )
                cu/=cu[-1]
                cuInds = np.arange(len(cu))+1
                if np.any(cu>=0.9):
                    cuInds = cuInds[cu>=0.9]
                else:
                    pdb.set_trace()
                numberOfStarsToReach90percentAluminum[i,k] =  cuInds[0]
            else:
                numberOfStarsToReach90percentAluminum[i,k] = 0

            assert len(timesSinceExplosion)==N
            assert np.all( timesSinceExplosion[exploded[i,:,k]] > 0)
            if np.any(exploded[i,:,k]):
                timesOfExplosion[i,exploded[i,:,k],k] = timesSinceExplosion[exploded[i,:,k]]
                timesOfExplosion[i,np.logical_not(exploded[i,:,k]),k] = -1.0e6
                indOfMostRecentExplosion = np.argmin(  timesSinceExplosion[exploded[i,:,k]] )
                mostRecentExplosion[i,k] = timesSinceExplosion[exploded[i,:,k]][indOfMostRecentExplosion] 
            else:
                mostRecentExplosion[i,k] = 2.0e6

            Ngtr20[i,k] = np.sum( masses[i,:,k]>20 )
            Ngtr8living[i,k] = np.sum( masses[i,alive[i,:,k],k]>8 )

            if False:
                if fractions[i,k]<0.2 and numberOfStarsToReach90percentAluminum[i,k]==1:
                    pass # select this sample
                else:
                    wtnorm[i] = 0

    print("Finished aluminum frac plot")


    print("Enumerate a table of outcomes - # of sources x WR/SN")
    k=0
    #arr = np.dstack(( numberOfStarsToReach90percentAluminum[:,k].flatten(), fractions[:,k].flatten()))
    h, xe, ye = np.histogram2d( numberOfStarsToReach90percentAluminum[:,k].flatten(), fractions[:,k].flatten(), bins=(10, 101), range=[[-0.5, 9.5 ],[-0.005, 1.005 ]], weights=wtnorm, density=True )
    print( "                1 src        2 src      >2 srcs       total")
    print( "SN-dominated: ",np.sum(h[1,0:21]),"; ",np.sum(h[2,0:21]),"; ", np.sum(h[3:,0:21]),"; ", np.sum(h[:,0:21])  )
    print( "mixture: ",np.sum(h[1,21:80]),"; ",np.sum(h[2,21:80]),"; ", np.sum(h[3:,21:80]),"; ", np.sum(h[:,21:80])  )
    print( "WR-dominated: ",np.sum(h[1,80:]),"; ",np.sum(h[2,80:]),"; ", np.sum(h[3:,80:]),"; ", np.sum(h[:,80:])  )
    print( "Total: ",np.sum(h[1,:]),"; ",np.sum(h[2,:]),"; ", np.sum(h[3:,:]),"; ", np.sum(h[:,:])  )

    list_of_colors = ['r', 'purple', 'b', 'orange', 'lightblue', 'pink', 'green', 'gray', 'k']*10

    fractionOfLines = 200.0 / float(nsamples) # it's tough to see any more than like 300 lines. 


    fig,ax = plt.subplots()
    for k in range(ndatasets):
        for i in range(nsamples):
            if np.random.random()<fractionOfLines:
                ax.plot( masses[i,:,k], indAges[i,:,k]/1.0e6, c=list_of_colors[k], lw=1, alpha=0.1*wtnorm[i] )
                sc = ax.scatter( masses[i,:,k], indAges[i,:,k]/1.0e6, c=np.log10(alMasses[i,:,k]), vmin=-6, vmax=-3, lw=0, alpha=0.5*wtnorm[i] )
    cbar = plt.colorbar(sc)
    cbar.set_label(r'log Living Aluminum Mass')
    ax.set_xlabel('M (Msun)')
    ax.set_ylabel('age (Myr)')
    ax.set_xscale('log')
    plt.savefig(bn+'_mass_vs_age.pdf')
    plt.close(fig)
    print("Finished mass vs age plot")

    list_of_cmaps = ['cubehelix_r', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']*10

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        sc = ax.scatter( ages[:,k]/1.0e6, fractions[:,k], c='k', vmin=1, vmax=10, alpha=0.02, lw=0, s=5 )
    #cbar = plt.colorbar(sc)
    #cbar.set_label(r'Number of Stars with $M>20 M_\odot$')
    ax.set_xlabel(r'Age of Upper Sco (Myr)')
    ax.set_ylabel(r'Fraction of Live $^{26}$Al from WR')
    ax.set_ylim(-0.031,1.031)
    plt.savefig(bn+'_aluminum_frac_vs_age.pdf')
    plt.close(fig)
    print("Finished aluminum_frac_vs_age plot")

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            if np.random.random()<fractionOfLines*0.1:
                theModel.vsTime( samplesThis[i,:], axIn=ax, minTime=0, maxTime=2, k=k, weight=wtnorm[i] )  
    ax.set_xlabel(r'Time since cluster birth (Myr)')
    ax.set_ylabel(r'$^{26}$Al Alive (M$_\odot$)')
    ax.set_yscale('log')
    ax.set_ylim(2.0e-7, 3.0e-3)
    ax.set_xlim(-2,22)
    #ax.set_xlim(1.0e5, 1.3e7)
    plt.savefig(bn+'_aluminumIndividual_vsTime.pdf')
    plt.close(fig)
    print("Finished aluminum_vsTime plot")


    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            if np.random.random()<fractionOfLines*0.1:
                theModel.vsTimeFeAl( samplesThis[i,:], axIn=ax, minTime=0, maxTime=2, k=k, weight=wtnorm[i] )  
    ax.set_xlabel(r'Time since cluster birth (Myr)')
    ax.set_ylabel(r'$^{60}$Fe/$^{26}$Al Alive ')
    ax.set_yscale('log')
    #ax.set_ylim(2.0e-7, 3.0e-3)
    ax.set_xlim(-2,22)
    #ax.set_xlim(1.0e5, 1.3e7)
    plt.savefig(bn+'_ironAluminum_vsTime.pdf')
    plt.close(fig)


    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            if np.random.random()<fractionOfLines*0.1:
                theModel.vsTimeFe( samplesThis[i,:], axIn=ax, minTime=0, maxTime=2, k=k, weight=wtnorm[i] )  
    ax.set_xlabel(r'Time since cluster birth (Myr)')
    ax.set_ylabel(r'$^{60}$Fe Alive (M$_\odot$)')
    ax.set_yscale('log')
    #ax.set_ylim(2.0e-7, 3.0e-3)
    ax.set_xlim(-2,22)
    #ax.set_xlim(1.0e5, 1.3e7)
    plt.savefig(bn+'_iron_vsTime.pdf')
    plt.close(fig)

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            if np.random.random()<fractionOfLines*0.1:
                theModel.vsTimeFe( samplesThis[i,:], axIn=ax, minTime=0, maxTime=2, k=k, weight=wtnorm[i] , reverse=True)  
    ax.set_xlabel(r'Lookback Time, $t_\mathrm{obs} - t$ (Myr)')
    ax.set_ylabel(r'Mass of $^{60}$Fe Alive at Time $t$ (M$_\odot$)')
    ax.set_yscale('log')
    #ax.set_ylim(2.0e-7, 3.0e-3)
    ax.set_xlim(-2,22)
    #ax.set_xlim(1.0e5, 1.3e7)
    plt.savefig(bn+'_iron_vsLookbackTime.pdf')
    plt.close(fig)



    if True:
        nsamplesDist = np.min( [nsamples, 10000] )
        sampleIndices = np.random.choice(nsamples, size=nsamplesDist, replace=False)
        fig,ax = plt.subplots()
        theTimes = np.linspace(0, 20.0e6, 40)
        quantiles = [ 0.025, 0.16, 0.5, 0.84, 0.975 ]
        for k in range(ndatasets):
            atTimeArr = np.zeros( (nsamplesDist, len(theTimes), 3) )
            timeArrQuantiles = np.zeros( ( 5, len(theTimes), 3) )
            samplesThis = samples
            if k>0:
                samplesThis = more_datasets[k-1]
            for i in range(nsamplesDist):
                for j in range(len(theTimes)):
                    atTimeArr[i, j, :] =  theModel.atTime( samplesThis[sampleIndices[i],:], theTimes[j], axIn=ax, k=k)  
            for j in range(len(theTimes)):
                timeArrQuantiles[:,j,0] = np.quantile(atTimeArr[:,j,0], quantiles)
                timeArrQuantiles[:,j,1] = np.quantile(atTimeArr[:,j,1], quantiles)
                timeArrQuantiles[:,j,2] = np.quantile(atTimeArr[:,j,2], quantiles)
            ax.fill_between( theTimes/1.0e6, timeArrQuantiles[1,:,0], timeArrQuantiles[3,:,0], color='b', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[1,:,0], color='b', alpha=1.0 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[3,:,0], color='b', alpha=1.0 )
            ax.fill_between( theTimes/1.0e6, timeArrQuantiles[1,:,1], timeArrQuantiles[3,:,1], color='r', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[1,:,1], color='r', alpha=1.0 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[3,:,1], color='r', alpha=1.0 )
            #ax.fill_between( theTimes, timeArrQuantiles[1,:,2], timeArrQuantiles[3,:,2], color='k', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[2,:,0], c='b', lw = 4, zorder=3, label='WR' )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[2,:,1], c='r', lw = 4, zorder=3, label='SN' )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[2,:,2], c='k', lw = 5, zorder=3, label='Total' )
            
        ax.legend()
        ax.set_xlabel(r'$t - t_\mathrm{SF}$ (Myr)')
        ax.set_ylabel(r'Mass of $^{26}$Al Alive at time $t$ (M$_\odot$)')
        #ax.set_ylabel(r'dM$_{^{26}Al}$/dt (M$_\odot$/yr)')
        ax.set_yscale('log')
        ax.set_ylim(2.0e-8, 1.0e-3)
        #ax.set_xlim(1.0e5, 1.3e7)
        plt.savefig(bn+'_aluminumDistr_vsTime.pdf')
        plt.close(fig)
        print("Finished aluminum_vsTime plot")


        fig,ax = plt.subplots()
        theTimes = np.linspace(0, 15.0e6, 40)
        quantiles = [ 0.025, 0.16, 0.5, 0.84, 0.975 ]
        for k in range(ndatasets):
            atTimeArr = np.zeros( (nsamplesDist, len(theTimes), 3) )
            timeArrQuantiles = np.zeros( ( 5, len(theTimes), 3) )
            samplesThis = samples
            if k>0:
                samplesThis = more_datasets[k-1]
            for i in range(nsamplesDist):
                for j in range(len(theTimes)):
                    atTimeArr[i, j, :] =  theModel.atTime( samplesThis[sampleIndices[i],:], theTimes[j], axIn=ax, k=k, reverse=True)  
            for j in range(len(theTimes)):
                timeArrQuantiles[:,j,0] = np.quantile(atTimeArr[:,j,0], quantiles)
                timeArrQuantiles[:,j,1] = np.quantile(atTimeArr[:,j,1], quantiles)
                timeArrQuantiles[:,j,2] = np.quantile(atTimeArr[:,j,2], quantiles)
            ax.fill_between( theTimes/1.0e6, timeArrQuantiles[1,:,0], timeArrQuantiles[3,:,0], color='b', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[1,:,0], color='b', alpha=1.0 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[3,:,0], color='b', alpha=1.0 )
            ax.fill_between( theTimes/1.0e6, timeArrQuantiles[1,:,1], timeArrQuantiles[3,:,1], color='r', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[1,:,1], color='r', alpha=1.0 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[3,:,1], color='r', alpha=1.0 )
            #ax.fill_between( theTimes, timeArrQuantiles[1,:,2], timeArrQuantiles[3,:,2], color='k', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[2,:,0], c='b', lw = 4, zorder=3, label='WR' )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[2,:,1], c='r', lw = 4, zorder=3, label='SN' )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[2,:,2], c='k', lw = 8, zorder=3, label='Total' )
            
        ax.legend()
        ax.set_xlabel(r'Lookback Time, $t_\mathrm{obs} - t$ (Myr)')
        ax.set_ylabel(r'Mass of $^{26}$Al Alive at time $t$ (M$_\odot$)')
        #ax.set_ylabel(r'dM$_{^{26}Al}$/dt (M$_\odot$/yr)')
        ax.set_yscale('log')
        ax.set_ylim(2.0e-8, 1.0e-3)
        #ax.set_xlim(1.0e5, 1.3e7)
        plt.savefig(bn+'_aluminum_vsTimeReverse.pdf')
        plt.close(fig)
        print("Finished aluminum_vsTimeReverse plot")


    if True:
        nsamplesDist = np.min( [nsamples, 10000] )
        sampleIndices = np.random.choice(nsamples, size=nsamplesDist, replace=False)
        fig,ax = plt.subplots()
        theTimes = np.linspace(0, 30.0e6, 40)
        quantiles = [ 0.025, 0.16, 0.5, 0.84, 0.975 ]
        for k in range(ndatasets):
            atTimeArr = np.zeros( (nsamplesDist, len(theTimes), 3) )
            atTimeArrAl = np.zeros( (nsamplesDist, len(theTimes), 3) )
            timeArrQuantiles = np.zeros( ( 5, len(theTimes), 3) )
            timeArrQuantilesAl = np.zeros( ( 5, len(theTimes), 3) )
            samplesThis = samples
            if k>0:
                samplesThis = more_datasets[k-1]
            for i in range(nsamplesDist):
                for j in range(len(theTimes)):
                    atTimeArrAl[i, j, :] =  theModel.atTime( samplesThis[sampleIndices[i],:], theTimes[j], axIn=ax, k=k)  
                    atTimeArr[i, j, :] =  theModel.atTimeFe( samplesThis[sampleIndices[i],:], theTimes[j], axIn=ax, k=k)  
            for j in range(len(theTimes)):
                timeArrQuantiles[:,j,0] = np.quantile(atTimeArr[:,j,0], quantiles)
                timeArrQuantiles[:,j,1] = np.quantile(atTimeArr[:,j,1], quantiles)
                timeArrQuantiles[:,j,2] = np.quantile(atTimeArr[:,j,2], quantiles)

                timeArrQuantilesAl[:,j,0] = np.quantile(atTimeArrAl[:,j,0], quantiles)
                timeArrQuantilesAl[:,j,1] = np.quantile(atTimeArrAl[:,j,1], quantiles)
                timeArrQuantilesAl[:,j,2] = np.quantile(atTimeArrAl[:,j,2], quantiles)
            ax.fill_between( theTimes/1.0e6, timeArrQuantilesAl[1,:,2], timeArrQuantilesAl[3,:,2], color='purple', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantilesAl[1,:,2], color='purple', alpha=1.0 )
            ax.plot( theTimes/1.0e6, timeArrQuantilesAl[3,:,2], color='purple', alpha=1.0 )
            ax.fill_between( theTimes/1.0e6, timeArrQuantiles[1,:,1], timeArrQuantiles[3,:,1], color='orange', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[1,:,2], color='orange', alpha=1.0 )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[3,:,2], color='orange', alpha=1.0 )
            #ax.fill_between( theTimes, timeArrQuantiles[1,:,2], timeArrQuantiles[3,:,2], color='k', alpha=0.3 )
            ax.plot( theTimes/1.0e6, timeArrQuantilesAl[2,:,2], c='purple', lw = 4, zorder=3, label=r'$^{26}$Al' )
            ax.plot( theTimes/1.0e6, timeArrQuantiles[2,:,2], c='orange', lw = 4, zorder=3, label=r'$^{60}$Fe' )
            #####ax.plot( theTimes/1.0e6, timeArrQuantilesAl[2,:,2] + timeArrQuantiles[2,:,2], c='k', lw = 5, zorder=3, label='Total' )
            
        ax.legend()
        ax.set_xlabel(r'Time Since Birth of Upper Sco, $t - t_\mathrm{SF}$, (Myr)')
        ax.set_ylabel(r'Mass of SLRs Alive at Time $t$ (M$_\odot$)')
        #ax.set_ylabel(r'dM$_{^{26}Al}$/dt (M$_\odot$/yr)')
        ax.set_yscale('log')
        ax.set_ylim(2.0e-7, 1.0e-3)
        #ax.set_xlim(1.0e5, 1.3e7)
        plt.savefig(bn+'_ironDistr_vsTime.pdf')
        plt.close(fig)




    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            if np.random.random()<fractionOfLines*0.07:
                theModel.vsTime( samplesThis[i,:], axIn=ax, reverse=True, k=k, weight=wtnorm[i] )
    ax.set_xlabel(r'Lookback Time, $t_\mathrm{obs} - t$, (Myr Ago)')
    ax.set_ylabel(r'Mass of $^{26}$Al Alive at time $t$ (M$_\odot$)')
    #ax.set_ylabel(r'dM$_{^{26}Al}$/dt (M$_\odot$/yr)')
    ax.set_yscale('log')
    ax.set_ylim(9.0e-7, 1.1e-3)
    ax.set_xlim( 10.2, -0.2)
    ax.errorbar( 0, 1.1e-4, yerr= 2.9e-5, ecolor='black', elinewidth=2, capsize=3, fmt='o', mfc='black' )
    plt.savefig(bn+'_aluminum_vsLookbackTime.pdf')
    plt.close(fig)
    print("Finished aluminum_vsLookbackTime plot")

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            if np.random.random()<fractionOfLines*0.3:
                theModel.vsTimeFeAl( samplesThis[i,:], axIn=ax, reverse=True, k=k, weight=wtnorm[i] )
    ax.set_xlabel(r'Lookback Time (Myr)')
    ax.set_ylabel(r'$^{60}$Fe / $^{26}$Al Alive ')
    #ax.set_ylabel(r'dM$_{^{26}Al}$/dt (M$_\odot$/yr)')
    ax.set_yscale('log')
    #ax.set_ylim(2.0e-6, 3.0e-3)
    ax.set_xlim(-.8, 10.2)
    plt.savefig(bn+'_ironAluminum_vsLookbackTime.pdf')
    plt.close(fig)


    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        tcores = np.zeros(nsamples)
        preTerms = np.zeros(nsamples)
        accumTerms = np.zeros(nsamples)
        preTermsFixedDist = np.zeros(nsamples)
        accumTermsFixedDist = np.zeros(nsamples)
        for i in range(nsamples):
            tcore = 10.0**(4.5+np.random.random()*2) # 10^4.5 - 10^6.5 yrs
            accumTerm, preTerm, accumTermFixedDist, preTermFixedDist = theModel.coreAccumulateFast( samples[i,:], tcore=tcore, chi=100.0, rcore=10000.0 ) 
            tcores[i] = tcore
            preTerms[i] = preTerm
            accumTerms[i] = accumTerm
            preTermsFixedDist[i] = preTermFixedDist
            accumTermsFixedDist[i] = accumTermFixedDist
        ntc = 20
        tcs = np.logspace( 4.5, 6.5, ntc+1)
        tcc = np.sqrt(tcs[1:]*tcs[:-1])
        perc = [1,5,16,50,84,95,99]
        summariesPreTerms = np.zeros( (ntc, len(perc)) )
        summariesPreTermsFixedDist = np.zeros( (ntc, len(perc)) )
        summariesAccumTerms = np.zeros( (ntc, len(perc)) )
        summariesAccumTermsFixedDist = np.zeros( (ntc, len(perc)) )
        inflate = np.power(2.0, np.random.randn(nsamples)*np.sqrt(5))  # factor of 2...
        for j in range(ntc):
            selec = np.logical_and( tcs[j] < tcores, tcores<tcs[j+1] )
            summariesPreTerms[j,:] = np.percentile( preTerms[selec]*inflate[selec] , perc )
            summariesPreTermsFixedDist[j,:] = np.percentile( preTermsFixedDist[selec]*inflate[selec], perc )
            summariesAccumTerms[j,:] = np.percentile( accumTerms[selec]*inflate[selec], perc )
            summariesAccumTermsFixedDist[j,:] = np.percentile( accumTermsFixedDist[selec]*inflate[selec], perc )
        alpha=0.1
        ax.plot( tcc, 100*summariesPreTerms[:, 3], c='salmon', lw=3, label='Pre-Enriched Term' ) # median
        ax.plot( tcc, 100*summariesAccumTerms[:, 3], c='salmon', lw=3, label='Accum. Term', ls='--' ) # median
        ax.plot( tcc, 100*summariesPreTermsFixedDist[:, 3]/10.0**3, c='blue', lw=3, label='Pre-Enriched Term 3 pc' ) # median
        ax.plot( tcc, 100*summariesAccumTermsFixedDist[:, 3]/10.0**3, c='blue', lw=3, label='Accum. Term 3 pc', ls='--' ) # median
# central 1 sigma
        ax.fill_between( tcc, 100*summariesPreTerms[:, 2], 100*summariesPreTerms[:, 4], color='salmon', alpha=alpha ) 
        ax.fill_between( tcc, 100*summariesAccumTerms[:, 2], 100*summariesAccumTerms[:, 4], color='salmon', alpha=alpha  ) 
        ax.fill_between( tcc, 100*summariesPreTermsFixedDist[:, 2]/10.0**3, 100*summariesPreTermsFixedDist[:, 4]/10.0**3, color='blue', alpha=alpha ) 
        ax.fill_between( tcc, 100*summariesAccumTermsFixedDist[:, 2]/10.0**3, 100*summariesAccumTermsFixedDist[:, 4]/10.0**3, color='blue', alpha=alpha ) 
    ax.plot( [10**4.5, 10**6.5], [3.71e-9]*2, c='maroon', zorder=20, ls=':', label='Canonical Solar System') # canonical value - reading off from Gritschneder, Lin+ 2012
    xerr = np.zeros((2,1))
    xerr[:,0] = [2.25e5, 4.5e5]
    ax.text( 2.4e5, 4.0e-13, 'Enoch+ (2008) Ages', color='black', zorder=40, clip_on=False)
    ax.errorbar( [4.5e5], [1.6e-13], xerr=xerr, fmt='o-', ecolor='gray', elinewidth=2, capsize=10, capthick=2, clip_on=False )
    ax.plot( [alHalfLife]*2, [1.0e-15, 1.0e-5], c='maroon', zorder=20, ls=':', label=r'$^{26}$Al Half Life', alpha=0.4)
    ax.set_ylim(1.0e-13, 2.0e-6)
    ax.set_xlim( np.min(tcc), np.max(tcc))
    ax.set_xlabel(r'$t_c$ (yr)')
    ax.set_ylabel(r'$M_{^{26}\mathrm{Al}, \mathrm{core}} (M_\odot)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.text( 10**4.6, 5.0e-9, 'Solar System value', color='maroon', zorder=40, bbox=dict(facecolor='ivory', alpha=0.8)) 
    ax.text( alHalfLife*0.8, 6.0e-12, r'$^{26}$Al Half Life ', color='maroon', zorder=40, rotation='vertical', bbox=dict(facecolor='ivory', alpha=0.8)) 
    legend_elements = [ matplotlib.patches.Patch( facecolor='salmon', alpha=0.5, label='Broad Distance Distribution'), matplotlib.patches.Patch( facecolor='blue', alpha=0.5, label=r'Fixed Distance of 10 pc'), matplotlib.lines.Line2D([0],[0], color='black', lw=3, label='Pre-Enrichment Term'), matplotlib.lines.Line2D( [0],[0], color='black', lw=3, ls='--', label='Accumulation Term')]
    leg = ax.legend( handles=legend_elements, loc='upper right', ncol=2, frameon= True, facecolor='ivory', framealpha=0.98)
    leg.set_zorder(100)

    #ax.legend()
    plt.savefig(bn+'_accumSimplified10pcInf.pdf', rasterized=True) 
    plt.close(fig)

    print("Writing ", bn+'_accumSimplified.pdf')

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        #alMassCoresAbInitio = np.zeros(nsamples)
        #tCores = np.zeros(nsamples)
        for i in range(nsamples):
            if np.random.random() < fractionOfLines*0.1:
                tCores = np.linspace(samplesThis[i,0]*-0.5, samplesThis[i,0]*0.5, 25) # +/- (1/2) avg age of cluster
                alMassCoresAbInitio = np.zeros(len(tCores))
                for j in range(len(tCores)):
# this constant factor is a fiducial value divided by 10^-4 solar masses.
# 10^-4 Msun * (10^4 AU / 3 pc)^3 * 100 * 2^(-tff/thalf)
                    alMassCoresAbInitio[j] = theModel.AlMassAtTime(samplesThis[i,:], tCores[j] + samplesThis[i,0] ) *9.89899e-5
                ax.plot(tCores/1.0e6, alMassCoresAbInitio, c='k', alpha=wtnorm[i]*0.7 )
        ax.plot( [np.min(tCores)/1.0e6, np.max(tCores)/1.0e6], [3.71e-9]*2, c='r')
        ax.plot( [0]*2, [1e-10, 1e-6], c='r')
        ax.set_xlim(np.min(tCores)/1.0e6, np.max(tCores)/1.0e6)
    ax.set_xlabel( r'$t_{core\ progenitor} - t_{obs}$ (Myr)')
    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 1.0e-7)
    ax.set_ylabel(r'$M_{^{26}\mathrm{Al}} (M_\odot)$')
    plt.savefig(bn+'_core_abInitio.pdf')
    plt.close(fig)

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        #ax.scatter( samplesThis[:,0]/1.0e6, samplesThis[:,4]/1.0e6, c='k', lw=0, s=30, alpha=0.1)
        ax.hist2d( samplesThis[:,0]/1.0e6, samplesThis[:,5]/1.0e6, lw=0, alpha=1.0, cmap=list_of_cmaps[k], bins=70, weights=wtnorm)
        ax.scatter( [np.mean(samplesThis[:,0])/1.0e6], [np.mean(samplesThis[:,5])/1.0e6], marker='+', c=list_of_colors[k], s=100 ) 
    ax.set_xlabel(r'Age (Myr)')
    ax.set_ylabel(r'Age Spread (Myr)')
    plt.savefig(bn+'_ageVsAgespread.pdf')
    plt.close(fig)


    x,y = np.mgrid[-8:-2:.01, -4:-2:.01]
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x; pos[:,:,1] = y
    fig,ax = plt.subplots()
    ax.contour(np.power(10.0,x),np.power(10.0,y), theModel.rv.pdf(pos), colors='k')
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        ax.hist2d( samplesThis[:,1], samplesThis[:,2],  lw=0, alpha=1.0, cmap=list_of_cmaps[k], bins=(10.0**np.linspace(-8,-2,70),10.0**np.linspace(-4,-2,70)), range=[[1.0e-8,1.0e-2],[1.0e-4,1.0e-2]], weights=wtnorm)
    #ax.scatter( samples[:,1], samples[:,2], c='k', lw=0, s=30, alpha=0.1)
    #ax.scatter( np.log10(samples[:,1]), np.log10(samples[:,2]), c='k', lw=0, s=30, alpha=0.1)
    ax.set_xlabel(r'$ A_{20} (M_\odot)$')
    ax.set_ylabel(r'$A_{120} (M_\odot)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_xlabel(r'$\log_{10} A_{20}$')
    #ax.set_ylabel(r'$\log_{10} A_{120}$')
    plt.savefig(bn+'_WRyields.pdf')
    plt.close(fig)



    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            if np.random.random() < fractionOfLines*2:
                #fade from blue to red as the pop ages
                r = (ages[i,k] - np.min(ages[:,k]))/(np.max(ages[:,k])-np.min(ages[:,k]))
                if r<0:
                    r=0
                if r>1:
                    r=1
                b = 1 - r
                g = 0.3
                color = ( r,g,b, 0.1*wtnorm[i])
                ax.plot( masses[i, alive[i,:,k],k], np.arange(np.sum(alive[i,:,k]))+1 , c=color)
            #ax.scatter( masses[i, alive[i,:]], np.arange(np.sum(alive[i,:]))+1 , c='k', alpha=0.5, s=30 )
    ax.scatter( [20.0, 17.2, 14.7, 14.6, 12.6, 12.2, 11.2], np.arange(7)+1, c='k', zorder=200) # zeta Oph + stars from table 7 of Pecaut et al 2012
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(8,101)
    ax.set_ylim(0.95,25)
    ax.set_xlabel(r'ZAMS Mass $M$ (M$_\odot$)')
    ax.set_ylabel(r'Number of Living Stars at $t_\mathrm{obs}$ with Mass $>M$')
    plt.savefig(bn+'_cdfs_alt_living.pdf')
    plt.close(fig)
    print("Finished CDF3 plot")

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        samplesThis = samples
        if k>0:
            samplesThis = more_datasets[k-1]
        for i in range(nsamples):
            if np.random.random() < fractionOfLines:
                #fade from blue to red as the pop ages
                r = (ages[i,k] - np.min(ages[:,k]))/(np.max(ages[:,k])-np.min(ages[:,k]))
                if r<0:
                    r=0
                if r>1:
                    r=1
                b = 1 - r
                g = 0.3
                color = ( r,g,b, 0.1*wtnorm[i])
                ax.plot( masses[i, :, k], np.arange(len(masses[i,:,k]))+1 , c=color )
                #ax.scatter( masses[i, :], np.arange(len(masses[i,:]))+1 , c='k', alpha=0.5, s=30 )
    ax.set_xscale('log')
    ax.set_xlim(8,150)
    ax.set_xlabel(r'$M$ (M$_\odot$)')
    ax.set_ylabel(r'Currently N($>M$)')
    plt.savefig(bn+'_cdfs_alt.pdf')
    plt.close(fig)
    print("Finished CDF4 plot")

    fig,ax = plt.subplots()
    mostMassiveLiving = np.zeros((nsamples,ndatasets))
    for k in range(ndatasets):
        for i in range(nsamples):
            mostMassiveLiving[i,k] = np.max( masses[i, alive[i,:,k], k] )
        ax.hist(mostMassiveLiving[:,k], bins=50, weights=wtnorm, histtype='step',density=True, color=list_of_colors[k])
    ax.set_xlabel(r'Most Massive Currently Living Star ($M_\odot$)')
    ax.set_ylabel(r'PDF of Most Massive Living Star ($M_\odot^{-1})$')
    plt.savefig(bn+'_mostmassiveliving.pdf')
    plt.close(fig)


    # let's do a "simple" PDF
    k=0
    xr = np.log10( np.array( [0.9e-7, 1.1e-3] ))
    fig,ax = plt.subplots()
    h,x,p = ax.hist( np.log10(np.sum(alMasses[:, : ,k],axis=1)), color='k', lw=3, histtype='step', bins=53, weights=wtnorm, range=xr )
    yl = [0, np.max(h)*1.1]
    l1, = ax.plot([0],[0],c='r',label='0 SNe')
    l2, = ax.plot([0],[0],c='r',ls='--',label='1 SN')
    l2a, = ax.plot([0],[0],c='r',ls='-.',label='2 SNe')
    l2b, = ax.plot([0],[0],c='r',ls=':',label='>2 SNe')
    l3, = ax.plot([0],[0],c='blue',label='young cluster')
    l4, = ax.plot([0],[0],c='blue',ls='--',label='old cluster')
    l5, = ax.plot([0],[0],c='orange',label='Most Massive < 19')
    l6, = ax.plot([0],[0],c='orange',ls='--',label='Most Massive > 19')

    # split by number of supernovae:
    nexploded = np.sum(exploded[:,:,k], axis=1) 
    zeroExplosions = nexploded==0
    oneExplosion = nexploded==1
    twoExplosions = nexploded==2
    gtTwoExplosions = nexploded>2
    ax.hist( np.log10(np.sum(alMasses[zeroExplosions, : ,k],axis=1)) , color='red', lw=1, histtype='step', bins=53, weights=wtnorm[zeroExplosions] , range=xr)
    ax.hist( np.log10(np.sum(alMasses[oneExplosion, : ,k],axis=1)) , color='red', ls='--', lw=1, histtype='step', bins=53, weights=wtnorm[oneExplosion] , range=xr)
    ax.hist( np.log10(np.sum(alMasses[twoExplosions, : ,k],axis=1)) , color='red', ls='-.', lw=1, histtype='step', bins=53, weights=wtnorm[twoExplosions] , range=xr)
    ax.hist( np.log10(np.sum(alMasses[gtTwoExplosions, : ,k],axis=1)) , color='red', ls=':', lw=1, histtype='step', bins=53, weights=wtnorm[gtTwoExplosions] , range=xr)

    # split by young vs. old
    young = ages[:,k] < 8.25e6
    ax.hist( np.log10(np.sum(alMasses[young, : ,k],axis=1)) , color='blue', lw=1, histtype='step', bins=53, label='young cluster', weights=wtnorm[young] , range=xr)
    ax.hist( np.log10(np.sum(alMasses[np.logical_not(young), : ,k],axis=1)) , color='blue', ls='--', lw=1, histtype='step', bins=53, label='old cluster', weights=wtnorm[np.logical_not(young)] , range=xr)

    # split by mass of most massive living star
    young = mostMassiveLiving[:,k] < 19
    ax.hist( np.log10(np.sum(alMasses[young, : ,k],axis=1)) , color='orange', lw=1, histtype='step', bins=53, label='Most Massive < 19', weights=wtnorm[young] , range=xr)
    ax.hist( np.log10(np.sum(alMasses[np.logical_not(young), : ,k],axis=1)) , color='orange', lw=1, ls='--', histtype='step', bins=53, label='Most Massive > 19', weights=wtnorm[np.logical_not(young)] , range=xr)

    ax.fill_between( np.log10(np.array([3.4e-4, 4.2e-4])), [0]*2, [yl[1]]*2, color='beige', alpha=0.5)
    ax.set_ylim(yl[0],yl[1])
    ax.set_xlim(xr[0],xr[1])
    ax.set_xlabel(r'$\log_{10}\ ^{26}$Al Mass (M$_\odot$)')
    ax.set_ylabel('Number of Samples')
    first_legend = plt.legend( handles=[l1,l2,l2a,l2b], loc='upper right', bbox_to_anchor=(0.98, 0.99) )
    ax.add_artist(first_legend)

    second_legend = plt.legend( handles=[l3,l4] , loc='upper right', bbox_to_anchor=(0.98, 0.7))
    ax.add_artist( second_legend)

    plt.legend( handles=[l5,l6] , loc='upper right', bbox_to_anchor=(0.98, 0.55))

    plt.savefig(bn+'_orionPDF.pdf')
    plt.close(fig)


    fig,ax = plt.subplots()
    for k in range(ndatasets):
        try:
            h,binedges,_ = ax.hist( fractions[:,k], bins=50, weights=wtnorm, histtype='step', density=True, color=list_of_colors[k])
        except:
            pdb.set_trace()
        print("TABULATED WR VS SN PROBABILITIES: ", np.sum(h[0:5]), np.sum(h[5:45]), np.sum(h[45:]))
        ax.text(binedges[0], h[0]*1.02, str(h[0]/np.sum(h)*100.0)[:2]+r'%')
        ax.text(binedges[-1], h[-1]*1.02, str(h[-1]/np.sum(h)*100.0)[:2]+r'%')
    ax.set_xlabel(r'Fraction of Living $^{26}$Al from WR')
    ax.set_ylabel(r'PDF of WR Fraction')
    plt.savefig(bn+'_WR_pdf.pdf')
    plt.close(fig)

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        ax.hist( mostRecentExplosion[:,k]/1.0e6, bins=50, weights=wtnorm, histtype='step', density=True, color=list_of_colors[k])
    ax.set_xlabel('Time Since Most Recent Explosion (Myr)')
    ax.set_ylabel('PDF of Time Since SN (Myr$^{-1}$)')
    plt.savefig(bn+'_timeSinceSN.pdf')
    plt.close(fig)



    def conditional(x, y, ranges, bins, scenarioBins, scenarioLabels, xl, yl, fn, stacked=True, weights=wtnorm, left=False):
        h,xe,ye = np.histogram2d( x, y, bins=bins, weights=weights, range=ranges )
        vsMassPosteriorDensity = np.sum(h, axis=1)
        vsMassPosteriorDensityNorm = vsMassPosteriorDensity/np.max(vsMassPosteriorDensity)

        scenarioLines = []
        for sci in range(len(scenarioBins)):
            scenarioLines.append( np.sum(h[:,scenarioBins[sci]], axis=1)/vsMassPosteriorDensity )
            
        vsMassMass = (xe[:-1] + xe[1:])/2.0
        fig,ax = plt.subplots()

        if not stacked:
            for j in range(len(vsMassPosteriorDensity)):
                for sci in range(len(scenarioBins)):
                    ax.scatter( vsMassMass[j], scenarioLines[sci][j], c=list_of_colors[sci], alpha=vsMassPosteriorDensityNorm[j], s=60)
            for sci in range(len(scenarioBins)):
                ax.plot( vsMassMass, scenarioLines[sci], c=list_of_colors[sci], alpha=0.5, label=scenarioLabels[sci])
        else:
# construct some interpolations of the things we need to plot!
            for sci in range(len(scenarioBins)):
                if sci==0:
                    yy= scenarioLines[0]*0
                    yLowInterp = interp1d( vsMassMass, yy, fill_value='extrapolate' )
                else:
                    yy += scenarioLines[sci-1]
                    yLowInterp = interp1d( vsMassMass, yy, fill_value='extrapolate' )
                yHighInterp = interp1d( vsMassMass, yy + scenarioLines[sci], fill_value='extrapolate' )

                alphainterp = interp1d( vsMassMass, vsMassPosteriorDensityNorm )


                xThis = np.linspace( vsMassMass[0], vsMassMass[-1], 300 )
                for j in range(len(xThis)-1):
                    xThisThis =[xThis[j], xThis[j] + 1.00*(xThis[j+1] - xThis[j]) ]
                    ax.fill_between( xThisThis , yLowInterp(xThisThis), yHighInterp(xThisThis), color=list_of_colors[sci], alpha=alphainterp(xThis[j])*0.7, lw=0, rasterized=True )
                    ax.plot( xThisThis , yHighInterp(xThisThis), color='k', lw=1 )
                    ax.plot( xThisThis , yLowInterp(xThisThis), color='k', lw=1 )

                if left:
                    xplot = xThis[ int(0.1*len(xThis)) ]
                    yplot = yLowInterp(xplot) + 0.5 * (yHighInterp(xplot)- yLowInterp(xplot))
                else:
                    xplot = xThis[ int(0.8*len(xThis)) ]
                    yplot = yLowInterp(xplot) + 0.5 * (yHighInterp(xplot)- yLowInterp(xplot))
                #while np.isnan(yplot):
                #    print("WARNING: regular location for label has failed. Trying random values")
                #    xplot = np.random.random()*(vsMassMass[-1] - vsMassMass[0])*0.8 + vsMassMass[0]
                #    yplot = yLowInterp(xplot) + 0.5 * (yHighInterp(xplot)- yLowInterp(xplot))

                ax.text( xplot, yplot, scenarioLabels[sci], fontsize=12, color='white', zorder=50, verticalalignment='center', horizontalalignment='center', bbox=dict(boxstyle='round', ec=list_of_colors[sci], fc=list_of_colors[sci]) )

                #ax.plot(xThis, xThis*0-1, c=list_of_colors[sci], label=scenarioLabels[sci])
                #for j in range(len(vsMassPosteriorDensity)):
                #    ax.scatter( vsMassMass[j], scenarioLines[sci][j], c=list_of_colors[sci], alpha=vsMassPosteriorDensityNorm[j], s=60)

        if xl==r'Time since most recent SN (Myr)':
            ax.plot([1.9]*2, [0,1], c='k', lw=2, ls='--', zorder=500)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(ranges[0][0], ranges[0][1])
        #ax.set_ylim(ranges[1][0], ranges[1][1])
        ax.set_ylim(0,1)
        #ax.legend()
        plt.savefig(bn+fn)
        plt.close(fig)

    conditional( mostMassiveLiving[:,k], fractions[:,k], [[9.9,32],[-0.01,1.01]], (50,100), [np.arange(0,20), np.arange(20,80), np.arange(80,100)], ['SN-Dominated', 'Mixture', 'WR-Dominated'], r'Most Massive Currently Living Star (M$_\odot$)', r'Share of Conditional Probability of Scenario | M', '_conditionalOnMostMassive.pdf')
    conditional( mostMassiveLiving[:,k], fractions[:,k], [[9.9,32],[-0.01,1.01]], (50,100), [np.arange(0,20), np.arange(20,80), np.arange(80,100)], ['SN-Dominated', 'Mixture',  'WR-Dominated', ], r'Most Massive Currently Living Star (M$_\odot$)', r'Share of Conditional Probability of Scenario | M', '_conditionalOnMostMassiveStacked.pdf',stacked=False)
    conditional( mostMassiveLiving[:,k], numberOfStarsToReach90percentAluminum[:,k], [[9.9,32], [0.5,5.5]], (50,5), [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])], ['1 Source', '2 Sources', '3 Sources', '4 Sources', '5 Sources'], r'Most Massive Currently Living Star (M$_\odot$)', r'Share of Conditional Probability of Scenario | M', '_numberOfSourcesConditionalOnMostMassive.pdf')
    conditional( mostMassiveLiving[:,k],  timeSinceDeathOfLargestAluminumSource[:,k]/1.0e6  , [[9.9,32], [-8,8]], (50,16), [np.array([0,1,2,3,4,5,6,7]), np.array([8]), np.array([9]), np.array([10]), np.array([11,12,13,14,15])], ['Still Alive', 'Last Myr', '1-2 Mya', '2-3 Mya', '3+ Mya'], r'Most Massive Currently Living Star (M$_\odot$)', r'Share of Conditional Probability of Scenario | M', '_timeSinceDeathOfLargestConditionalOnMostMassive.pdf')
    conditional( mostMassiveLiving[:,k], zetas[:,k], [[9.9,32], [-1,2]], (50,4), [np.array([0]), np.array([1]), np.array([2]), np.array([3])], ['Extremely weak', 'W18-like', 'N20-like', 'Extremely strong'], r'Most Massive Currently Living Star (M$_\odot$)', r'Share of Conditional Probability of Scenario | M', '_SNStrengthConditionalOnMostMassive.pdf') 




    conditional( numberOfStarsToReach90percentAluminum[:,k], fractions[:,k], [[0.5,5.5],[-0.01,1.01]], (5,100), [np.arange(0,20), np.arange(20,80), np.arange(80,100)], ['SN-Dominated', 'Mixture', 'WR-Dominated'], r'Number of Sources to Reach 90%', r'Share of Conditional Probability of Scenario | N', '_conditionalOnNumberOfSources.pdf')


    conditional( ages[:,k]/1.0e6, fractions[:,k], [[2,16],[-0.01,1.01]], (33,100), [np.arange(0,20), np.arange(20,80), np.arange(80,100)], ['SN-Dominated', 'Mixture', 'WR-Dominated' ], r'Age of Upper Sco, $t_\mathrm{obs}-t_\mathrm{SF}$, (Myr)', r'Share of Conditional Probability of Scenario | Age', '_conditionalOnAge.pdf', left=True)
    conditional( ages[:,k]/1.0e6, numberOfStarsToReach90percentAluminum[:,k], [[2,16], [0.5,10.5]], (33,10), [np.array([0]), np.array([1]), np.arange(2,10)], ['1 Source', '2 Sources', '3+ Sources'], r'Age of Upper Sco, $t_\mathrm{obs}-t_\mathrm{SF}$,  (Myr)', r'Share of Conditional Probability of Scenario | Age', '_numberOfSourcesConditionalOnAge.pdf')
    conditional( ages[:,k]/1.0e6,  timeSinceDeathOfLargestAluminumSource[:,k]/1.0e6  , [[2,16], [-8,8]], (33,16), [np.array([0,1,2,3,4,5,6,7]), np.array([8]), np.array([9]), np.array([10,11,12,13,14,15])], ['Still Alive', 'Last Myr', '1-2 Mya', '2+ Mya'], r'Age of Upper Sco, $t_\mathrm{obs}-t_\mathrm{SF}$, (Myr)', r'Share of Conditional Probability of Scenario | Age', '_timeSinceDeathOfLargestConditionalOnAge.pdf')
    conditional( ages[:,k]/1.0e6, zetas[:,k], [[2,16], [-1,2]], (32,4), [np.array([0]), np.array([1]), np.array([2]), np.array([3])], ['Extremely weak', 'W18-like', 'N20-like', 'Extremely strong'], r'Age of Upper Sco, $t_\mathrm{obs}-t_\mathrm{SF}$, (Myr)', r'Share of Conditional Probability of Scenario | Age', '_SNStrengthConditionalOnAge.pdf') 


    conditional( zetas[:,k], fractions[:,k], [[-0.5,1.5],[-0.01,1.01]], (12,100), [np.arange(0,20), np.arange(20,80), np.arange(80,100)], ['SN-Dominated', 'Mixture', 'WR-Dominated'], r'$\zeta$', r'Share of Conditional Probability of Scenario | $\zeta$', '_conditionalOnZeta.pdf')
    conditional( zetas[:,k], numberOfStarsToReach90percentAluminum[:,k], [[-0.5,1.5], [0.5,5.5]], (12,5), [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])], ['1 Source', '2 Sources', '3 Sources', '4 Sources', '5 Sources'], r'$\zeta$', r'Share of Conditional Probability of Scenario | $\zeta$', '_numberOfSourcesConditionalOnZeta.pdf')
    conditional( zetas[:,k],  timeSinceDeathOfLargestAluminumSource[:,k]/1.0e6  , [[-0.5,1.5], [-8,8]], (12,16), [np.array([0,1,2,3,4,5,6,7]), np.array([8]), np.array([9]), np.array([10]), np.array([11,12,13,14,15])], ['Still Alive', 'Last Myr', '1-2 Mya', '2-3 Mya', '3+ Mya'], r'$\zeta$', r'Share of Conditional Probability of Scenario | $\zeta$', '_timeSinceDeathOfLargestConditionalOnZeta.pdf')
    conditional( zetas[:,k], zetas[:,k], [[-1,2], [-0.5,1.5]], (12,4), [np.array([0]), np.array([1]), np.array([2]), np.array([3])], ['Extremely weak', 'W18-like', 'N20-like', 'Extremely strong'], r'$\zeta$', r'Share of Conditional Probability of Scenario | $\zeta$', '_SNStrengthConditionalOnZeta.pdf') 

    conditional( fSNs[:,k], fractions[:,k], [[0,3],[-0.01,1.01]], (12,100), [np.arange(0,20), np.arange(20,80), np.arange(80,100)], ['SN-Dominated', 'Mixture', 'WR-Dominated'], r'$f_\mathrm{SN}$', r'Share of Conditional Probability of Scenario | $f_\mathrm{SN}$', '_conditionalOnfsn.pdf')
    conditional( fSNs[:,k], numberOfStarsToReach90percentAluminum[:,k], [[0,5], [0.5,5.5]], (32,5), [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])], [ '1 Source', '2 Sources', '3 Sources', '4 Sources', '5 Sources'], r'$f_\mathrm{SN}$', r'Share of Conditional Probability of Scenario | $f_\mathrm{SN}$', '_numberOfSourcesConditionalOnfsn.pdf')
    conditional( fSNs[:,k],  timeSinceDeathOfLargestAluminumSource[:,k]/1.0e6  , [[0,5], [-8,8]], (32,16), [np.array([0,1,2,3,4,5,6,7]), np.array([8]), np.array([9]), np.array([10]), np.array([11,12,13,14,15])], ['Still Alive', 'Last Myr', '1-2 Mya', '2-3 Mya', '3+ Mya'], r'$f_\mathrm{SN}$', r'Share of Conditional Probability of Scenario | $f_\mathrm{SN}$', '_timeSinceDeathOfLargestConditionalOnfsn.pdf')
    conditional( fSNs[:,k], zetas[:,k], [[0,5], [-1,2]], (32,4), [np.array([0]), np.array([1]), np.array([2]), np.array([3])], ['Extremely weak', 'W18-like', 'N20-like', 'Extremely strong'], r'$f_\mathrm{SN}$', r'Share of Conditional Probability of Scenario | $f_\mathrm{SN}$', '_SNStrengthConditionalOnfsn.pdf') 

    conditional( np.log10(at20s[:,k]), fractions[:,k], [[-10,-2],[-0.01,1.01]], (15,100), [np.arange(0,20), np.arange(20,80), np.arange(80,100)], ['SN-Dominated', 'Mixture', 'WR-Dominated'], r'$A_{20}$', r'Share of Conditional Probability of Scenario | $A_{20}$', '_conditionalOnA20.pdf')
    conditional( np.log10(at20s[:,k]), numberOfStarsToReach90percentAluminum[:,k], [[-10,-2], [0.5,5.5]], (15,5), [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])], ['1 Source', '2 Sources', '3 Sources', '4 Sources', '5 Sources'], r'$A_{20}$', r'Share of Conditional Probability of Scenario | $A_{20}$', '_numberOfSourcesConditionalOnA20.pdf')
    conditional( np.log10(at20s[:,k]),  timeSinceDeathOfLargestAluminumSource[:,k]/1.0e6  , [[-10,-2], [-8,8]], (15,16), [np.array([0,1,2,3,4,5,6,7]), np.array([8]), np.array([9]), np.array([10]), np.array([11,12,13,14,15])], ['Still Alive', 'Last Myr', '1-2 Mya', '2-3 Mya', '3+ Mya'], r'$A_{20}$', r'Share of Conditional Probability of Scenario | $A_{20}$', '_timeSinceDeathOfLargestConditionalOnA20.pdf')
    conditional( np.log10(at20s[:,k]), zetas[:,k], [[-10,-2], [-1,2]], (15,4), [np.array([0]), np.array([1]), np.array([2]), np.array([3])], ['Extremely weak', 'W18-like', 'N20-like', 'Extremely strong'], r'$A_{20}$', r'Share of Conditional Probability of Scenario | $A_{20}$', '_SNStrengthConditionalOnA20.pdf') 

    conditional( np.log10(at120s[:,k]), fractions[:,k], [[-4,-2.5],[-0.01,1.01]], (15,100), [np.arange(0,20), np.arange(20,80), np.arange(80,100)], ['SN-Dominated', 'Mixture', 'WR-Dominated'], r'$A_{120}$', r'Share of Conditional Probability of Scenario | $A_{120}$', '_conditionalOnA120.pdf')
    conditional( np.log10(at120s[:,k]), numberOfStarsToReach90percentAluminum[:,k], [[-4,-2.5], [0.5,5.5]], (15,5), [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])], [ '1 Source', '2 Sources', '3 Sources', '4 Sources', '5 Sources'], r'$A_{120}$', r'Share of Conditional Probability of Scenario | $A_{120}$', '_numberOfSourcesConditionalOnA120.pdf')
    conditional( np.log10(at120s[:,k]),  timeSinceDeathOfLargestAluminumSource[:,k]/1.0e6  , [[-4,-2.5], [-8,8]], (15,16), [np.array([0,1,2,3,4,5,6,7]), np.array([8]), np.array([9]), np.array([10]), np.array([11,12,13,14,15])], ['Still Alive', 'Last Myr', '1-2 Mya', '2-3 Mya', '3+ Mya'], r'$A_{120}$', r'Share of Conditional Probability of Scenario | $A_{120}$', '_timeSinceDeathOfLargestConditionalOnA20.pdf')
    conditional( np.log10(at120s[:,k]), zetas[:,k], [[-4,-2.5], [-1,2]], (15,4), [np.array([0]), np.array([1]), np.array([2]), np.array([3])], ['Extremely weak', 'W18-like', 'N20-like', 'Extremely strong'], r'$A_{20}$', r'Share of Conditional Probability of Scenario | $A_{120}$', '_SNStrengthConditionalOnA120.pdf') 



    fig,ax = plt.subplots()
    for k in range(ndatasets):
        ax.hist2d( mostRecentExplosion[:,k]/1.0e6, mostMassiveLiving[:,k], weights=wtnorm, bins=50, density=True, range=[[0,5.1],[5,30]], cmap=list_of_cmaps[k] )
    ax.scatter( [1.0], [20.0], c='k', marker='P', zorder=10)
    ax.set_xlabel('Time Since Most Recent Explosion (Myr)')
    ax.set_ylabel(r'Most Massive Currently Living Star ($M_\odot$)')
    plt.savefig(bn+'_wheresZetaOph.pdf')
    plt.close(fig)

    fig,ax = plt.subplots()
    for k in range(ndatasets):
        totalAlMasses = np.sum(alMasses[:,:,k], axis=1)
        ax.hist2d( totalAlMasses*fractions[:,k] , totalAlMasses*(1.0-fractions[:,k]), bins=50, weights=wtnorm, range=[[0,1.5e-4],[0,1.5e-4]], cmap=list_of_cmaps[k] )

    aluminumMassMeasured, aluminumMassUncertainty = 1.1e-4, 1.1e-4 * np.sqrt(1.0**2+1.2**2)/6.1 # from Diehl 2010
    xs = np.linspace(0, 3.0e-4, 100)
    ysMed = aluminumMassMeasured - xs
    ysPlus = aluminumMassMeasured + aluminumMassUncertainty - xs 
    ysMinus = aluminumMassMeasured - aluminumMassUncertainty - xs 
    valid = ysMed>0
    ax.plot(xs[valid], ysMed[valid], c='k', lw=3, zorder=10 )
    valid = ysPlus>0
    ax.plot(xs[valid], ysPlus[valid], c='k', lw=2, zorder=10 )
    valid = ysMinus>0
    ax.plot(xs[valid], ysMinus[valid], c='k', lw=2, zorder=10 )
    ax.set_xlabel(r'$ ^{26}$Al Mass from WR (M$_\odot$)')
    ax.set_ylabel(r'$ ^{26}$Al Mass from SN (M$_\odot$)')
    ax.set_xlim(0,1.5e-4)
    ax.set_ylim(0,1.5e-4)
    plt.savefig(bn+'_AlContributions2dPDF.pdf')
    plt.close(fig)




def run_dynesty( rotModels=False, agespread=True, mpi=True):
    if mpi:
        pool = schwimmbad.MPIPool()

        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool = None

    stars = read_stellar_models()
    noRot = setOfModels( [star for star in stars if star.V<0.1] )
    withRot = setOfModels( [star for star in stars if star.V>0.1])

    prior_mean_age, prior_age_uncertainty, prior_max_agespread, prior_minimum_mass, N = 10.0e6, 3.0e6, 3.0e6, 2.0, 76 # what we used for ophiuchus


    ndim = 7 + 3*N

    if not rotModels:
        theModel = alModel(N,noRot,agespread, prior_mean_age, prior_age_uncertainty, prior_max_agespread, prior_minimum_mass)
    else:
        theModel = alModel(N,withRot,agespread , prior_mean_age, prior_age_uncertainty, prior_max_agespread, prior_minimum_mass )

    dsampler = dynesty.DynamicNestedSampler( lnlik, ptransform, ndim, pool=pool, ptform_args=[theModel,], logl_args=[theModel,], first_update={'min_eff':0.01}, bootstrap=20, bound='single', sample='unif' )

    print("Started sampling! ", time.localtime())
    # These are far from ideal parameters.
    dsampler.run_nested(maxcall_init=6000000, maxcall=6000000, nlive_init=1000, use_stop=False, wt_kwargs={'pfrac':0.95}, print_progress=True, )
    print("Finished sampling! Time to save the results", time.localtime())
    dresults = dsampler.results

    with open('dsampler_results_76.pickle', 'wb') as f:
        pickle.dump( dresults, f)
    
def analyze_dynesty():
    plotsBn = 'dynesty_ophiuchus_76'

    # Do the same set of posterior and prior plots we did before
    stars = read_stellar_models()
    noRot = setOfModels( [star for star in stars if star.V<0.1] )
    withRot = setOfModels( [star for star in stars if star.V>0.1])

    prior_mean_age, prior_age_uncertainty, prior_max_agespread, prior_minimum_mass, N = 10.0e6, 3.0e6, 3.0e6, 2.0, 76 # what we used for ophiuchus


    rotModels = False
    agespread = True

    if not rotModels:
        theModel = alModel(N,noRot,agespread, prior_mean_age, prior_age_uncertainty, prior_max_agespread, prior_minimum_mass)
    else:
        theModel = alModel(N,withRot,agespread , prior_mean_age, prior_age_uncertainty, prior_max_agespread, prior_minimum_mass )



    #theModel = alModel(N,noRot,True)
    ndim =7+ 3*N

    inAHurry = False

    with open('dsampler_results_76.pickle', 'rb') as f:
        dresults = pickle.load(f)




    postSamples, postWeights = dresults.samples, np.exp(dresults.logwt - dresults.logz[-1]) 
    postSamples = dyfunc.resample_equal(postSamples,postWeights)
    postWeights = np.ones(len(postWeights))


# Do a test to make sure the plots work as expected!
    npostsamples = np.shape(postSamples)[0]
    if False:
        print("Calling analyze sample with ", plotsBn+"_proto")
        protosamp = np.zeros(( 400, ndim) )
        for i in range(np.shape(protosamp)[0]):
            protosamp[i, :] = theModel.sample_from_prior() # whatever... this is just to test out the plots!
        analyze_sample(protosamp, plotsBn+"_proto", theModel)

        print("Finished prototype plots")

    if not inAHurry:
        print("Starting to generate prior sample")
        priorsamp = np.zeros((70000, ndim)) # Draw 10k samples from the prior 
        for i in range(np.shape(priorsamp)[0]):
            priorsamp[i, :] = theModel.sample_from_prior()

        print("Finished generating prior sample")

    print("Calling analyze sample with ", plotsBn+"_post")
    analyze_sample( postSamples, plotsBn+'_post', theModel, weights=postWeights )
    if not inAHurry:
        print("Calling analyze sample with ", plotsBn+"_prior")
        analyze_sample( priorsamp, plotsBn+'_prior', theModel )



def getPowerlaw( at20, at120):
    # Assume y = a *x^b
    # => at20 = a*20^b
    # and at120=a*120^b
    # => at120/at20 = 6^b => log10(at120/at20) = b * log10(6)
    # => a = at120/120^b
    b = np.log10(at120/at20) / np.log10(6.0)
    a = at120/120**b
    #return a*np.power(xs,b)
    return a,b

def getNormalization():
    yieldsAt20And120 = np.zeros(( 6,2))
    yieldsAandB = np.zeros(( 6,2))
    yieldsAt20And120[0, :] = (1.0e-7, 5.0e-4) # Langer+ 1995
    yieldsAt20And120[1, :] = (2.5e-5, 1.3e-3) # Palacios+ 2005 - Rot
    yieldsAt20And120[2, :] = (2.0e-8, 6.0e-4) # Palacios+ 2005 - NoRot
    yieldsAt20And120[3, :] = (4.3e-8, 2.8e-4) # Limongi & Chieffi 2006
    yieldsAt20And120[4, :] = (9.0e-7, 3.8e-4) # Ekstrom+ 2012
    yieldsAt20And120[5, :] = (6.1e-6, 7.5e-4) # Ekstrom+ 2012 - rot

    for k in range(6):
        yieldsAandB[k,:] = getPowerlaw( yieldsAt20And120[k,0], yieldsAt20And120[k,1] )

    xs = np.linspace(20,120,50)

    print('a,b:',yieldsAandB)
    print('at20 and 120:', yieldsAt20And120)
    fig,ax = plt.subplots(nrows=3, figsize=(8,12))
    for k in range(6):
        ax[0].plot( xs, yieldsAandB[k,0] * np.power( xs, yieldsAandB[k,1] ) )
    ax[1].scatter(  yieldsAandB[:,0],   yieldsAandB[:,1] ) 
    ax[2].scatter(  yieldsAt20And120[:,0], yieldsAt20And120[:,1] )

    ax[0].set_xlabel(r'M')
    ax[0].set_ylabel(r'$^{26}$Al')
    ax[0].set_yscale('log')
    ax[0].set_xlim(20,140)
    ax[0].set_ylim(1.0e-8,1.0e-2)

    ax[1].set_xlabel('a')
    ax[1].set_xscale('log')
    ax[1].set_ylabel('b')
    ax[1].set_xlim(1.0e-14, 1.0e-7)
    ax[1].set_ylim(2,6)

    ax[2].set_xlabel('at20')
    ax[2].set_ylabel('at120')
    ax[2].set_xlim(1.0e-8, 1.0e-4)
    ax[2].set_ylim(1.0e-4,2.0e-3)
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')

    nx = 100

    x,y = np.mgrid[-8:-4:.01, -4:-2.8:.01]
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x; pos[:,:,1] = y
    sigmax = np.sqrt(np.log10(yieldsAt20And120[:,0]))
    sigmay = np.sqrt(np.log10(yieldsAt20And120[:,1]))
    mux = np.mean(np.log10(yieldsAt20And120[:,0]))
    muy = np.mean(np.log10(yieldsAt20And120[:,1]))
    cov = np.cov( np.log10(yieldsAt20And120).T )
    print(np.shape(cov))
    rv = multivariate_normal( [mux,muy], cov) #[[sigmax*sigmax, cov],[cov, sigmay*sigmay]] )
    ax[2].contour(np.power(10.0,x),np.power(10.0,y), rv.pdf(pos))


    plt.tight_layout()
    plt.savefig('totalprod.png')
    plt.close(fig)

    return mux,muy,cov


def quickplots():
    fig,ax = plt.subplots()
    stars = read_stellar_models()
    cmap = matplotlib.cm.viridis
    for star in stars:
        if star.mass > 1.0 and star.mass < 100.0:
            #cval = np.clip( np.log10(star.mass/4.0), 0, 1)
            cval = matplotlib.colors.Normalize(0.5,2)(np.log10(star.mass))
            c = cmap(cval)
            ls  = '-'
            if star.V>0:
                ls='--'
            ax.plot( 1.0 - star.lifetime_fraction(), star.aluminum26_vs_time() * (1.0-star.lifetime_fraction()), c=c, ls=ls, alpha=1.0 )
    #ax.set_xlim(0,10.0e6)
    sc = ax.scatter( [0],[0], c=[0], vmin=0.5,vmax=2, cmap=cmap)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\log_{10} M/M_\odot$')
    ax.set_xscale('log')
    ax.set_xlim(1.0e-4,1)
    ax.set_ylim(1.0e-15,2.0e-9)
    ax.set_yscale('log')
    ax.set_xlabel(r'1 - Age/Lifetime')
    #ax.set_ylabel(r'$^{26}$Al/time (M$_\odot$/yr)')
    ax.set_ylabel(r'$^{26}$Al/ log time (M$_\odot$/log yr)')
    plt.savefig('history_aluminum.png')
    plt.close(fig)


    noRot = setOfModels( [star for star in stars if star.V<0.1] )
    withRot = setOfModels( [star for star in stars if star.V>0.1])

    y,x = np.mgrid[ slice(0,2,.01),
            slice(-5,0,.01)]
    z = np.zeros( np.shape(x))
    z = noRot.estimate_aluminum_prod_rate( x, y ) # log10 (1-t/lifetime),  log10(M)
    fig,ax = plt.subplots()
    pc = ax.pcolor( x, y, z, vmin=-15, vmax=-8, cmap=cmap)
    plt.colorbar(pc)
    plt.savefig('interp_al_pcolor_noRot.png')
    plt.close(fig)
    

    z = withRot.estimate_aluminum_prod_rate( x, y ) # log10 (1-t/lifetime),  log10(M)
    fig,ax = plt.subplots()
    pc = ax.pcolor( x, y, z, vmin=-15, vmax=-8, cmap=cmap)
    plt.colorbar(pc)
    plt.savefig('interp_al_pcolor_withRot.png')
    plt.close(fig)




if __name__ =='__main__':
    quickplots()

    #run_dynesty()
    analyze_dynesty()




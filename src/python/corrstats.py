"""
Functions for calculating the statistical significant differences between two dependent or independent correlation
coefficients.
The Fisher and Steiger method is adopted from the R package http://personality-project.org/r/html/paired.r.html
and is described in detail in the book 'Statistical Methods for Psychology'
The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
Credit goes to the authors of above mentioned packages!

Author: Philipp Singer (www.philippsinger.info)
"""

from __future__ import division

__author__ = 'psinger'

import numpy as np
from scipy.stats import t, norm
from math import atanh, pow
from numpy import tanh

def rz_ci(r, n, conf_level = 0.95):
    zr_se = pow(1/(n - 3), .5)
    moe = norm.ppf(1 - (1 - conf_level)/float(2)) * zr_se
    zu = atanh(r) + moe
    zl = atanh(r) - moe
    return tanh((zl, zu))

def rho_rxy_rxz(rxy, rxz, ryz):
    num = (ryz-1/2.*rxy*rxz)*(1-pow(rxy,2)-pow(rxz,2)-pow(ryz,2))+pow(ryz,3)
    den = (1 - pow(rxy,2)) * (1 - pow(rxz,2))
    return num/float(den)

def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, method='steiger'):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    if method == 'steiger':
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz)/2
        cube = (1 - yz) * (1 - yz) * (1 - yz)

        t2 = d * np.sqrt((n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + av * av * cube)))
        p = 1 - t.cdf(abs(t2), n - 3)

        if twotailed:
            p *= 2

        return t2, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(xz, n, conf_level=conf_level)[0]
        U2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = xy - xz - pow((pow((xy - L1), 2) + pow((U2 - xz), 2) - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)), 0.5)
        upper = xy - xz + pow((pow((U1 - xy), 2) + pow((xz - L2), 2) - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')

def independent_corr(xy, ab, n, n2 = None, twotailed=True, conf_level=0.95, method='fisher'):
    """
    Calculates the statistic significance between two independent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between a and b
    @param n: number of elements in xy
    @param n2: number of elements in ab (if distinct from n)
    @param twotailed: whether to calculate a one or two tailed test, only works for 'fisher' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'fisher' or 'zou'
    @return: z and p-val
    """

    if method == 'fisher':
        xy_z = 0.5 * np.log((1 + xy)/(1 - xy))
        ab_z = 0.5 * np.log((1 + ab)/(1 - ab))
        if n2 is None:
            n2 = n

        se_diff_r = np.sqrt(1/(n - 3) + 1/(n2 - 3))
        diff = xy_z - ab_z
        z = abs(diff / se_diff_r)
        p = (1 - norm.cdf(z))
        if twotailed:
            p *= 2

        return z, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(ab, n2, conf_level=conf_level)[0]
        U2 = rz_ci(ab, n2, conf_level=conf_level)[1]
        lower = xy - ab - pow((pow((xy - L1), 2) + pow((U2 - ab), 2)), 0.5)
        upper = xy - ab + pow((pow((U1 - xy), 2) + pow((ab - L2), 2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')

print "old"
print "msa","w2v","mc",dependent_corr(.82,.93,.84, 30, twotailed=False,method='steiger')
print "msa","w2v","rg",dependent_corr(.84,.88,.79, 65, twotailed=False,method='steiger')
print "msa","w2v","wss",dependent_corr(.76,.72,.72, 203, twotailed=False,method='steiger')
print "msa","w2v","wsr",dependent_corr(.64,.68,.67, 252, twotailed=False,method='steiger')
print "msa","w2v","ws",dependent_corr(.71,.69,.68, 352, twotailed=False,method='steiger')
print "msa","w2v","men",dependent_corr(.79,.71,.76, 1000, twotailed=False,method='steiger')
print "msa","adw","mc",dependent_corr(.90,.93,.85, 30, twotailed=False,method='steiger')
print "msa","adw","rg",dependent_corr(.92,.88,.79, 65, twotailed=False,method='steiger')
print "msa","adw","wss",dependent_corr(.75,.72,.62, 203, twotailed=False,method='steiger')
print "msa","nasari","mc",dependent_corr(.80,.93,.81, 30, twotailed=False,method='steiger')
print "msa","nasari","rg",dependent_corr(.78,.88,.76, 65, twotailed=False,method='steiger')
print "msa","nasari","wss",dependent_corr(.73,.72,.64, 203, twotailed=False,method='steiger')
print "adw","w2v","mc",dependent_corr(.90,.82,.80, 30, twotailed=False,method='steiger')
print "adw","w2v","rg",dependent_corr(.92,.84,.81, 65, twotailed=False,method='steiger')
print "adw","w2v","wss",dependent_corr(.75,.76,.68, 203, twotailed=False,method='steiger')
print "adw","nasari","mc",dependent_corr(.90,.80,.82, 30, twotailed=False,method='steiger')
print "adw","nasari","rg",dependent_corr(.92,.78,.80, 65, twotailed=False,method='steiger')
print "adw","nasari","wss",dependent_corr(.75,.73,.76, 203, twotailed=False,method='steiger')
print "nasari","w2v","mc",dependent_corr(.82,.80,.75, 30, twotailed=False,method='steiger')
print "nasari","w2v","rg",dependent_corr(.84,.78,.71, 65, twotailed=False,method='steiger')
print "nasari","w2v","wss",dependent_corr(.76,.73,.66, 203, twotailed=False,method='steiger')

print "new"
print "msa","w2v","mc",dependent_corr(.82,.89,.75, 30, twotailed=False,method='steiger')
print "msa","w2v","rg",dependent_corr(.84,.83,.74, 65, twotailed=False,method='steiger')

print "msa","adw","mc",dependent_corr(.90,.89,.83, 30, twotailed=False,method='steiger')
print "msa","adw","rg",dependent_corr(.92,.83,.76, 65, twotailed=False,method='steiger')

print "msa","nasari","mc",dependent_corr(.80,.89,.75, 30, twotailed=False,method='steiger')
print "msa","nasari","rg",dependent_corr(.78,.83,.77, 65, twotailed=False,method='steiger')

print "adw","w2v","mc",dependent_corr(.90,.82,.80, 30, twotailed=False,method='steiger')
print "adw","nasari","mc",dependent_corr(.90,.80,.82, 30, twotailed=False,method='steiger')
print "adw","w2v","rg",dependent_corr(.92,.84,.81, 65, twotailed=False,method='steiger')
print "adw","nasari","rg",dependent_corr(.92,.78,.80, 65, twotailed=False,method='steiger')

print "nasari","w2v","mc",dependent_corr(.82,.80,.75, 30, twotailed=False,method='steiger')
print "nasari","w2v","rg",dependent_corr(.84,.78,.71, 65, twotailed=False,method='steiger')

print "2207"
print "msa","w2v","mc",dependent_corr(.82,.87,.84, 30, twotailed=False,method='steiger')
print "msa","w2v","rg",dependent_corr(.84,.86,.78, 65, twotailed=False,method='steiger')
print "msa","w2v","wss",dependent_corr(.76,.77,.79, 203, twotailed=False,method='steiger')
print "msa","w2v","wsr",dependent_corr(.64,.71,.70, 252, twotailed=False,method='steiger')
print "msa","w2v","ws",dependent_corr(.71,.73,.72, 352, twotailed=False,method='steiger')
print "msa","w2v","men",dependent_corr(.79,.75,.78, 1000, twotailed=False,method='steiger')
print "msa","adw","mc",dependent_corr(.90,.87,.78, 30, twotailed=False,method='steiger')
print "msa","adw","rg",dependent_corr(.92,.86,.78, 65, twotailed=False,method='steiger')
print "msa","adw","wss",dependent_corr(.75,.77,.67, 203, twotailed=False,method='steiger')
print "msa","nasari","mc",dependent_corr(.80,.87,.73, 30, twotailed=False,method='steiger')
print "msa","nasari","rg",dependent_corr(.78,.86,.77, 65, twotailed=False,method='steiger')
print "msa","nasari","wss",dependent_corr(.73,.77,.70, 203, twotailed=False,method='steiger')
print "adw","w2v","mc",dependent_corr(.90,.82,.80, 30, twotailed=False,method='steiger')
print "adw","w2v","rg",dependent_corr(.92,.84,.81, 65, twotailed=False,method='steiger')
print "adw","w2v","wss",dependent_corr(.75,.76,.68, 203, twotailed=False,method='steiger')
print "adw","nasari","mc",dependent_corr(.90,.80,.82, 30, twotailed=False,method='steiger')
print "adw","nasari","rg",dependent_corr(.92,.78,.80, 65, twotailed=False,method='steiger')
print "adw","nasari","wss",dependent_corr(.75,.73,.76, 203, twotailed=False,method='steiger')
print "nasari","w2v","mc",dependent_corr(.82,.80,.75, 30, twotailed=False,method='steiger')
print "nasari","w2v","rg",dependent_corr(.84,.78,.71, 65, twotailed=False,method='steiger')
print "nasari","w2v","wss",dependent_corr(.76,.73,.66, 203, twotailed=False,method='steiger')


#print independent_corr(0.5 , 0.6, 103, 103, method='fisher')

#print dependent_corr(.396, .179, .088, 200, method='zou')
#print independent_corr(.560, .588, 100, 353, method='zou')

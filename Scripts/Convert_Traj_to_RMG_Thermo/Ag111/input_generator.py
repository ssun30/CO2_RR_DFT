import numpy as np
import scipy
import pylab
import matplotlib
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter, MaxNLocator, LogLocator
from ase.io.trajectory import Trajectory
import os
import pandas as pd
import re
from ase.visualize import view

font_size = 16
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = 20

# declare a class for molecules
class Molecule:

        def __init__(self):
                #start by defining some physical constants
                self.R = 8.3144621E-3 #ideal Gas constant in kJ/mol-K
                self.kB = 1.38065e-23 #Boltzmann constant in J/K
                self.h = 6.62607e-34 #Planck constant in J*s
                self.c = 2.99792458e8 #speed of light in m/s
                self.amu = 1.6605e-27 #atomic mass unit in kg
                self.Avogadro = 6.0221E23 #mole^-1
                self.GHz_to_Hz = 1.0E9 #convert rotational constants from GHz to Hz
                self.invcm_to_invm = 1.0E2 #convert cm^-1 to m^-1, for frequencies
                self.P_ref = 1.0E5 #reference pressure, 1 bar = 1E5 Pascal
                self.hartree_to_kcalpermole = 627.5095 #convert hartree/molecule to kcal/mol
                self.hartree_to_kJpermole = 2627.25677 #convert hartree/molecule to kJ/mol
                self.eV_to_kJpermole = 96.485 #convert eV/molecule to kJ/mol
                self.T_switch = 1000.0 #K, switching temperature in NASA polynomial. Default. can overwrite.
                self.site_occupation_number = 1 #number of sites occupied by adsorbate
                self.unit_cell_area = 62.10456e-20/9.0 #m2 - using surface area per binding site (nine binding sites per cell)
                self.cutoff_frequency = 100.0 #cm^-1
                self.twoD_gas = False

# HERE BEGINS THE LONG LIST OF SUBROUTINES
#-------------------------------------------------------------------------
# subroutine for the translational mode
def get_translation_thermo(molecule,temperature):
    # unpack the constants (not essential, but makes it easier to read)
    R = molecule.R
    kB = molecule.kB
    h = molecule.h
    amu = molecule.amu
    P_ref = molecule.P_ref
    m = molecule.adsorbate_mass
    pi = np.pi
    area = molecule.unit_cell_area
    sites = molecule.site_occupation_number

    #initialize the arrays for the partition function, entropy, enthalpy,
    #and heat capacity.
    Q_trans  = np.ones(len(temperature))
    S_trans  = np.zeros(len(temperature))
    dH_trans  = np.zeros(len(temperature))
    Cp_trans  = np.zeros(len(temperature))

    if molecule.twoD_gas:
        print("switching to 2D-gas for 2 lowest modes for %s"%molecule.name)
        # cycle through each temperature
        for (i,T) in enumerate(temperature):
            # partition function is: (2*pi*mass*kB*T/h**2)^(2/2) * area
            if (1==0): #3D gas, really here just for inspiration
                V = kB*T/P_ref
                Q_trans[i] = (2*pi*m*amu*kB*T/h**2)**(1.5) * V
                S_trans[i] = R * (2.5 + np.log( Q_trans[i] )) #
                Cp_trans[i] = R * 2.5 #NOTE: Cp = Cv + R
                dH_trans[i] = R * 2.5 * T
            else: #surface
                if (1==0): #Campbell + Arnadottir
                    V = kB*T/P_ref
                    Q_trans[i] = (2*pi*m*amu*kB*T/h**2)**(1.0) *V**0.66667
                    S_trans[i] = R * (2.0 + np.log( Q_trans[i] ))
                    Cp_trans[i] = R * 1.66667 #NOTE: Cp = Cv + 2/3R
                    dH_trans[i] = R * 1.66667 * T

                else: #area is not a function of temperature
                    Q_trans[i] = (2*pi*m*amu*kB*T/h**2) * area * sites
                    S_trans[i] = R * (2.0 + np.log( Q_trans[i] ))
                    Cp_trans[i] = R * 1.0 #NOTE: Cp = Cv
                    dH_trans[i] = R * 1.0 * T

    # add the results to the thermo object
    molecule.Q_trans = Q_trans
    molecule.S_trans = S_trans
    molecule.dH_trans = dH_trans
    molecule.Cp_trans = Cp_trans


    return


# subroutine for the vibrational mode
def get_vibrational_thermo(molecule,temperature):
    units = 1.0
    units *= molecule.h * molecule.c / molecule.kB * molecule. invcm_to_invm # K * cm
    amu = molecule.amu
    kB = molecule.kB
    h = molecule.h
    P_ref = molecule.P_ref
    mass = float(molecule.adsorbate_mass)


    #initialize the arrays for the partition function, entropy, enthalpy,
    #and heat capacity.
    Q_vib  = np.ones(len(temperature))
    S_vib  = np.zeros(len(temperature))
    dH_vib  = np.zeros(len(temperature))
    Cv_vib  = np.zeros(len(temperature))

    for (t,temp) in enumerate(temperature):
        for (n,nu) in enumerate(molecule.frequencies):
            if molecule.twoD_gas==True and n <= 1: #skip the first two if we do 2D gas
                #do nothing!
                Q_vib[t] *= 1.0
                S_vib[t] += 0.0
                dH_vib[t] += 0.0
                Cv_vib[t] += 0.0
            else:
                x = nu * units / temp #cm^-1 * K cm / K = dimensionless
                Q_vib[t]  *= 1.0 / (1.0 - np.exp( - x) )
                S_vib[t]  += -np.log( 1.0 - np.exp( - x ) ) + x * np.exp( - x) / (1.0 - np.exp( - x) )
                dH_vib[t] += x * np.exp( - x) / (1.0 - np.exp( - x) )
                Cv_vib[t] += x**2.0 * np.exp( - x) / (1.0 - np.exp( - x) )**2.0
        S_vib[t]  *= molecule.R
        dH_vib[t] *= molecule.R * temp
        Cv_vib[t] *= molecule.R

    # add the results to the thermo object
    molecule.Q_vib = Q_vib
    molecule.S_vib = S_vib
    molecule.dH_vib = dH_vib
    molecule.Cv_vib = Cv_vib #NOTE: the correction from Cv to Cp is handled in the translation partition function.
                             #if the molecule is tightly bound and thus the 2D-gas is not used,
                             #then we assume that Cp=Cv for the adsorbate.

    return

#-------------------------------------------------------------------------
#create the main thermo function that calls the individual modes
def thermo(molecule, temperature):

    #

    # call the subroutine for the vibrational partition function
    get_translation_thermo(molecule,temperature)
    get_vibrational_thermo(molecule,temperature)


    #now compute the correction to the heat of formation as you go from 0 to 298 K
    h_correction = 4.234 #kJ/mol. enthalpy_H(298) - enthalpy_H(0)
    c_correction = 1.051 #kJ/mol. enthalpy_C(298) - enthalpy_C(0)
    n_correction = 4.335 #kJ/mol. enthalpy_N(298) - enthalpy_N(0)
    o_correction = 4.340 #kJ/mol. enthalpy_O(298) - enthalpy_O(0)

    molecule.heat_of_formation_correction = 0.0
    molecule.heat_of_formation_correction += molecule.composition['H'] * h_correction
    molecule.heat_of_formation_correction += molecule.composition['C'] * c_correction
    molecule.heat_of_formation_correction += molecule.composition['N'] * n_correction
    molecule.heat_of_formation_correction += molecule.composition['O'] * o_correction

    # note that the partition function is the production of the individual terms,
    # whereas the thermodynamic properties are additive
    molecule.Q = molecule.Q_trans * molecule.Q_vib
    molecule.S = molecule.S_trans + molecule.S_vib
    molecule.dH = molecule.dH_trans + molecule.dH_vib
    molecule.Cp = molecule.Cp_trans + molecule.Cv_vib # see comments in each section regarding Cp vs Cv
    molecule.heat_of_formation_298K = molecule.heat_of_formation_0K + molecule.dH[0] - molecule.heat_of_formation_correction
    molecule.H = molecule.heat_of_formation_298K + molecule.dH - molecule.dH[0]

    print(molecule.heat_of_formation_298K)
    print(molecule.H[0])
    #This writes H_298, S_298 and appropriate indices of Cp to file (preparation for computing adsorption corrections)
    g = open("Pt_thermodata_adsorbates.py",'a+')
    g.write('[' + str(molecule.name) + ', Cpdata:, ' +  str(molecule.Cp[np.where(temperature==300)]*239.0057)[1:-1] + ', ' + str(molecule.Cp[np.where(temperature==400)]*239.0057)[1:-1] + ', '+ str(molecule.Cp[np.where(temperature==500)]*239.0057)[1:-1] + ', ' + str(molecule.Cp[np.where(temperature==600)]*239.0057)[1:-1] + ', ' + str(molecule.Cp[np.where(temperature==800)]*239.0057)[1:-1] + ', ' + str(molecule.Cp[np.where(temperature==1000)]*239.0057)[1:-1] + ', ' + str(molecule.Cp[np.where(temperature==1500)]*239.0057)[1:-1] + ', ' + ",'cal/(mol*K)', H298, " + str(molecule.H[0]*0.2390057) + ", 'kcal/mol', S298, " + str(molecule.S[0]*239.0057) + ", 'cal/(mol*K)']")
    g.write('\n')
    g.close()

    # now that we've computed the thermo properties, go ahead and fit them to a NASA polynomial
    fit_NASA(temperature, molecule)
    format_output(molecule)
    return

#-------------------------------------------------------------------------
#compute thermo properties from nasa polynomials
def get_thermo_from_NASA(temperature, molecule):

    a_low = molecule.a_low
    a_high = molecule.a_high
    R = molecule.R
    T_switch = molecule.T_switch

    i_switch = -1
    for i in range(len(temperature)):
        if temperature[i]==T_switch:
            i_switch = i

    cp_fit = np.zeros(len(temperature))
    h_fit = np.zeros(len(temperature))
    s_fit = np.zeros(len(temperature))
    for (i,temp) in enumerate(temperature):
        if temp <= T_switch:
            cp_fit[i] = a_low[0] + a_low[1]*temp + a_low[2]*temp**2.0  + a_low[3]*temp**3.0  + a_low[4]*temp**4.0
            h_fit[i] = a_low[0]*temp + a_low[1]/2.0*temp**2.0 + a_low[2]/3.0*temp**3.0  + a_low[3]/4.0*temp**4.0  + a_low[4]/5.0*temp**5.0 + a_low[5]
            s_fit[i] = a_low[0]*np.log(temp) + a_low[1]*temp + a_low[2]/2.0*temp**2.0  + a_low[3]/3.0*temp**3.0  + a_low[4]/4.0*temp**4.0 + a_low[6]
        else:
            cp_fit[i] = a_high[0] + a_high[1]*temp + a_high[2]*temp**2.0  + a_high[3]*temp**3.0  + a_high[4]*temp**4.0
            h_fit[i] = a_high[0]*temp + a_high[1]/2.0*temp**2.0 + a_high[2]/3.0*temp**3.0  + a_high[3]/4.0*temp**4.0  + a_high[4]/5.0*temp**5.0 + a_high[5]
            s_fit[i] = a_high[0]*np.log(temp) + a_high[1]*temp + a_high[2]/2.0*temp**2.0  + a_high[3]/3.0*temp**3.0  + a_high[4]/4.0*temp**4.0 + a_high[6]

    cp_fit *= R
    h_fit *= R
    s_fit *= R

    molecule.Cp_fit = cp_fit
    molecule.H_fit = h_fit
    molecule.S_fit = s_fit
    return


#-------------------------------------------------------------------------
#fit nasa coefficients
def fit_NASA(temperature, molecule):

    R = molecule.R
    heat_capacity = molecule.Cp
    reference_enthalpy = molecule.H[0]
    reference_entropy = molecule.S[0]
    T_switch = molecule.T_switch

    i_switch = -1
    for i in range(len(temperature)):
        if temperature[i]==T_switch:
            i_switch = i
    if i_switch==-1:
        print("We have a problem! Cannot find switching temperature")


    #start by creating the independent variable matrix for the low-temperature fit
    YT = np.array( [ np.ones(len(temperature[:i_switch+1])), temperature[:i_switch+1], temperature[:i_switch+1]**2.0, temperature[:i_switch+1]**3.0, temperature[:i_switch+1]**4.0 ],dtype=np.float64 ) #this is transpose of our Y
    Y = YT.transpose() #this is the desired Y

    b = heat_capacity[:i_switch+1] / R
    a_low = np.linalg.lstsq(Y, b)[0]

    T_ref = 298.15
    #now determine the enthalpy coefficient for the low-T region
    subtract = a_low[0] + a_low[1]/2.0*T_ref + a_low[2]/3.0*T_ref**2.0 + a_low[3]/4.0*T_ref**3.0  + a_low[4]/5.0*T_ref**4.0
    a_low = np.append(a_low, reference_enthalpy / R - subtract * T_ref)
    #now determine the entropy coefficient for the low-T region
    subtract = a_low[0] * np.log(T_ref) + a_low[1]*T_ref     + a_low[2]/2.0*T_ref**2.0  + a_low[3]/3.0*T_ref**3.0  + a_low[4]/4.0*T_ref**4.0
    a_low = np.append(a_low, reference_entropy / R - subtract )

    #
    # NOW SWITCH TO HIGH-TEMPERATURE REGIME!
    #
    T_ref = T_switch
    #compute the heat capacity, enthalpy, and entropy at the switching point
    Cp_switch = a_low[0] + a_low[1]*T_ref + a_low[2]*T_ref**2.0  + a_low[3]*T_ref**3.0  + a_low[4]*T_ref**4.0
    H_switch = a_low[0]*T_ref + a_low[1]/2.0*T_ref**2.0 + a_low[2]/3.0*T_ref**3.0  + a_low[3]/4.0*T_ref**4.0  + a_low[4]/5.0*T_ref**5.0 + a_low[5]
    S_switch = a_low[0]*np.log(T_ref) + a_low[1]*T_ref + a_low[2]/2.0*T_ref**2.0  + a_low[3]/3.0*T_ref**3.0  + a_low[4]/4.0*T_ref**4.0 + a_low[6]

    #now repeat the process for the high-temperature regime
    a_high = [0.0]
    YT = np.array( [ temperature[i_switch:], temperature[i_switch:]**2.0, temperature[i_switch:]**3.0, temperature[i_switch:]**4.0 ],dtype=np.float64 ) #this is transpose of our Y
    Y = YT.transpose() #this is the desired Y

    b = heat_capacity[i_switch:] / R - Cp_switch
    a_high = np.append(a_high, np.linalg.lstsq(Y, b)[0])
    a_high[0] = Cp_switch - (a_high[0] + a_high[1]*T_switch + a_high[2]*T_switch**2.0  + a_high[3]*T_switch**3.0  + a_high[4]*T_switch**4.0)

    a_high = np.append(a_high, H_switch - (a_high[0] + a_high[1]/2.0*T_ref + a_high[2]/3.0*T_ref**2.0  + a_high[3]/4.0*T_ref**3.0  + a_high[4]/5.0*T_ref**4.0)*T_ref )
    a_high = np.append(a_high, S_switch - (a_high[0]*np.log(T_ref) + a_high[1]*T_ref + a_high[2]/2.0*T_ref**2.0  + a_high[3]/3.0*T_ref**3.0  + a_high[4]/4.0*T_ref**4.0) )

    #Check to see if there is a discontinuity
    if (1==0):
        print("\ncheck for discontinuities:")
        cp_low_Tswitch = a_low[0] + a_low[1]*T_switch + a_low[2]*T_switch**2.0  + a_low[3]*T_switch**3.0  + a_low[4]*T_switch**4.0
        cp_high_Tswitch = a_high[0] + a_high[1]*T_switch + a_high[2]*T_switch**2.0  + a_high[3]*T_switch**3.0  + a_high[4]*T_switch**4.0
        H_low_Tswitch = a_low[0]*T_switch + a_low[1]/2.0*T_switch**2.0 + a_low[2]/3.0*T_switch**3.0  + a_low[3]/4.0*T_switch**4.0  + a_low[4]/5.0*T_switch**5.0 + a_low[5]
        H_high_Tswitch = a_high[0]*T_switch + a_high[1]/2.0*T_switch**2.0 + a_high[2]/3.0*T_switch**3.0  + a_high[3]/4.0*T_switch**4.0  + a_high[4]/5.0*T_switch**5.0 + a_high[5]
        S_low_Tswitch = a_low[0]*np.log(T_switch) + a_low[1]*T_switch + a_low[2]/2.0*T_switch**2.0  + a_low[3]/3.0*T_switch**3.0  + a_low[4]/4.0*T_switch**4.0 + a_low[6]
        S_high_Tswitch = a_high[0]*np.log(T_switch) + a_high[1]*T_switch + a_high[2]/2.0*T_switch**2.0  + a_high[3]/3.0*T_switch**3.0  + a_high[4]/4.0*T_switch**4.0 + a_high[6]

        print("discontinuity at T_switch for Cp/R is %.4F"%(cp_low_Tswitch - cp_high_Tswitch))
        print("discontinuity at T_switch for H/R is %.4F"%(H_low_Tswitch - H_high_Tswitch))
        print("discontinuity at T_switch for S/R is %.4F"%(S_low_Tswitch - S_high_Tswitch))

    #line = '\n\t !cut and paste this value into the cti file!\n'
    line = '\tthermo = (\n'
    line += "\t\tNASA( [%.1F, %.1F], [%.8E, %.8E,\n \t\t %.8E, %.8E, %.8E,\n \t\t %.8E, %.8E]), \n"%(300.0, 1000.0, a_low[0], a_low[1], a_low[2], a_low[3], a_low[4], a_low[5], a_low[6])
    line += "\t\tNASA( [%.1F, %.1F], [%.8E, %.8E,\n \t\t %.8E, %.8E, %.8E,\n \t\t %.8E, %.8E]), \n"%(1000.0, max(temperature), a_high[0], a_high[1], a_high[2], a_high[3], a_high[4], a_high[5], a_high[6])
    line += "\t\t ),\n"

    molecule.thermo_lines = line

    molecule.a_low = a_low
    molecule.a_high = a_high

    return


#-------------------------------------------------------------------------
#compare NASA fits to computed fits
def compare_NASA_to_thermo(temperature, molecule):

    fig = pylab.figure(dpi=300,figsize=(12,4))
    gs = gridspec.GridSpec(1, 3)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    if (1==1): #use this to plot the absolute curves
        ax0.plot(temperature, molecule.Cp, marker='o', markeredgecolor='r',color='w',alpha=0.5,linestyle='None')
        ax0.plot(temperature, molecule.Cp_fit, 'b', linewidth=2)
        ax1.semilogy(temperature, molecule.H - molecule.heat_of_formation_298K, marker='o', markeredgecolor='r',color='w',alpha=0.5,linestyle='None')
        ax1.semilogy(temperature, molecule.H_fit - molecule.heat_of_formation_298K, 'b', linewidth=2)
        ax2.semilogy(temperature, molecule.S, marker='o', markeredgecolor='r',color='w',alpha=0.5,linestyle='None')
        ax2.semilogy(temperature, molecule.S_fit, 'b', linewidth=2)
        ax0.set_ylim(min(molecule.Cp_fit)*0.9, max(molecule.Cp_fit)*1.025)
        ax1.set_ylim(min(molecule.H - molecule.heat_of_formation_298K)*0.9, max(molecule.H - molecule.heat_of_formation_298K)*1.025)
        ax2.set_ylim(min(molecule.S_fit)*0.9, max(molecule.S_fit)*1.025)
        ax1.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))

    else: #use this one to plot the percent change
        ax0.plot(temperature, 1.0 - molecule.Cp/molecule.Cp_fit, 'b', linewidth=2)
        ax1.plot(temperature, 1.0 - molecule.H/molecule.H_fit, 'b', linewidth=2)
        ax2.plot(temperature, 1.0 - molecule.S/molecule.S_fit, 'b', linewidth=2)
        ax0.set_ylim(-5E-3, 5E-3)
        ax1.set_ylim(-5E-3, 5E-3)
        ax2.set_ylim(-5E-3, 5E-3)
        ax1.yaxis.set_major_locator(MaxNLocator(4))
        ax2.yaxis.set_major_locator(MaxNLocator(4))

    # now make it look better
    ax0.set_xlim(min(temperature)*0.95, max(temperature)*1.025)
    ax0.xaxis.set_major_locator(MaxNLocator(4))
    ax0.yaxis.set_major_locator(MaxNLocator(4))
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.set_title("Heat capacity")
    ax0.set_xlabel("temperature [K]", fontsize=12)

    ax1.set_xlim(min(temperature)*0.95, max(temperature)*1.025)
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title("Change in enthalpy")
    ax1.set_xlabel("temperature [K]", fontsize=12)

    ax2.set_xlim(min(temperature)*0.95, max(temperature)*1.025)
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_title("Entropy")
    ax2.set_xlabel("temperature [K]", fontsize=12)
    plt.tight_layout()

    return


#-------------------------------------------------------------------------
#compare NASA fits to computed fits
def compare_Cantera_to_thermo(temperature, molecule):

    fig = pylab.figure(dpi=300,figsize=(12,4))
    gs = gridspec.GridSpec(1, 3)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    ax0.plot(temperature, molecule.Cp, marker='o', markeredgecolor='r',color='w',alpha=0.5,linestyle='None')
    ax0.plot(temperature, molecule.cantera_cp, 'b', linewidth=2)

    ax1.semilogy(temperature, molecule.H- molecule.heat_of_formation_298K, marker='o', markeredgecolor='r',color='w',alpha=0.5,linestyle='None')
    ax1.semilogy(temperature, molecule.cantera_h- molecule.heat_of_formation_298K, 'b', linewidth=2)

    ax2.semilogy(temperature, molecule.S, marker='o', markeredgecolor='r',color='w',alpha=0.5,linestyle='None')
    ax2.semilogy(temperature, molecule.cantera_s, 'b', linewidth=2)

    # now make it look better
    ax0.set_xlim(min(temperature)*0.95, max(temperature)*1.025)
    ax0.set_ylim(min(molecule.Cp_fit)*0.9, max(molecule.Cp_fit)*1.025)
    ax0.xaxis.set_major_locator(MaxNLocator(4))
    ax0.yaxis.set_major_locator(MaxNLocator(4))
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.set_title("Heat capacity")
    ax0.set_xlabel("temperature [K]", fontsize=20)

    ax1.set_xlim(min(temperature)*0.95, max(temperature)*1.025)
    ax1.set_ylim(min(molecule.H - molecule.heat_of_formation_298K)*0.9, max(molecule.H - molecule.heat_of_formation_298K)*1.025)
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title("Change in enthalpy")
    ax1.set_xlabel("temperature [K]", fontsize=20)

    ax2.set_xlim(min(temperature)*0.95, max(temperature)*1.025)
    ax2.set_ylim(min(molecule.S_fit)*0.9, max(molecule.S_fit)*1.025)
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_title("Entropy")
    ax2.set_xlabel("temperature [K]", fontsize=20)
    plt.tight_layout()

    return



#-------------------------------------------------------------------------
#print(output in cti format)
def format_output(molecule):

    line = '\n'
    line += 'species(name = "%s",\n'%(molecule.name)
    line += '\tatoms = "'
    for element in molecule.composition:
        if molecule.composition[element]>0:
            line += " %s:%d"%(element, molecule.composition[element])
    line += '",\n'
    line += "\tsize = %d,\n"%(molecule.site_occupation_number)
    line += molecule.thermo_lines
    line += '    longDesc = u"""Calculated by x at x University using statistical mechanics (file: compute_NASA_for_Pt-adsorbates.ipynb). \n'
    line += "                   Based on DFT calculations by x at x.\n"
#     line += "            DFT binding energy: %.3F %s.\n" %(molecule.DFT_binding_energy, molecule.DFT_binding_energy_units.replace("'",""))
#     if molecule.site_occupation_number == 1:
#         line += "            Linear scaling parameters: ref_adatom_%s = %.3F %s, psi = %.5F %s, gamma_%s(X) = %.3F."%(molecule.binding_atom1, molecule.ref_adatom_Eb1, molecule.ref_adatom_Eb1_units.replace("'",""), molecule.linear_scaling_psi, molecule.linear_scaling_psi_units.replace("'",""), molecule.binding_atom1, molecule.linear_scaling_gamma)
#     else:
#         line += "            Linear scaling parameters: ref_adatom_%s1 = %.3F %s, ref_adatom_%s2 = %.3F %s, psi = %.5F %s, gamma_%s1(X) = %.3F, gamma_%s2(X) = %.3F."%(molecule.binding_atom1, molecule.ref_adatom_Eb1, molecule.ref_adatom_Eb1_units.replace("'",""), molecule.binding_atom2, molecule.ref_adatom_Eb2, molecule.ref_adatom_Eb2_units.replace("'",""), molecule.linear_scaling_psi, molecule.linear_scaling_psi_units.replace("'",""), molecule.binding_atom1, molecule.linear_scaling_gamma, molecule.binding_atom2, molecule.linear_scaling_gamma_B)
    if molecule.twoD_gas:
        line += '\n            The two lowest frequencies, %.1F and %.1F %s, where replaced by the 2D gas model.' %(molecule.frequencies[0], molecule.frequencies[1], molecule.frequencies_units.replace("'",""))
    line += '""",\n\t)\n'

    molecule.species_lines = line

    return


#-------------------------------------------------------------------------
#Define the input parser
def parse_input_file(inputfile, molecule, element1):

    import sys, os
    script_dir= str(element1)
    rel_path = str(inputfile)
    abs_file_path = os.path.join(script_dir, rel_path)

    input_file = open(abs_file_path,'r')
    lines = input_file.readlines()
    input_file.close()

    error_name = True
    error_DFT_binding_energy = True
    error_heat_of_formation_0K = True
    error_composition = True
    error_sites = True
    error_adsorbate_mass = True
    error_frequencies = True
#     error_linear_scaling_gamma = True
#     error_linear_scaling_gamma_B = True
#     error_linear_scaling_psi = True
#     error_ref_adatom_Eb1 = True
#     error_ref_adatom_Eb2 = True

    molecule.binding_atom1 =  str(element1)

    for line in lines:
        #start by looking for the name
        if line.strip().startswith("name"):
            bits = line.split('=')
            name = bits[1].strip().replace("'","").replace('"','')
            molecule.name = name
            error_name = False
        #now look for the binding energy
#         elif line.strip().startswith("DFT_binding_energy"):
#             bits = line.split('=')
#             binding_energy_info = bits[1].strip().replace("[","").replace("]","").split(',')
#             binding_energy = float(binding_energy_info[0])
#             units = binding_energy_info[1].strip().replace("'","").replace('"','')
#             if units=='eV' or units=='kJ/mol':
#                 molecule.DFT_binding_energy = binding_energy
#                 molecule.DFT_binding_energy_units = units.strip()
#                 error_DFT_binding_energy = False
#             else:
#                 print("DFT binding energy is missing proper units!\n Please use either 'eV' or 'kJ/mol'")
#                 break
        #now look for the heat of formation
        elif line.strip().startswith("heat_of_formation_0K"):
            bits = line.split('=')
            heat_info = bits[1].strip().replace("[","").replace("]","").split(',')
            heat_of_formation = float(heat_info[0])
            units = heat_info[1].strip().replace("'","").replace('"','')
            #make sure that the units are given, and that the final value is kJ/mol
            if units=='kJ/mol':
                molecule.heat_of_formation_0K = heat_of_formation
                molecule.heat_of_formation_0K_units = units.strip()
                error_heat_of_formation_0K = False
            elif units=='eV':
                molecule.heat_of_formation_0K = heat_of_formation * molecule.eV_to_kJpermole
                molecule.heat_of_formation_0K_units = 'kJ/mol'
                error_heat_of_formation_0K = False
            else:
                print("heat of formation is missing proper units!\n Please use either 'eV' or 'kJ/mol'")
                break
        #now look for the composition
        elif line.strip().startswith("composition"):
            bits = line.split('=')
            composition = bits[1].strip().replace("{","").replace("}","").split(',')
            molecule.composition = {}
            for pair in composition:
                element, number = pair.split(":")
                element = element.strip().replace("'","").replace('"','')
                number = int(number)
                molecule.composition[element]=number
            N_adsorbate_atoms = 0
            metal_surface = ['Pt', 'Cu', 'Ni', 'Ag']
            for element in molecule.composition:
                if element not in metal_surface:
                    N_adsorbate_atoms += molecule.composition[element]
            error_composition = False
        #now look for the site occupancy
        elif line.strip().startswith("sites"):
            bits = line.split('=')
            site_occupation_number = int(bits[1])
            molecule.site_occupation_number = site_occupation_number
            error_sites = False
        #now look for the molecule mass
        elif line.strip().startswith("adsorbate_mass"):
            bits = line.split('=')
            adsorbate_mass_info = bits[1].strip().replace("[","").replace("]","").split(',')
            adsorbate_mass = float(adsorbate_mass_info[0])
            units = adsorbate_mass_info[1].strip().replace("'","").replace('"','')
            if units=='amu':
                molecule.adsorbate_mass = adsorbate_mass
                molecule.adsorbate_mass_units = units.strip()
                error_adsorbate_mass = False
            else:
                print("Adsorbate mass is missing proper units!\n Please use either 'eV' or 'kJ/mol'")
                break
        #now look for the reference adatom binding energy
#         elif line.strip().startswith("linear_scaling_binding_atom ="):
#             bits = line.split('=')
#             ref_adatom_info = bits[1].strip().replace("[","").replace("]","").split(',')
#             ref_adatom_Eb1 = float(ref_adatom_info[0])
#             units = ref_adatom_info[1].strip().replace("'","").replace('"','')
#             if units=='eV' or units=='kJ/mol':
#                 molecule.ref_adatom_Eb1 = ref_adatom_Eb1
#                 molecule.ref_adatom_Eb1_units = units.strip()
#                 error_ref_adatom_Eb1 = False
#             else:
#                 print("Reference adatom binding energy is missing proper units!\n Please use either 'eV' or 'kJ/mol'")
#                 break
        #now look for the linear scaling parameter gamma
#         elif line.strip().startswith("linear_scaling_gamma(X) ="):
#             bits = line.split('=')
#             linear_scaling_gamma = bits[1].strip().replace("[","").replace("]","")
#             molecule.linear_scaling_gamma = float(linear_scaling_gamma)
#             error_linear_scaling_gamma = False
#         #now look for the linear scaling parameter gamma for second atom (if it is bidentate)
#         elif line.strip().startswith("linear_scaling_gamma(X)_B"):
#             bits = line.split('=')
#             linear_scaling_gamma_B = bits[1].strip().replace("[","").replace("]","")
#             molecule.linear_scaling_gamma_B = float(linear_scaling_gamma_B)
#             error_linear_scaling_gamma_B = False
        #now look for the linear scaling parameter psi
#         elif line.strip().startswith("linear_scaling_psi"):
#             bits = line.split('=')
#             linear_scaling_psi_info = bits[1].strip().replace("[","").replace("]","").split(',')
#             linear_scaling_psi = float(linear_scaling_psi_info[0])
#             units = linear_scaling_psi_info[1].strip().replace("'","").replace('"','')
#             if units=='eV' or units=='kJ/mol':
#                 molecule.linear_scaling_psi = linear_scaling_psi
#                 molecule.linear_scaling_psi_units = units.strip()
#                 error_linear_scaling_psi = False
#             else:
#                 print("Linear scaling parameter psi is missing proper units!\n Please use either 'eV' or 'kJ/mol'")
#                 break
        #now look for the frequencies
        elif line.strip().startswith("frequencies"):
            bits = line.split('=')
            freq_info = bits[1].strip().replace("[","").replace("]","").split(',')
            N_freq_computed = 3*N_adsorbate_atoms
            if len(freq_info)!=N_freq_computed+1:
                print("ERROR: The number of frequencies is not what was expected\n %d expected, but only %d received"%(N_freq_computed, len(freq_info)-1))
            units = freq_info[-1]
            if units=='eV' or units!='cm-1':
                molecule.frequencies_units = units.strip()
                molecule.frequencies = []
                for i in range(len(freq_info)-1):
                    temp_string = freq_info[1]
                    match = re.search(r'np\.float64\((.*?)\)', temp_string)
                    val = np.float64(match.group(1))
                    molecule.frequencies.append(float(val))
                error_frequencies = False
                #if the two lowest frequencies are less than the cutoff value (This assumes that they are sorted!)
                if molecule.frequencies[1]<molecule.cutoff_frequency:
                    #print("switching to 2D-gas for 2 lowest modes for %s"%name
                    molecule.twoD_gas = True
        #now look for the second reference adatom binding energy (if bidentate)
#         elif line.strip().startswith("linear_scaling_binding_atom_B"):
#             bits = line.split('=')
#             ref_adatom_info = bits[1].strip().replace("[","").replace("]","").split(',')
#             ref_adatom_Eb2 = float(ref_adatom_info[0])
#             units = ref_adatom_info[1].strip().replace("'","").replace('"','')
#             if units=='eV' or units=='kJ/mol':
#                 molecule.ref_adatom_Eb2 = ref_adatom_Eb2
#                 molecule.ref_adatom_Eb2_units = units.strip()
#                 error_ref_adatom_Eb2 = False
#             else:
#                 print("Reference adatom binding energy is missing proper units!\n Please use either 'eV' or 'kJ/mol'")
#                 break
        #now look for the second binding atom (if bidentate)
#         elif line.strip().startswith("binding_atom_B"):
#             bits = line.split('=')
#             binding_atom2 = bits[1][1:-1]
#             molecule.binding_atom2 = str(binding_atom2)
#         elif line.strip().startswith("exception"):
#             molecule.twoD_gas = False



#     if error_name or error_DFT_binding_energy or error_heat_of_formation_0K or error_composition or error_sites or error_frequencies or error_linear_scaling_gamma or error_linear_scaling_psi:
    if error_name or error_heat_of_formation_0K or error_composition or error_sites or error_frequencies:
        print("Input file is missing information: %s"%(inputfile))
    else:
        print("successfully parsed file %s"%(inputfile))

    return


def input_generation(compositions, traj_ps, output_p, ocp=False):
    """
     This block calculates the heat of formation at 0 K
     :compositions: a dictionary with species name and its composition eg: {'cx':{'H':0, 'C':1, 'N':0, 'O':0, 'Rh':1}}
     :traj_ps: a list of lists with the trajectory file and vibrational file paths for each adsorbate
     :output_p: path of the output file for the input data
     :ocp: Bool, must add the gas phase counterpart path into traj_ps
    """
    adsorbates = list(compositions.keys())
    # ATCT values
    h0_ch4 = -66.556 # kJ/mol
    h0_h2o = -238.938 # kJ/mol
    h0_nh3 = -38.565 # kJ/mol
    h0_h2 = 0.0 # kJ/mol
    h0_co = -113.804 # kJ/mol
    h0_n2 = 0.0 # kJ/mol

    # Frequencies
    ads_frequencies = {}
    if not ocp:
        # ch4, h2, h2o BEEF-VDW calculated results
        paths = ['ch4', 'h2', 'h2o']
        zpes = {'ch4': 1.196, 'h2': 0.277, 'h2o': 0.609}
        potential_energies = {}
        for p in paths:
            traj = Trajectory(os.path.join(p, 'ads_vib.traj'))[-1]
            potential_energies[p] = traj.get_potential_energy()
        E_ch4_g = potential_energies['ch4'] + zpes['ch4']
        E_h2_g = potential_energies['h2'] + zpes['h2']
        E_h2o_g = potential_energies['h2o'] + zpes['h2o']
        print(E_ch4_g)
        E_nh3_g = 0
        h0_gas = {}
        for i, v in compositions.items():
            a = v['C']
            b = v['O']
            c = v['N']
            d = v['H']
            e = d / 2 - 2 * a - b - 1.5 * c
            h0_atct = (a * h0_ch4 + b * h0_h2o + c * h0_nh3 + e * h0_h2) / 96.4869
            E_g = a * E_ch4_g + b * E_h2o_g  + c * E_nh3_g + e * E_h2_g
            h0_gas[i] = h0_atct - E_g
    else:
        E_o_g_ocp = -7.204
        E_h_g_ocp = -3.477
        E_c_g_ocp = -7.282
        E_n_g_ocp = -8.083
        E_co_g = E_c_g_ocp + E_o_g_ocp
        E_h2_g = E_h_g_ocp * 2
        E_h2o_g = E_h_g_ocp * 2 + E_o_g_ocp
        E_n2_g = E_n_g_ocp * 2
        h0_gas = {}
        for i, v in compositions.items():
            a = v['C']
            f = v['O']
            c = v['N']
            d = v['H']
            e = d / 2 - (f - a)
            h0_atct = (a * h0_co + (f - a) * h0_h2o + c / 2 * h0_n2 + e * h0_h2) / 96.4869
            E_g = a * E_co_g + (f - a) * E_h2o_g  + c / 2 * E_n2_g + e * E_h2_g
            h0_gas[i] = h0_atct - E_g

    # Read slab energy and zero-corrected energy of adsorbates from BEEF_VDW
    E_adss = {}
    # use your slab energy
    E_slab = Trajectory('slab_restart.traj')[-1].get_potential_energy()
    for i, v in enumerate(adsorbates):
        p = traj_ps[i][0]
        E_ads = Trajectory(p)[-1].get_potential_energy()
        with open(traj_ps[i][1]) as f:
            freqs = f.read()
        zpe = float(freqs.split()[-2])
        frequencies = np.array(pd.to_numeric(freqs.split()[7:-5:3], errors='coerce'))
        # set imaginary modes to 12 cm^-1
        frequencies = np.nan_to_num(frequencies, nan=12)
        ads_frequencies[v] = frequencies
        E_ads += zpe
        if ocp:
            E_atomic = {'H':-3.477, 'O':-7.204, 'C':-7.282, 'N':-8.083}
            E_gas = 0
            for j in E_atomic.keys():
                E_gas += E_atomic[j] * compositions[v][j]
            E_adss[v] = E_ads + E_gas
        else:
            E_adss[v] = E_ads - E_slab

    # calculate the heat of formation of adsorbates at 0 K unit kJ/mol
    h0_ads = {}
    for i in adsorbates:
        h0_ads[i] = (h0_gas[i] + E_adss[i]) * 96.4869
        print(f"The heat of formation at 0 K for {i} is {h0_ads[i]} kJ/mol")

    # generate the input data to the functions below for each adsorbate
    input_data = {}
    atomic_mass = {'C': 12.011, 'O': 15.999, 'H': 1.00784, 'N': 14.0067, 'Ag': 0, 'Pt': 0, 'Cu': 0}
    for i, v in h0_ads.items():
        ads_details = {}
        ads_details['name'] = i
        ads_details['heat_of_formation_0K'] = [v, 'kJ/mol']
        ads_details['composition'] = compositions[i]
        ads_details['sites'] = 1
        molecular_mass = 0
        for key in compositions[i].keys():
            molecular_mass += compositions[i][key] * atomic_mass[key]
        ads_details['adsorbate_mass'] = [molecular_mass, 'amu']
        freq_list = list(ads_frequencies[i])
        freq_list.append('cm-1')
        ads_details['frequencies'] = freq_list
        input_data[i] = ads_details

    # write data into files
    if not os.path.exists(output_p):
        os.makedirs(output_p)
    for ads in adsorbates:
        data_str_ls = []
        for i, v in input_data[ads].items():
            new_line = str(i) + ' = ' + str(v)
            data_str_ls.append(new_line)
        data_str = "\n".join(data_str_ls)
        with open(f'{output_p}/{ads}.dat', 'w') as f:
            f.write(data_str)

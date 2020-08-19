import time
from sympy import *
from fdint import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ternary

#_____________________________________________________________________________________________________________________________________________________#
#*****************************************************************************************************************************************************#
# DILUTE SOLVUS BOUNDARY CALCULATION
# MOTIVE: TO CALCULATE THE SOLVUS BOUNDARY AS A FUNCTION OF TEMPERATURE AND CHEMICAL POTENTIAL OF ATOMIC SPECIES (DEPENDS ON THE THREE PHASE REGION).
# THE FOLLOWING EQUATIONS ARE USED
#
# 1 _ EXPRESSION FOR DEFECT FORMATION ENERGY AS A FUNCTION OF ELECTRON CHEMICAL POTENTIAL (mu_e) AND 
#     CHEMICAL POTENTIAL OF ATOMIC SPECIES (mu_i) IN THE 3 (OR 2) PHASE EQUILLIBRIUM UNDER CONSIDERATION.
# 2 _ EQUATION ENFORCING CHARGE NEUTRALITY CONDITION REQUIRES:
#             *  EXPRESSION FOR DEFECT CONCENTRATION AS A FUNCTION OF DEFECT FORMATION ENERGY.
#             *  EXPRESSION FOR HOLE CARRIER CONCENTRATION (NEEDS DOS MASS)
#             *  EXPRESSION FOR ELECTRON CARRIER CONCENTRATION (NEEDS DOS MASS)
# 3 _ CHANGE IN COMPOSITION DUE TO DEFECTS.
#
# SUBSTITUTE 1 IN 2 AND SOLVE TO CALCULATE THE ELECTRON CHEMICAL POTENTIAL AS A FUNCTION OF TEMPERATURE [mu_e(T)] FOR A PARTICULAR SET OF ATOMIC 
# CHEMICAL POTENTIALS [mu_i] ASSOCIATED WITH DIFFERENT 2-PHASE EQUILIBRIA. THIS FUNCTION IS USED SUBSEQUENTLY TO DETERMINE DEFECT CONCENTRATIONS AS A 
# FUNCTION OF TEMPERATURE. TEMPERATURE DEPENDENT PHASE BOUNDARY COMPOSITIONS ARE CALCULATED USING 3 FROM THE DEFECT CONCENTRATIONS OBTAINED.
#*****************************************************************************************************************************************************#
#_____________________________________________________________________________________________________________________________________________________#

start_time = time.time()

#CHARGE ON A DEFECT OF ANY TYPE
ne_5 = -5
ne_4 = -4
ne_3 = -3
ne_2 = -2
ne_1 = -1
ne0 = 0
ne1 = 1
ne2 = 2
ne3 = 3
ne4 = 4

#NUMBER OF DEFECT SITES IN A DEFECT SUPERCELL OF ANY TYPE
N_site1 = 1
N_site2 = 2
N_site3 = 3

#NUMBER OF SYMMETRICALLY EQUIVALENT SITES OF A PARTICULAR DEFECT TYPE
N_sym1 = 1
N_sym2 = 2

#NUMBER OF ATOMS ADDED
N_1 = -1
N0 = 0
N1 = 1

#STOICHIOMETRIC NUMBER OF ATOMS
N_Mg0 = 3
N_Sb0 = 2
N_Te0 = 0 

#NUMBER OF STEPS
a = 1                                                                       #number of steps over which the temperature is varied
b = 600                                                                     #number of steps over which the chemical potential is varied
c = 100                                                                      #step size for temperature
d = -0.001                                                                  #step size for electron chemical potential
e = 800                                                                        #initial temperature
f = 0.6                                                                      #initial electron chemical potential
g = 3                                                                        #number of atomic chemical potential
h1 = 40 	                                                                      #Points in the 2-phase region                                                    

#BAND GAP
e_g = 0.4276

#OTHER CONSTANTS
v = 7.5127 * 10**21                                                          #number of formula units per cm-3. (calculated from experimental lattice parameters).
k1 = 1.38 * 10**-23                                                          #Boltzman constant in m2 kg s-2 K-1
k = 8.6173303 * 10**-5                                                       #Boltzman constant in eV K-1
h = 6.636*10**-34								#Plank's constant
m = 9.1 * 10**-31
pi = 3.14159

#EQUILLIBRIUM DEPENDENT ATOMIC CHEMICAL POTENTIAL
mu_Mg = [0, 0, -0.824, -0.824]						     #atomic chemical potentials of Mg in different 2 and 3-phase regions.
mu_Sb = [-1.221, -1.221, 0, 0]						     #atomic chemical potentials of Sb in different 2 and 3-phase regions.
mu_Te = [-1.601, -0.801, 0.0, -0.80]				     		#atomic chemical potentials of Te in different 2 and 3-phase regions.

mu_Mg1 = [[(1-u)*mu_Mg[t] + u*mu_Mg[t+1] for u in np.linspace(0,1,h1)] for t in range(g)]  #linear combination of atomic chemical potentials
mu_Sb1 = [[(1-u)*mu_Sb[t] + u*mu_Sb[t+1] for u in np.linspace(0,1,h1)] for t in range(g)]
mu_Te1 = [[(1-u)*mu_Te[t] + u*mu_Te[t+1] for u in np.linspace(0,1,h1)] for t in range(g)]

w = [0 for q in range(b)]                                                     #Table writing the carrier concentration as a function of electron chemical potential
y = [0 for q in range(b)]                                                     #Table to solve for electron chemical potential by enforcing the charge neutrality condition
z = [[[0 for r in range(a)] for x in range(h1)] for t in range(g)]            #Electron chemical potential will be written in this table as a function of temperature

#TEMPERATURE DEPENDENCE OF COMPOSITION
T = [c * (s + 1) + e for s in range(a)]                                                       #Table for writing Sb composition as a function temperature 
cSb = [[[0 for s in range(a)] for x in range(h1)] for t in range(g)] 
cMg = [[[0 for s in range(a)] for x in range(h1)] for t in range(g)]
cTe = [[[0 for s in range(a)] for x in range(h1)] for t in range(g)]

for o in range(g):                                                                     			#for loop over the atomic chemical potential between different two-phase regions
	for p in range(h1):             								#for loop over the atomic chemical potential in a two-phase region
		for i in range(a):                                                           		#for loop over the temperature range
			for j in range(b):                                                    		#for loop for electron chemical potential
				mu_e =  f + (j * d)
				Mg1Vac_2 = N_site1 * exp(-((-2 * mu_e + 1.5915 + mu_Mg1[o][p])/(k*T[i]))) * v           #Mg vacancy #1, charge = -2
				Mg2Vac_2 = N_site2 * exp(-((-2 * mu_e + 1.6433 + mu_Mg1[o][p])/(k*T[i]))) * v           #Mg vacancy #2, charge = -2
				SbVac1 = N_site2 * exp(-((1 * mu_e + 1.8304 + mu_Sb1[o][p]))/(k*T[i])) * v       	#Sb vacancy, charge = +1
				Mgi2 = N_site1 * exp(-((2 * mu_e + 0.0389 - mu_Mg1[o][p])/(k*T[i]))) * v         	#Mg interstitial, charge = +2
				Mg_Sb3  =  N_site2 * exp(-((3 * mu_e + 1.658 - mu_Mg1[o][p] + mu_Sb1[o][p])/(k*T[i]))) * v 	#Mg_Sb anti-site, charge = +3

				Sb_1Mg1 = N_site1 * exp(-((1.347 + 1 * mu_e - mu_Sb1[o][p] + mu_Mg1[o][p])/(k*T[i]))) * v	#Sb_Mg anti-site #1, charge = +1
				Sb_2Mg1 = N_site2 * exp(-((1.703 + 1 * mu_e - mu_Sb1[o][p] + mu_Mg1[o][p])/(k*T[i]))) * v	#Sb_Mg anti-site #2, charge = +1
				Te_Sb1 = N_site2 * exp(-((1 * mu_e + 0.256 - mu_Te1[o][p] + mu_Sb1[o][p])/(k*T[i]))) * v    	#Te_Sb substitution, charge = 1,
 
				n_e = (4*pi*(((2*0.542*m*k1*T[i])/(h**2))**1.5)) * fdk(0.5, (-(e_g - mu_e)/(k * T[i]))) * 10**-6                  #number of n-type carriers
				p_h = (4*pi*(((2*0.409*m*k1*T[i])/(h**2))**1.5)) * fdk(0.5, (-(mu_e)/(k * T[i]))) * 10**-6                       #number of p-type carriers

				w[j] = ((ne_2 * Mg1Vac_2 + ne_2 * Mg2Vac_2 + ne1 * SbVac1 + ne2 * Mgi2 + ne3 * Mg_Sb3 + ne1 * Sb_1Mg1 + ne1 * Sb_2Mg1 + ne1 * Te_Sb1) - n_e + p_h)
				y[j] = abs((ne_2 * Mg1Vac_2 + ne_2 * Mg2Vac_2 + ne1 * SbVac1 + ne2 * Mgi2 + ne3 * Mg_Sb3 + ne1 * Sb_1Mg1 + ne1 * Sb_2Mg1 + ne1 * Te_Sb1) - n_e + p_h)
		
			z[o][p][i] = f + (d*(np.argmin(y)))                                                    		 #writting chemical potential at different temperature T[i]
			Mg1Vac_2 = N_site1 * exp(-((-2 * z[o][p][i] + 1.5915 + mu_Mg1[o][p])/(k*T[i])))        		 #Mg vacancy #1, charge = -2
			Mg2Vac_2 = N_site2 * exp(-((-2 * z[o][p][i] + 1.6433 + mu_Mg1[o][p])/(k*T[i])))         	 #Mg vacancy #2, charge = -2
			SbVac1 = N_site2 * exp(-((1 * z[o][p][i] + 1.8304 + mu_Sb1[o][p]))/(k*T[i]))            	 #Sb vacancy, charge = 1
			Mgi2 = N_site1 * exp(-((2 * z[o][p][i] + 0.0389 - mu_Mg1[o][p])/(k*T[i])))              	 #Mg interstitial, charge = +2
			Mg_Sb3 =  N_site2 * exp(-((3 * z[o][p][i] + 1.658 - mu_Mg1[o][p] + mu_Sb1[o][p])/(k*T[i])))      #Mg_Sb anti-site, charge = +3
			Sb_1Mg1 = N_site1 * exp(-((1.347 + 1 * z[o][p][i] - mu_Sb1[o][p] + mu_Mg1[o][p])/(k*T[i])))   	 #Sb_Mg anti-site #1, charge = +1
			Sb_2Mg1 = N_site2 * exp(-((1.703 + 1 * z[o][p][i] - mu_Sb1[o][p] + mu_Mg1[o][p])/(k*T[i])))   	 #Sb_Mg anti-site #2, charge = +1
			Te_Sb1 = N_site2 * exp(-((1 * z[o][p][i] + 0.256 - mu_Te1[o][p] + mu_Sb1[o][p])/(k*T[i])))       #Te_Sb substitution, charge = 1

			Del_NMg = N_1 * (Mg1Vac_2 + Mg2Vac_2 + Sb_1Mg1 + Sb_2Mg1) + N1 * (Mgi2 + Mg_Sb3)
			Del_NSb = N_1 * (SbVac1 + Te_Sb1 + Mg_Sb3) + N1 * (Sb_1Mg1 + Sb_2Mg1)
			Del_NTe = N1 * (Te_Sb1)
			cSb[o][p][i] = (N_Sb0 + Del_NSb)/(N_Sb0 + Del_NSb + N_Mg0 + Del_NMg + N_Te0 + Del_NTe)          	#Composition of Sb
			cMg[o][p][i] = (N_Mg0 + Del_NMg)/(N_Sb0 + Del_NSb + N_Mg0 + Del_NMg + N_Te0 + Del_NTe)          	#Composition of Mg   	
			cTe[o][p][i] = (N_Te0 + Del_NTe)/(N_Sb0 + Del_NSb + N_Mg0 + Del_NMg + N_Te0 + Del_NTe)			#Composition of Te


print("Time to calculate electron chemical potential as a function of temperature: %s seconds " % (time.time() - start_time))

for m1 in range(1):
	pt = [[(0,0) for x in range(h1)] for t in range(g)]
	for o in range(g):                                                                            #for loop over the atomic chemical potential between different two-phase regions
		for p in range(h1):             							      #for loop over the atomic chemical potential in a two-phase region
			pt[o][p] = (cSb[o][p][(m1)]*100, cTe[o][p][(m1)]*100)
	list_of_lists = [pt[0][:], pt[1][:], pt[2][:]]
	points = [val for sublist in list_of_lists for val in sublist]



for l1 in range(1):
	mpl.rc('font', **{'sans-serif' : 'Arial',
                           'family' : 'sans-serif'})

	plt.rcParams['figure.figsize'] = 12,12  			#aspect ratio x,y
	plt.rcParams['pdf.fonttype'] = 42 				#This is how to make characters editable as text in PDF

	## Boundary and Gridlines
	scale = 1
	figure, tax = ternary.figure(scale=scale)

	# Draw Boundary and Gridlines
	tax.boundary(linewidth=2.0)

	# Set Axis labels and Title
	fontsize = 20

	# Set ticks
	tax.ticks(axis='lbr', linewidth=1, multiple=0.0002)

	#drawing custom straight line
	#(bottom, right) but (bottom, right, left) works as well. program just ignore "left"
	#in this case (Sb, Te, Mg)

	pt = [[(0,0) for x in range(h1)] for t in range(g)]
	for o in range(g):                                                                            #for loop over the atomic chemical potential between different two-phase regions
		for p in range(h1):             							      #for loop over the atomic chemical potential in a two-phase region
			pt[o][p] = (cSb[o][p][(l1)], cTe[o][p][(l1)])

	list_of_lists = [pt[0][:], pt[1][:], pt[2][:]]
	points = [val for sublist in list_of_lists for val in sublist]
	tax.plot(points, linewidth=3, linestyle="-",color='black')
	tax.legend()

	points4 = [(0.399999983319668, 4.25817879726861e-46),(0.399859535562667, 0.000140491766001806),(0.399990651711234, 1.26969691494223e-5),(0.400001416067918, 1.23815840487119e-48)]
	tax.legend()

	p15 = (0.399999983319668, 4.25817879726861e-46)
	p16 = (0.399859535562667, 0.000140491766001806)
	p17 = (0.399990651711234, 1.26969691494223e-5)
	p18 = (0.400001416067918, 1.23815840487119e-48)
	p19 = (0, 0)
	p20 = (0, 0.50)
	p21 = (1, 0)
	p22 = (0, 0.40024)
	p23 = (0.400, 0)
	p24 = (0.398, 0.002)
	p25 = (0.408, 0)
	p26 = (0, 0.408)
	p27 = (0.399998809, 0)
	p28 = (0.39833520, 0.002084456)	

	points1 = [p24]
	points2 = [pt[0][(h1-1)]]
	points3 = [pt[1][(h1-1)]]
	tax.plot(points1, marker='s', markersize=8.0, color='red')
	tax.plot(points2, marker='s', markersize=8.0, color='m')
	tax.plot(points3, marker='s', markersize=8.0, color='m')

	tax.line(pt[0][(h1-1)], p19, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[0][(h1-2)], p19, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[0][(h1-3)], p19, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[0][(h1-4)], p19, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[0][(h1-5)], p19, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[0][(h1-6)], p19, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[0][(h1-7)], p19, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[0][(h1-8)], p19, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[0][(h1-9)], p19, linewidth=1.0, color='g', linestyle="-") 
 
	tax.line(pt[0][(h1-1)], p20, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[1][(h1-1)], p20, linewidth=1.0, color='g', linestyle="-")
	tax.line(pt[1][(h1-1)], p21, linewidth=1.0, color='g', linestyle="-")
	tax.line(p22, p23, linewidth=1.0, color='r', linestyle="--")
	tax.line(p20, p23, linewidth=1.0, color='b', linestyle="--")
	tax.line(p27, p28, linewidth=1.0, color='c', linestyle="-")

	plt.text(13, 42, 'MgTe', fontsize=15)
	plt.text(4, 35, 'Mg'+ '$_3$' + 'Te'+'$_2$', fontsize=15)
	plt.text(32, -6.5, 'Mg'+ '$_3$' + 'Sb'+'$_2$', fontsize=15)

	tax.ax.set_xlim([0.4004, 0.3976])    
	tax.ax.set_ylim([0.0000, 0.003464102])   

	tax.ax.yaxis.set_ticks([0, 0.0008660254, 0.0017320508, 0.0025980762, 0.003464102])  
	tax.ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=20)

	tax.ax.xaxis.set_ticks([0.3976, 0.398, 0.3984, 0.3988, 0.3992, 0.3996, 0.4000, 0.4004])
	tax.ax.set_xticklabels([0.3976, 0.398, 0.3984, 0.3988, 0.3992, 0.3996, 0.4000, 0.4004], fontsize=20)

	tax.ax.set_xlabel('Mg %', fontsize=20)
	tax.ax.set_ylabel('Te %', fontsize=20)
	plt.show()

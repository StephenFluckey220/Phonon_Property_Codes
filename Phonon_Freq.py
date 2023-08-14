import numpy as np
import scipy
import matplotlib.pyplot as plt
import logging
import os
from struct import *
import shutil
#import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gs
import sys
from scipy.sparse.linalg import eigsh
from scipy import sparse
import cmath
import math
import matplotlib as mpl
from matplotlib import rcParams  
from matplotlib import rc
import vasprun as vaspr
import pymatgen as mg
import pymatgen.io.vasp as vasp    

if os.path.exists('./results'):
  shutil.rmtree('./results')

os.mkdir('./results')
logging.basicConfig(filename='./results/out.log',filemode='w',level=logging.DEBUG)
logger=logging.getLogger()

class Phonon_Effects: 
	def __init__(self,data,irrep_data,vasprun,vasp_data):#def __init__(self,data,dataPlus,dataMinus,data_lo,data_loPlus,data_loMinus,poscar,poscarPlus,poscarMinus,thickness) 
		self.data=data
		self.dynmat_data = data['phonon'][0]['dynamical_matrix']
		self.irrep_data=irrep_data
		self.irrep_data_character_table=[]
		self.irrep_data_nm=irrep_data['normal_modes']
		self.irrep_data_frequency=[]
		self.irrep_data_indices=[]
		if vasprun==True:
			self.vr=vasp.Vasprun(vasp_data)
			self.energies = self.vr.dielectric[0]
			self.e1 = np.array(self.vr.dielectric[1])
			self.e1x = self.e1[:,0]
			self.e1y = self.e1[:,1]
			self.e1z = self.e1[:,2]
			self.e2 = np.array(self.vr.dielectric[2])
			self.e2x = self.e2[:,0]
			self.e2y = self.e2[:,1]
			self.e2z = self.e2[:,2]
			self.etot = (self.e1x+self.e1y+self.e1z)/3
			self.etoti = (self.e2x+self.e2y+self.e2z)/3
		
		for l in range(len(irrep_data['normal_modes'])):
			
			self.irrep_data_character_table.append(irrep_data['normal_modes'][l]['ir_label'])
			self.irrep_data_frequency.append(irrep_data['normal_modes'][l]['frequency'])#check LO modes to see if irrep matrices with IR activity correspond to the same indices of frequency modes
			self.irrep_data_indices.append(irrep_data['normal_modes'][l]['band_indices'][:])
			for m in range(len(self.irrep_data_indices[l])):
				self.irrep_data_indices[l][m]=int(self.irrep_data_indices[l][m])-1			

		self.irrep_data_frequency=np.array(self.irrep_data_frequency,dtype=float)
		self.irrep_data_indices=(self.irrep_data_indices)
		self.irrep_data_irreps=irrep_data['irreps'] 

	def Dynamical_Matrix(self):
		dynmat = []
		
		#dynmat_data = data['phonon'][0]['dynamical_matrix']
		#print(dynmat_data)
		#print("width",len(dynmat_data[0][0,:]))
		#print("height",len(dynmat_data[0][:,0]))
		for row in self.dynmat_data:
	    		vals = np.reshape(row, (-1, 2))
	   
	    		dynmat.append(vals[:, 0].astype('complex128') + (vals[:, 1].astype('complex128') * 1j))
		dynmat = np.array(dynmat)
		return dynmat

	def CellRead(self,rf):
		comment = rf.readline().strip()
		latconst = float(rf.readline().split()[0])
		newcell = np.zeros((3, 3))

		newcell[0, :] = latconst * np.array([float(x) for x in rf.readline().split()[:3]])
		newcell[1, :] = latconst * np.array([float(x) for x in rf.readline().split()[:3]])
		newcell[2, :] = latconst * np.array([float(x) for x in rf.readline().split()[:3]])
		
		CellVolume=newcell[0,0]*newcell[1,1]*newcell[2,2]
		return CellVolume

	def Frequencies(self,dynmat,THZ=True):
		eigvals, eigvecs, = np.linalg.eigh(dynmat)
		#print(dynmat)
		#frequencies = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)
		conversion_factor_to_THz = 15.633302
		if THZ==True:
			OP_Frequencies=conversion_factor_to_THz*eigvals#frequencies 
		return OP_Frequencies

	def Eigenstates(self, dynmat):
		eigvals, eigvecs, = np.linalg.eigh(dynmat)
		Eigstates=eigvecs
		return Eigstates

	def Gruneseisen(self, Eigstates, deldynmat, delvolume, freq,THz=True):#need to change frequencies to resonances, accounting for complex part
		dDdV=np.zeros(len(delvolume))
		gruneisen=np.zeros(len(eigstates[0,:]))
		if THZ==True: 
			conversion_factor_to_THz = 15.633302
			freq=freq*conversion_factor_to_THz
			deldynmat=deldynmat*conversion_factor_to_THz
		freqRes=np.zeros(len(freq))
		
		for m in range(len(freq)):
			freqRes[m]=np.sqrt(freq[m].real**2+freq[m].imag**2) #resonances are the modulus of the imaginary part of the mode frequency
			
			
		dDdV=deldynmat[:,:]/delvolume
		print(np.shape(dDdV))
		for l in range(len(gruneisen)):
			gruneisen[l]=freqRes[l]/2*Eigstates[l].conj().transpose()*dDdV*Eigstates[l]
			print(gruneisen[l]/freq)
			
		
		return gruneisen

	def Gruneisenload(self, filename, frequency):
		gruneisen=2
		return gruneisen

	def VaspDielectric(self,plot=False):
		vaspeps=self.etot+1j*self.etoti
		vaspenergy=self.energies
		if plot==True:
			plt.figure()
			plt.title("Complex Dielectric Function")
			plt.xlabel("Energy")
			plt.ylabel("Dielectric function")
			plt.plot(np.array(self.energies)/6.28, self.etot, 'r', label = "Real")
			plt.plot(np.array(self.energies)/6.28, self.etoti, 'b', label = "Img")
			plt.xlabel('frequency (THz)')
			plt.legend()
			plt.grid('on')
			#plt.xlim(0,20)
			plt.minorticks_on()
			plt.savefig("Dielectric.pdf")

		return vaspeps, vaspenergy


	def DielectricFun(self,TOdamping, LOdamping ,TOModes, LOModes, freqRange,Epsilon0, THz=True,printEpsilonInfinity=False,conditionCheck=False,test=False):
		TOR=np.zeros(len(TOModes))
		LOR=np.zeros(len(LOModes))
		ResRatio=np.zeros(len(LOModes))
		DielectricShiftElementTO=np.zeros((len(TOR),len(freqRange)),dtype=complex)#needs to be a matrix which covers all frequencies on one axis and all resonances on the other 
		DielectricShiftElementLO=np.zeros((len(TOR),len(freqRange)),dtype=complex)
		#print(np.shape(DielectricShiftElementTO))	
		#check IR active vs raman active modes, for now do this in the run script not the object 	

		if conditionCheck==True:
			print('Lowndes condition check (Sum of LO damping-TO damping):',np.sum(LOdamping-TOdamping))
			if np.sum(LOdamping-TOdamping)<=0:
				print('Lowndes condition violation')
		for m in range(len(TOModes)):
			TOR[m]=np.sqrt(TOModes[m].real**2+TOModes[m].imag**2) #resonances are the modulus of the imaginary part of the mode frequency(purely real)
			LOR[m]=np.sqrt(LOModes[m].real**2+LOModes[m].imag**2) 
			ResRatio=LOR[m]**2/TOR[m]**2
			if conditionCheck==True:
				print('Physical meaning of Epsilon check:',LOdamping[m]/TOdamping[m]-(LOR[m]/TOR[m])**2)
				if LOdamping[m]/TOdamping[m]<(LOR[m]/TOR[m])**2:
					print('Error: Unphysical Epsilon (only necessarily valid for one mode pair)')
		epsilon=np.zeros(len(freqRange),dtype=complex)
		EpsilonInfinity=Epsilon0*1/(np.prod(ResRatio))
		if printEpsilonInfinity==True:
			print('High Frequency Dielectric Constant:', EpsilonInfinity)

		for l in range(len(freqRange)):
			DielectricShiftElementTOf=np.zeros(len(TOR),dtype=complex)
			DielectricShiftElementLOf=np.zeros(len(LOR),dtype=complex)
			DielectricShiftElementDiv=np.zeros(len(LOR),dtype=complex)

			for n in range(len(TOR)):
				DielectricShiftElementTOf[n]=complex(TOR[n]**2-freqRange[l]**2+1j*TOdamping[n]*freqRange[l])
				#print('TO Dielectric:',DielectricShiftElementTO[n])
				DielectricShiftElementLOf[n]=complex(LOR[n]**2-freqRange[l]**2+1j*LOdamping[n]*freqRange[l])
				#print('LO Dielectric:',DielectricShiftElementLO[n])
				DielectricShiftElementDiv[n]=complex(DielectricShiftElementLOf[n]/DielectricShiftElementTOf[n])
				
				'''DielectricShiftElementTO[l,n]=DielectricShiftElementTOf[n]
				DielectricShiftElementLO[l,n]=DielectricShiftElementLOf[n]
				print('TO mode contribution:',DielectricShiftElementTO[n,l])
				print('LO mode contribution:',DielectricShiftElementLO[n,l])'''

			DielectricShift=np.prod(DielectricShiftElementDiv)			

			epsilon[l]=EpsilonInfinity*DielectricShift
		return epsilon, DielectricShiftElementTO, DielectricShiftElementLO


#'''Remove absolute values from n and k when testing proper optical model from QE modes and Linewidths'''
	def Reflectance(self, DielectricFun, freqRange,test=False):
		n=np.zeros(len(DielectricFun))
		k=np.zeros(len(DielectricFun))
		n2=np.zeros(len(DielectricFun))
		k2=np.zeros(len(DielectricFun))
		R=np.zeros(len(DielectricFun))
		R2=np.zeros(len(DielectricFun))
		for l in range(len(n)):
			#n[l]=np.sqrt((np.sqrt(DielectricFun[l].real**2+DielectricFun[l].imag**2)+(DielectricFun[l].real))/2)
			n2[l]=np.sqrt(np.absolute((DielectricFun[l]).real))
			if test==True:
				print('index of refraction:',n[l],n2[l])
				print('Imaginary Dielectric Function:',DielectricFun[l].imag)
				print('Real Dielectric Function:',DielectricFun[l].real)
			#k[l]=np.sqrt(np.sqrt((DielectricFun[l].real**2+DielectricFun[l].imag**2)-(DielectricFun[l].real))/2)
			k2[l]=np.sqrt(np.absolute((DielectricFun[l]).imag))
			if test==True:
				print('extinction coeffiecient:',k[l],k2[l])
			R[l]=((n2[l]-1)**2+k2[l]**2)/((n2[l]+1)**2+k2[l]**2)
			R2[l]=np.sqrt(((np.sqrt(DielectricFun[l])-1)/(np.sqrt(DielectricFun[l])+1)).real**2+((np.sqrt(DielectricFun[l])-1)/(np.sqrt(DielectricFun[l])+1)).imag**2)
			if test==True:
				print('reflectance:',R[l])
				print('reflectance 2nd formulation:',R2[l])
			if R[l] > 1:
				print('Reflectance greater than 1')
		plt.figure(7)
		
		plt.plot(np.array(freqRange)/6.28,n2)
		plt.xlabel('frequency (THz)')
		plt.ylabel('$n$')
		plt.savefig('index of refraction.png',format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
		return R,n2,k2

	def Transmittance(self, DielectricFun, freqRange, thickness,test=False):
		n=np.zeros(len(DielectricFun),dtype=complex)
		k=np.zeros(len(DielectricFun),dtype=complex)
		T=np.zeros(len(DielectricFun))
		freqRange=np.array(freqRange)/(2*cmath.pi)
		for l in range(len(n)):
			n[l]=np.sqrt(np.absolute((DielectricFun[l]).real))
			#if n[l] < 1:
			#	print('index of refraction less than 1')
			#lambd=3*10**8/(freqRange[l]*10**12)#*n[l])#wavelength = velocity/frequency
			#print('Wavelength $\mu m$:',lambd/(10**-6))
			k[l]=np.sqrt(np.absolute((DielectricFun[l]).imag))
			
			if test==True:
				print('extinction coefficient:',k[l])
			T[l]=np.real(cmath.exp(-2*cmath.pi*k[l]*freqRange[l]*10**12*thickness/(3*10**8)))#np.real(cmath.exp(-4*cmath.pi*k[l]*thickness/lambd))
			if test==True:
				print('Transmittance:',T[l])
			if T[l] > 1:
				print('Transmittance greater than 1')
		print(len(k),len(freqRange))
		plt.figure(6)
		plt.plot(freqRange,k)
		plt.xlabel('frequency (THz)')
		plt.ylabel('$k$')
		plt.savefig('attenuation coefficient.png',format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)

		plt.figure(7)
		plt.plot(freqRange,(2*cmath.pi*k*freqRange*10**12)/(3*10**8))
		plt.xlabel('frequency (THz)')
		plt.ylabel('Absorption Coefficient')
		plt.ylim(0,20000)
		plt.savefig('absorption coefficient.png',format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
		
		return T,n,k

	def emissivity(self, freqRange,R,T,test=False):
		emiss=np.zeros(len(freqRange),dtype=float)
		for l in range(len(emiss)):
			T[l]=float(T[l])
			R[l]=float(R[l])		
			emiss[l]=((1-R[l])*(1-T[l]))/(1-T[l]*R[l])

		return emiss



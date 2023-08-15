import numpy as np
from numpy.linalg import solve,inv
import MultiphononParameters as mp
from MultiphononParameters import *
import cmath
import math
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import yaml
from yaml import load, dump, BaseLoader, SafeLoader
import os
import time
import Read_Phonon_Properties as rp
import DOS_Approx as DS
import Phonon_Dipole as PD

"""Calculate 2nd order dipole moments from Fourier transformed Born difference, and phonon branches"""
def get_ChiN(M,cell_divisions,meshdata,NCell,n=2,testEigenvector=False,testChi=False):
	ChiN=np.zeros((int(meshdata['nqpoint']),3*len(cell_divisions[0,:]),3*len(cell_divisions[0,:]),3),dtype=complex)
	print(np.shape(ChiN))
	weightsum=np.prod(np.array(meshdata['mesh'],dtype=int))
	"""get_ChiN calculates the dipole element in each cartesian direction by solving the following equation:"""
	"""M(q,v1,v2)=1/(2N sqrt(omega_v1 omega_v2))Sum_(atom1,atom2,direction1,direction2) M^(atom1,atom2)_(direction1,direction2)(q)e^atom1_direction1/sqrt(m_atom1)(q,v1) e_atom2_direction2(-q,v2)/sqrt(m_atom2)"""


	
	"""iterates over the considered atom, the displaced atom, and the direction of the electric field, contraction over other directions (and the atomic indices) is done via dot product"""
	for qphon in range(int(meshdata['nqpoint'])):
		for iphon, phoni in enumerate(meshdata['phonon'][qphon]['band']):
			for jphon, phonj in enumerate(meshdata['phonon'][qphon]['band'][:]):
		
				"""loads atomic eigenvectors for each phonon mode and combines real and imaginary parts into a single complex number"""
				Evec1=np.array(meshdata['phonon'][qphon]['band'][iphon]['eigenvector'],dtype=complex)
				Evec2=np.array(meshdata['phonon'][qphon]['band'][jphon]['eigenvector'],dtype=complex)
				eigenvectors=Evec1[:,:,0]+1j*Evec1[:,:,1]
				eigenvectorsneg=Evec2[:,:,0]+1j*Evec2[:,:,1]
				weight=int(meshdata['phonon'][qphon]['weight'])
				eigenvectorsneg=eigenvectorsneg.conj()

				#for EField_direction in range(len(M[0,0,0,:,0,0])):
				"""Summing over k1 and k2 with alpha1 and alpha2 being done via dot product"""
				for displaced_atom in range(len(M[0,0,:,0,0,0])):
					for atom in range(len(M[0,:,0,0,0,0])):
						magnitudeAtom=1
					
						"""gets e^atom1_direction1/sqrt(m_atom1)(q,v1) and e_atom2_direction2(-q,v2)/sqrt(m_atom2)"""
						magnitudeDisplaced=1
						#print(np.array(meshdata['points'][displaced_atom]['coordinates'],dtype=float))
						#print(	np.array(meshdata['phonon'][qphon]['q-position'],dtype=float)	)
						eigenvectors[atom]=eigenvectors[atom]/(magnitudeAtom)#*np.exp(1j*np.array(meshdata['phonon'][qphon]['q-position'],dtype=float).dot(np.array(meshdata['points'][atom]['coordinates'],dtype=float)))
						eigenvectorsneg[displaced_atom]=eigenvectorsneg[displaced_atom]/(magnitudeDisplaced)#*np.exp(-1j*np.array(meshdata['phonon'][qphon]['q-position'],dtype=float).dot(np.array(meshdata['points'][displaced_atom]['coordinates'],dtype=float)))					
					
						if testEigenvector==True:
							print(atom+1,'atom eigenvector:',eigenvectors[atom],'mass:',meshdata['points'][atom]['mass'])
							print(displaced_atom+1,'displaced eigenvector:',eigenvectors[displaced_atom],'mass:',meshdata['points'][displaced_atom]['mass'])
							print('M element atoms direction/N',':',atom,displaced_atom,':',M[qphon,atom,displaced_atom,:,:,:]/len(cell_divisions[:,0]))
							time.sleep(0.5)
					
						"""performs dot products M^(atom1,atom2)_(direction1,direction2)(q)e^atom1_direction1/sqrt(m_atom1)(q,v1) e_atom2_direction2(q,v2)/sqrt(m_atom2)"""
						"""and adds contribution from atom1 atom2, direction1, direction2 to total for q, v1, v2"""
						ChiN[qphon,iphon,jphon,:]+=(M[qphon,atom,displaced_atom,:,:,:].dot(eigenvectors[atom]/math.sqrt(float(meshdata['points'][atom]['mass'])*amu))).dot(					eigenvectorsneg[displaced_atom]/math.sqrt(float(meshdata['points'][displaced_atom]['mass'])*amu))
					

						if testEigenvector==True:
							"""Frequency prefactor verified accurate"""
							print('Frequencies:',iphon,float(phoni['frequency']),jphon,float(phonj['frequency']))
							print('Frequency Prefactor:',1/(2*NCell*2*cmath.pi*math.sqrt(np.absolute(float(phoni['frequency'])*float(phonj['frequency'])))*THz)	)								
				
						if testChi==True:
							Chidotone=M[qphon,atom,displaced_atom,:,:,:].dot(eigenvectors[atom]/math.sqrt(float(meshdata['points'][atom]['mass'])*amu))
							Chidottwo=Chidotone.dot(eigenvectorsneg[displaced_atom]/math.sqrt(float(meshdata['points'][displaced_atom]['mass'])*amu))
							print('Product of Fourier Transformed Dipole moment element and first eigenvector',iphon+1,jphon+1,Chidotone	)
							print('Product of Fourier Transformed Dipole moment element and eigenvectors',1/(2*NCell*len(cell_divisions[:,0])*2*cmath.pi*math.sqrt(np.absolute(float(phoni['frequency'])*float(phonj['frequency'])))*THz)*Chidottwo,"\n"	) 
				
				"""multiplies the dipole element defined by q, v1, v2 by 1/(2N sqrt(omega_v1 omega_v2))Sum_(atom1,atom2,direction1,direction2)"""
				ChiN[qphon,iphon,jphon,:]=1/(2*NCell*2*cmath.pi*math.sqrt(np.absolute(float(phoni['frequency'])*float(phonj['frequency'])))*THz)*ChiN[qphon,iphon,jphon,:]
				if testChi==True:
					print('Sum of products of Fourier Transformed Dipole moment element and eigenvectors with frequency prefactor',iphon,jphon,ChiN[qphon,iphon,jphon,:],"\n")
					time.sleep(0.75)
	#print(ChiN)
	return ChiN


def get_Imag_Susceptibility(Chi,meshdata,Temperature,freqRange,alpha,shift_factor,NCell=1,freqUnit=THz,n=2,LinewidthSimulate=False,testQpoint=False):
	Imag_Susceptibility=np.zeros((len(freqRange),3))
	EigenLength=(len(meshdata['phonon'][0]['band']))
	ChiMod=np.zeros((EigenLength,EigenLength,3))
	DOSDelta=np.zeros((EigenLength,EigenLength,3,len(freqRange)))
	DeltaSumMod=np.zeros((len(freqRange)))

	"""This function calculates the imaginary part of the frequency dependent susceptibility by the following relation:"""
	"""Chi_alpha(omega)=pi/2 Sum_(q,v1,v2) |M_alpha(q,v1,v2)|^2 Sum_(pm) delta(omega_v1 pm omega_v2 -omega)[(nBE(q,v2)+1/2) pm (nBE(q,v1)+1/2)]"""

	T=Temperature
	weightsum=np.prod(np.array(meshdata['mesh'],dtype=int))
	Chielem=np.zeros((weightsum,EigenLength,EigenLength,3),dtype=complex)	
	Q=[]
	Qdist=[]

	Imag_susceptibility_Tensor=np.zeros((int(meshdata['nqpoint']),EigenLength,EigenLength,3),dtype=float)
	Q=[]
	Qdist=[]
	Edist=np.zeros((weightsum,EigenLength,EigenLength,len(freqRange)-1),dtype=float)
	Omega1dist=np.zeros((weightsum,EigenLength),dtype=float)
	Omega2dist=np.zeros((weightsum,EigenLength),dtype=float)
		
	"""Frequencies are updated for each q-point, but chi still needs to be specified at a q-point"""
	"""For systems with significant non-cubic contributions, sum must be taken for all direction combinations, not just alpha-alpha (handbook der physik)"""

	"""Sums over q,v1,v2 to obtain susceptibility as a function of frequency"""
	if n==2:
		for nqpoint, qpointn in enumerate(meshdata['phonon']):# for silicon at least, the opposite q-points have the same frequency modes, eigenvectors are complex conjugate 
			Frequencies=[]
			DeltaSum=np.zeros((len(freqRange)))#delta function term with bose einstein functions for phonon branches as q ,(i,j)
			ChiMod=np.zeros((EigenLength,EigenLength,3))#Dipole moment element for q-point	

			Q.append(qpointn['q-position'])
			Qdist.append(qpointn['distance_from_gamma'])
	

			if testQpoint==True:
				print('q-point number:',nqpoint)
			for iphon, phoni in enumerate(meshdata['phonon'][nqpoint]['band']):
				for jphon, phonj in enumerate(meshdata['phonon'][nqpoint]['band']):
					"""check added since overtones should not contribute to IR spectra (literature)"""
					if iphon!=jphon:

						"""Calculated |M(q,v1,v2)|^2 contribution to q,v1,v2 summation term"""
						Chielem[nqpoint,iphon,jphon,:]=((Chi[nqpoint,iphon,jphon,:].real)**2+(Chi[nqpoint,iphon,jphon,:].imag)**2)
						ChiMod[iphon,jphon,:]=((Chi[nqpoint,iphon,jphon,:].real)**2+(Chi[nqpoint,iphon,jphon,:].imag)**2)
						if testQpoint==True:
							print('frequency 1:',float(phoni['frequency']),'frequency 2:',float(phonj['frequency']),'Bose Einstein Term:',DS.BEFunc(float(phonj['frequency'])*THz,T),DS.BEFunc(float(phoni['frequency'])*THz,T))	
							print('Dipole element for q,v1,v2:',ChiMod[iphon,jphon,:])

						#factors of 2pi in the DOS and Bose Einstein functions need to be investigated to ensure proper implementation 

						"""Calculates Sum_(pm) delta(omega_v1 pm omega_v2 -omega)[(nBE(q,v2)+1/2) pm (nBE(q,v1)+1/2)]"""
						deltaplus=np.array(DS.diracApprox(alpha[nqpoint,iphon,jphon],float(phoni['frequency'])*THz+float(phonj['frequency'])*THz-freqRange))						*((DS.BEFunc(float(phonj['frequency'])*THz,T)+1/2)+(DS.BEFunc(float(phoni['frequency'])*THz,T)+1/2))*int(qpointn['weight'])#/weightsum

						deltaminus=np.array(DS.diracApprox(alpha[nqpoint,iphon,jphon],float(phoni['frequency'])*THz-float(phonj['frequency'])*THz-freqRange))						*((DS.BEFunc(float(phonj['frequency'])*THz,T)+1/2)-(DS.BEFunc(float(phoni['frequency'])*THz,T)+1/2))*int(qpointn['weight'])#/weightsum
						"""Factor of 2pi to account for leaving it out of the delta function terms"""	
						DeltaSum=deltaplus/(2*cmath.pi)+deltaminus/(2*cmath.pi)	

						Imag_susceptibility_Tensor[nqpoint,iphon,jphon,0]=ChiMod[iphon,jphon,0]*max(DeltaSum-deltaminus/(2*cmath.pi))
						Imag_susceptibility_Tensor[nqpoint,iphon,jphon,1]=ChiMod[iphon,jphon,1]*max(DeltaSum-deltaminus/(2*cmath.pi))
						Imag_susceptibility_Tensor[nqpoint,iphon,jphon,2]=ChiMod[iphon,jphon,2]*max(DeltaSum-deltaminus/(2*cmath.pi))


						"""term to deal with comparing DOS between smearing and tetrahedron"""
						DeltaSumMod+=DS.get_delta_function_sum(np.array([float(phoni['frequency'])*freqUnit,float(phonj	['frequency'])*freqUnit]),freqRange,alpha[nqpoint,iphon,jphon])*int(qpointn['weight'])/weightsum	
						
						"""Combining dipole moments and relevant delta function term for q,i,j"""
						ChiSum1=ChiMod[iphon,jphon,0]*DeltaSum
						ChiSum2=ChiMod[iphon,jphon,1]*DeltaSum
						ChiSum3=ChiMod[iphon,jphon,2]*DeltaSum
					
						"""Contributing q,i,j term to the overall susceptibility"""
						Imag_Susceptibility[:,0]+=(ChiSum1)
						Imag_Susceptibility[:,1]+=(ChiSum2)
						Imag_Susceptibility[:,2]+=(ChiSum3)

						'''n=115
						Imag_susceptibility_Tensor[nqpoint,iphon,jphon,0,:]=Chielem[nqpoint,iphon,jphon,0]*(DeltaSum-deltaminus/(2*cmath.pi))
						Imag_susceptibility_Tensor[nqpoint,iphon,jphon,1,:]=Chielem[nqpoint,iphon,jphon,1]*(DeltaSum-deltaminus/(2*cmath.pi))
						Imag_susceptibility_Tensor[nqpoint,iphon,jphon,2,:]=Chielem[nqpoint,iphon,jphon,2]*(DeltaSum-deltaminus/(2*cmath.pi))'''

						Omega1dist[nqpoint,iphon]=(float(phoni['frequency'])*THz)/(freqRange[int(np.where(DeltaSum==max(DeltaSum))[0][0])])
						Omega2dist[nqpoint,jphon]=(float(phoni['frequency'])*THz)/(freqRange[int(np.where(DeltaSum==max(DeltaSum))[0][0])])
						Edist[nqpoint,iphon,jphon,:]=(float(phoni['frequency'])*THz+float(phonj['frequency'])*THz)/(freqRange[1:])
						#print(Edist[nqpoint,iphon,jphon,:])

	#plt.plot(freqRange/THz,DeltaSumMod*THz)
	#plt.plot(freqRange/THz,shift_factor/THz)
	#plt.show()
	print('Cubic dipole element:',Chielem.sum(axis=0).sum(axis=0).sum(axis=0)/weightsum)
	if LinewidthSimulate==True:
		Imag_Susceptibility[:,0]=cmath.pi*Imag_Susceptibility[:,0]/2
		Imag_Susceptibility[:,1]=cmath.pi*Imag_Susceptibility[:,1]/2
		Imag_Susceptibility[:,2]=cmath.pi*Imag_Susceptibility[:,2]/2
	else:
		Tetrahedral_Shift=(shift_factor/(DeltaSumMod))/THz	
		Tetrahedral_Shift[np.isnan(Tetrahedral_Shift)]=0	

		Imag_Susceptibility[:,0]=cmath.pi*Imag_Susceptibility[:,0]*Tetrahedral_Shift/(2)
		Imag_Susceptibility[:,1]=cmath.pi*Imag_Susceptibility[:,1]*Tetrahedral_Shift/(2)
		Imag_Susceptibility[:,2]=cmath.pi*Imag_Susceptibility[:,2]*Tetrahedral_Shift/(2)
	#print('TDOS Test:',((DeltaSum/magnitude)/TDOS))	
	#print('Modified TDOS magnitude:',scipy.integrate.simps(DeltaSum/magnitude,freqRange))	

	Q=np.array(Q,dtype=float)
	Qdist=np.array(Qdist,dtype=float)
	dist_Indices=np.argsort(Qdist)
	Qdist=Qdist[dist_Indices]

	bin_number=21	

	BinWidthomega=(2*38.6800606247)/bin_number

	BinWidthomega1=1/bin_number
	
	omegabinlist = [[] for _ in range(bin_number)]
	omegabinlistarr = np.empty(bin_number, object)
	omegabinlistarr[:]= omegabinlist

	omega1binlist = [[] for _ in range(bin_number)]	
	omega1binlistarr = np.empty(bin_number, object)
	omega1binlistarr[:]= omega1binlist

	SusceptBin=np.zeros((bin_number,bin_number),dtype=float)

	for bin_number_omega1 in range(bin_number):
		for bin_number_omega in range(bin_number):
			for nqpoint, qpointn in enumerate(meshdata['phonon']):
				nmode=0
				for iphon, phoni in enumerate(meshdata['phonon'][nqpoint]['band']):
					for jphon, phonj in enumerate(meshdata['phonon'][nqpoint]['band']):
						photon_freq=float(phoni['frequency'])+float(phonj['frequency'])
						phonon1_fraction=float(phoni['frequency'])/photon_freq
						Suscept_Elem=Imag_susceptibility_Tensor[nqpoint,iphon,jphon,0]
				
						#print('photon',BinWidthomega*bin_number,photon_freq)
						if photon_freq > BinWidthomega*bin_number_omega and photon_freq < BinWidthomega*(bin_number_omega+1):
							omegabinlist[bin_number_omega].append(photon_freq)
							SusceptBin[bin_number_omega,bin_number_omega1]+=Suscept_Elem

						if phonon1_fraction > BinWidthomega1*(bin_number_omega1) and phonon1_fraction < BinWidthomega1*(bin_number_omega1+1):
							omega1binlist[bin_number_omega1].append(phonon1_fraction)
							SusceptBin[bin_number_omega,bin_number_omega1]+=Suscept_Elem

			#omegabinlist[bin_number_omega]=np.array(omegabinlist[bin_number_omega],dtype=float)
		#omega1binlist[bin_number_omega1]=np.array(omega1binlist[bin_number_omega1],dtype=float)
	contouromega=np.linspace(0,2*38.6800606247,bin_number)
	contouromega1=np.linspace(0,1,bin_number)
	
	SusceptSum=SusceptBin.sum(axis=1)
	print((SusceptBin/SusceptSum[:,np.newaxis]).sum(axis=1))

	fig, ax = plt.subplots(figsize=(3,2))
	plt.contourf(contouromega1,contouromega*THz/(100*c),SusceptBin/SusceptSum[:,np.newaxis])
	cbar=plt.colorbar()
	cbar.set_label('Im($\chi_x$)')
	plt.xlabel('$\omega_1/\omega$')
	plt.ylabel('$\omega~(\mathrm{cm}^{-1})$')
	plt.show()
	fig.savefig('DiamondModeContributionContour.pdf',format='pdf',dpi=400,bbox_inches='tight',pad_inches=0.2)
	
	#plt.plot(freqRange/THz,Imag_Susceptibility[:,0])
	#print(np.shape(Imag_susceptibility_Tensor))
	#plt.plot(freqRange/THz,cmath.pi/2*Imag_susceptibility_Tensor.sum(axis=0).sum(axis=0).sum(axis=0)[0,:],color='r')
	#plt.show()


	Imag_Susceptibility_Tensor=np.transpose(Imag_susceptibility_Tensor,(0,1,2,3))

	Zs=np.zeros((weightsum,EigenLength*EigenLength),dtype=float)
	photonenergy=np.zeros((weightsum,EigenLength*EigenLength),dtype=float)
	phononenergy1Fraction=np.zeros((weightsum,EigenLength*EigenLength),dtype=float)
	fraclist=[]	
	#BinWidth=(max(Imag_suscepbibility_Tensor)-min(Imag_susceptibility_Tensor))/10

	for nqpoint, qpointn in enumerate(meshdata['phonon']):
		nmode=0
		for iphon, phoni in enumerate(meshdata['phonon'][nqpoint]['band']):
			for jphon, phonj in enumerate(meshdata['phonon'][nqpoint]['band']):
				if float(phoni['frequency']) > 0 and float(phonj['frequency'])>0:
					photonenergyelem=(float(phoni['frequency'])+float(phonj['frequency']))
					phononenergy1FracElem=float(phoni['frequency'])/photonenergyelem
					#print('mode 1 energy fraction:',phononenergy1FracElem)
					photonenergy[nqpoint,nmode]=photonenergyelem
					phononenergy1Fraction[nqpoint,nmode]=phononenergy1FracElem					

					fraclist.append(phononenergy1FracElem)
					Zs[nqpoint,nmode]=577.6386187400972*cmath.pi/2*Imag_susceptibility_Tensor[nqpoint,iphon,jphon,0]	
					nmode+=1
		#modesort=np.argsort(photonenergy[nqpoint,:])
		#photonenergy[nqpoint,:]=photonenergy[nqpoint,modesort[1]]
		#phononenergy1Fraction[nqpoint,:]=phononenergy1Fraction[nqpoint,modesort[1]]
		plt.scatter(phononenergy1Fraction[nqpoint,:],photonenergy[nqpoint,:],c=Zs[nqpoint,:],s=10)
			
		plt.xlabel('$\omega_1/\omega$')
		plt.ylabel('$\omega$')
					
	#print(np.argsort(photonenergy,axis=1))
	#print(photonenergy[0,np.argsort(photonenergy)],phononenergy1Fraction[0,np.argsort(photonenergy)])
	
	cbar=plt.colorbar()
	cbar.set_label('Im($\chi_x(\omega_1(\mathbf{q}),\omega_2(\mathbf{q}),\omega)$)')	
	plt.show()
	plt.figure(1)


	'''for i in range(EigenLength):
		for j in range(EigenLength):
			plt.scatter(Edist[:,i,j,:],cmath.pi/2*Imag_Susceptibility_Tensor[:,i,j,1:,0])
			plt.xlabel('phonon energy/photon energy')
			plt.ylabel('$\chi''$'+str('(')+str(i)+str(')'))
			#plt.ylim(0,3*10**-5)
			plt.xlim(0,2)
			plt.title(str('xyz'))

			plt.show()

	plt.figure(2)
	for i in range(EigenLength):
		for j in range(EigenLength):
			plt.scatter(Edist[:,i,j,:],506.56805156524547*cmath.pi/2*Imag_Susceptibility_Tensor[:,i,j,1:,1])
			plt.xlabel('phonon energy/photon energy')
			plt.ylabel('$\chi''$'+str('(')+str(i)+str(')'))
			#plt.ylim(0,3*10**-5)
			plt.xlim(0,2)
			plt.title(str('y'))

			plt.show()

	plt.figure(3)
	for i in range(EigenLength):
		for j in range(EigenLength):
			plt.scatter(Edist[:,i,j,:],506.56805156524547*cmath.pi/2*Imag_Susceptibility_Tensor[:,i,j,1:,2])
			plt.xlabel('phonon energy/photon energy')
			plt.ylabel('$\chi''$'+str('(')+str(i)+str(')'))
			#plt.ylim(0,3*10**-5)
			plt.xlim(0,2)
			plt.title(str('z'))

			plt.show()'''

	
	

	


	Susceptmax=Imag_susceptibility_Tensor[:,1,5,:].sum(axis=1)
	print(np.shape(Susceptmax))

	Susceptmax=506.56805156524547*cmath.pi/2*Susceptmax[dist_Indices]

	print('Susceptibility Sum:',np.sum(Susceptmax))

	plt.plot(Qdist,Susceptmax)
	plt.show()

	fig=plt.figure(figsize = (3,2))
	plt.plot(Susceptmax)
	ax = fig.add_subplot(projection='3d')
	
	img=ax.scatter(Q[:,0],Q[:,1],Q[:,2],c=Susceptmax,cmap=plt.hot())
	fig.colorbar(img)
	ax.set_xlabel('$\mathbf{Q}_x$')
	ax.set_ylabel('$\mathbf{Q}_y$')
	ax.set_zlabel('$\mathbf{Q}_z$')

	plt.show()

	
	return Imag_Susceptibility



def get_ChiN_single_phonon(M,cell_divisions,meshdata,NCell=1,testEigenvector=False,testChi=False):
	ChiN=np.zeros((1,3*len(cell_divisions[0,:]),3),dtype=complex)
	print('Single phonon M phonon branch shape:',np.shape(ChiN))#1 qpoint, 3Natom modes, 3 directions
 
	distance_from_gamma=[]
	for nqpoint, qpointn in enumerate(meshdata['phonon']):
		distance_from_gamma.append(meshdata['phonon'][nqpoint]['distance_from_gamma'])
	distance_from_gamma=np.array(distance_from_gamma)
	mindistanceindex=int(np.where(distance_from_gamma==min(distance_from_gamma))[0][0])
	print(distance_from_gamma[mindistanceindex],mindistanceindex)
	GammaEigenvectors=np.zeros(	(3*len(cell_divisions[0,:]),len(cell_divisions[0,:]),3),dtype=complex)
	GammaBorn=np.zeros((len(cell_divisions[0,:]),3,3),dtype=complex)

	"""iterates over the considered atom, the displaced atom, and the direction of the electric field, contraction over other directions (and the atomic indices) is done via dot product"""
	
	for iphon, phoni in enumerate(meshdata['phonon'][mindistanceindex]['band']):
		Evec=np.array(meshdata['phonon'][mindistanceindex]['band'][iphon]['eigenvector'],dtype=complex)
		eigenvectors=Evec[:,:,0]+1j*Evec[:,:,1]
		weight=int(meshdata['phonon'][mindistanceindex]['weight'])
		eigenvectorsneg=eigenvectors.conj()
		print(iphon+1,eigenvectors)
		
		for atom in range(len(M[0,:,0,0])):#sum over k_1
			#print(M[mindistanceindex,atom,:,:])
			GammaBorn[atom,:,:]=M[mindistanceindex,atom,:,:]
		
			for EField_direction in range(len(M[0,0,:,0])):#sum over alpha
				#if testEigenvector==True:
				#	print('Eigenvector for atoms ',atom+1,':',eigenvectors[atom],'mode:',iphon+1)
					#print('Orthogonality Check:',eigenvectorsneg[atom].dot(eigenvectors[atom+1]))
					#np.savez(str('./Data_Files/GaN_Born_and_Eigenvector/phononEigenvectorMode'+str(iphon)+str('.npz')),eigenvectors,'eigenvector')
				eigenvectors[atom]=eigenvectors[atom]
				GammaEigenvectors[iphon,atom,:]=eigenvectors[atom]
				
			
				EvecMass=eigenvectors[atom]/math.sqrt(float(meshdata['points'][atom]['mass'])*amu)					

				if testEigenvector==True:
					print('atom scaled eigenvector',atom+1,':',eigenvectors[atom]/math.sqrt(float(meshdata['points'][atom]['mass'])*amu),'mode:',iphon+1, 'frequency:',np.absolute(float(phoni['frequency'])))
					#print('M element:',M[mindistanceindex,atom,EField_direction,:])
					#np.savez(str('./Data_Files/GaN_Born_and_Eigenvector/phononDipoleElement'+str('.npz')),M,'BornCharge')				
	
				ChiN[0,iphon,EField_direction]+=(M[mindistanceindex,atom,EField_direction,:].dot(EvecMass)	)
				
				if testChi==True:
					print('mass of atom:',float(meshdata['points'][atom]['mass']))
				
		
		#print('Product of Dipole moment element and eigenvectors ',ChiN[0,iphon,:],'Frequency:',float(phoni['frequency'])	)
		#print('Frequency Prefactor:',1/np.sqrt(hbar*2*len(cell_divisions[:,0])*2*cmath.pi*np.absolute(float(phoni['frequency']))*THz)	)
		
		ChiN[0,iphon,:]=1/np.sqrt(hbar*2*len(cell_divisions[:,0])*2*cmath.pi*np.absolute(float(phoni['frequency']))*THz)*ChiN[0,iphon,:]
		

		#if np.sum(ChiN[0,iphon,:])/3<10**2:
			#ChiN[0,iphon,:]=np.zeros(3)
	#np.savez(str('./Data_Files/Born_and_Eigenvector/phononEigenvectorMode'+str(iphon)+str('.npz')),GammaEigenvectors,'eigenvector')
	#np.savez(str('./Data_Files/Born_and_Eigenvector/phononDipoleElement'+str('.npz')),GammaBorn,'BornCharge')

	return ChiN


def get_Imag_Susceptibility_single_phonon_linewidth(Chi,meshdata,Linewidth,freqRange,alpha,shift_factor,NCell=1,freqUnit=THz,testQpoint=False):
	Imag_Susceptibility=np.zeros((len(freqRange),3),dtype=complex)
	
	#EigenLength=6	
	EigenLength=(len(meshdata['phonon'][0]['band']))

	ChiMod=np.zeros((EigenLength,3))
	ChiMod2=np.zeros((EigenLength,3,len(freqRange)))

	DOSDelta=np.zeros((EigenLength,3,len(freqRange)),dtype=complex)
	DeltaSumMod=np.zeros((len(freqRange)),dtype=complex)
	DeltaSum=np.zeros((len(freqRange)),dtype=complex)
	weightsum=np.prod(np.array(meshdata['mesh'],dtype=int))
	

	distance_from_gamma=[]
	for nqpoint, qpointn in enumerate(meshdata['phonon']):
		distance_from_gamma.append(meshdata['phonon'][nqpoint]['distance_from_gamma'])
	distance_from_gamma=np.array(distance_from_gamma)
	mindistanceindex=int(np.where(distance_from_gamma==min(distance_from_gamma))[0][0])
	print(distance_from_gamma[mindistanceindex],mindistanceindex)
		
	"""For non-cubic systems, sum must be taken for all direction combinations, not just alpha-alpha (handbook der physik)"""
	
	Frequencies=[]
	ChiMod=np.zeros((EigenLength,3),dtype=complex)
	for b in range(len(meshdata['phonon'][mindistanceindex]['band'])):
		Frequencies.append(float(meshdata['phonon'][mindistanceindex]['band'][b]['frequency'])*THz)
	Frequencies=np.array(Frequencies)	
	deltatest=np.zeros(len(freqRange),dtype=complex)
	ChiModSum=np.zeros((EigenLength, 3,3),dtype=complex)	
	
	for iphon, phoni in enumerate(meshdata['phonon'][mindistanceindex]['band']):
			
			ChiMod[iphon,:]=Chi[0,iphon,:].real**2+Chi[0,iphon,:].imag**2
			delta=2*(2*cmath.pi*float(phoni['frequency'])*THz	)/((2*cmath.pi*float(phoni['frequency'])*THz)**2-(2*cmath.pi*freqRange+1j*2*cmath.pi*Linewidth)**2)
			print('Dipole modulus for mode',float(phoni['frequency']),':',ChiMod[iphon,:])	

			DeltaSum=delta			
			deltatest+=delta
			for xs in range(3):
				for ys in range(3):
					ChiModSum[iphon,xs,ys]=np.conj(Chi[0,iphon,xs])*(Chi[0,iphon,ys])#Chi[0,iphon,:].real**2+Chi[0,iphon,:].imag**2
			
				
			"""Combining dipole moments and relevant delta function term for q,i,j"""
			ChiSum1=ChiMod[iphon,0]*2*(2*cmath.pi*float(phoni['frequency'])*THz	)/((2*cmath.pi*float(phoni['frequency'])*THz)**2-(2*cmath.pi*freqRange+1j*2*cmath.pi*Linewidth)**2)#DeltaSum
			ChiSum2=ChiMod[iphon,1]*2*(2*cmath.pi*float(phoni['frequency'])*THz	)/((2*cmath.pi*float(phoni['frequency'])*THz)**2-(2*cmath.pi*freqRange+1j*2*cmath.pi*Linewidth)**2)#DeltaSum
			ChiSum3=ChiMod[iphon,2]*2*(2*cmath.pi*float(phoni['frequency'])*THz	)/((2*cmath.pi*float(phoni['frequency'])*THz)**2-(2*cmath.pi*freqRange+1j*2*cmath.pi*Linewidth)**2)#DeltaSum
						
			"""Contributing q,i,j term to the overall susceptibility"""
			Imag_Susceptibility[:,0]+=ChiSum1
			Imag_Susceptibility[:,1]+=ChiSum2
			Imag_Susceptibility[:,2]+=ChiSum3
			

	"""Normalizing density of states with respect to density of states without BE functions"""					
	
	#for l in range(len(freqRange)):
	#	print('Imag from loop',cmath.pi/2*Imag_Susceptibility[l,2].imag, 'frequency:',freqRange[l]/THz)#9642.583305005339	

	print('Dipole diagonal:',ChiMod.sum(axis=0))
	print('Sum of dipole element contributions:', ChiModSum.sum(axis=0))
	Imag_Susceptibility[:,0]=Imag_Susceptibility[:,0]
	Imag_Susceptibility[:,1]=Imag_Susceptibility[:,1]
	Imag_Susceptibility[:,2]=Imag_Susceptibility[:,2]
	#print('TDOS Test:',((DeltaSum/magnitude)/TDOS))	
	#print('Modified TDOS magnitude:',scipy.integrate.simps(DeltaSum/magnitude,freqRange))	
	

	Imag_Susceptibility_Single_Phonon=np.imag(Imag_Susceptibility)

	#for l in range(len(freqRange)):
	#	print('Imag final:',17279.382405959874*Imag_Susceptibility_Single_Phonon[l,2], 'frequency:',freqRange[l]/THz)

	#plt.plot(freqRange/THz,deltatest.real)
	#plt.plot(freqRange/THz,deltatest.imag)
	#plt.show()	

	return Imag_Susceptibility_Single_Phonon



def get_Imag_Susceptibility_NonCubic(Chi,meshdata,Temperature,freqRange,alpha,shift_factor,freqUnit=THz,testQpoint=False,LinewidthSimulate=False):
	Imag_Susceptibility=np.zeros((len(freqRange),3,3),dtype=complex)
	EigenLength=(len(meshdata['phonon'][0]['band']))
	ChiMod=np.zeros((EigenLength,EigenLength,3,3))
		
	T=Temperature
	weightsum=np.prod(np.array(meshdata['mesh'],dtype=int))
	print('q-points:',weightsum)
	Chielem=np.zeros((weightsum,EigenLength,EigenLength,3,3),dtype=complex)
	
	Imag_Susceptibility_Tensor=np.zeros((int(meshdata['nqpoint']),EigenLength,EigenLength,3,3,len(freqRange)),dtype=float)
	Imag_susceptibility_Tensor=np.zeros((int(meshdata['nqpoint']),EigenLength,EigenLength,3,3),dtype=float)
	Q=[]
	Qdist=[]
	Edist=np.zeros((weightsum,EigenLength,EigenLength),dtype=float)
		
	"""Frequencies are updated for each q-point, but chi still needs to be specified at a q-point"""
	"""For non-cubic systems, sum must be taken for all direction combinations, not just alpha-alpha (handbook der physik)"""

	DeltaSumMod=np.zeros((len(freqRange)))
	for nqpoint, qpointn in enumerate(meshdata['phonon']):

		Q.append(qpointn['q-position'])
		Qdist.append(qpointn['distance_from_gamma'])

		for iphon, phoni in enumerate(meshdata['phonon'][nqpoint]['band'][:]):
			for jphon, phonj in enumerate(meshdata['phonon'][nqpoint]['band'][:]):
				if iphon != jphon:
					#ChiMod=np.zeros((EigenLength,EigenLength,3,3),dtype=complex)
					#if testQpoint==True:
					#	print('q-point number:',nqpoint)
					
					"""term to deal with normalizing DOS"""	
					DeltaSumMod+=DS.get_delta_function_sum(np.array([float(phoni['frequency'])*freqUnit,float(phonj	['frequency'])*freqUnit]),freqRange,alpha)*int(qpointn['weight'])/weightsum

					#print('Q-point',meshdata['phonon'][nqpoint]['q-position'],'Frequency 1:',phoni['frequency'],'Frequency 2',phonj['frequency'])
					#print('Dipole Term:',np.conj(Chi[nqpoint,iphon,jphon,:,None])*(Chi[nqpoint,iphon,jphon,None,:]))

					deltaplus=np.array(DS.diracApprox(alpha,float(phoni['frequency'])*THz+float(phonj['frequency'])*THz-freqRange))*((DS.BEFunc(float(phonj['frequency'])*THz,T)+1/2)+(DS.BEFunc(float(phoni['frequency'])*THz,T)+1/2))*int(qpointn['weight'])#/weightsum

					deltaminus=np.array(DS.diracApprox(alpha,float(phoni['frequency'])*THz-float(phonj['frequency'])*THz-freqRange))*((DS.BEFunc(float(phonj['frequency'])*THz,T)+1/2)-(DS.BEFunc(float(phoni['frequency'])*THz,T)+1/2))*int(qpointn['weight'])#/weightsum
						
					DeltaSum=deltaplus/(2*cmath.pi)+deltaminus/(2*cmath.pi)
					
					"""Combining dipole moments and relevant delta function term for q,i,j"""
					Chielem[nqpoint,iphon,jphon,:,:]=np.conj(Chi[None,nqpoint,iphon,jphon,:,None])*(Chi[None,nqpoint,iphon,jphon,None,:])
					chin = DeltaSum[:,None,None]*np.conj(Chi[None,nqpoint,iphon,jphon,:,None])*(Chi[None,nqpoint,iphon,jphon,None,:])
					"""Contributing q,i,j term to the overall susceptibility"""
					Imag_Susceptibility[:,:,:]+=chin

					Imag_susceptibility_Tensor[nqpoint,iphon,jphon,:,:]=Chielem[nqpoint,iphon,jphon,:,:]*max(DeltaSum)
					Edist[nqpoint,iphon,jphon]=hbar*(float(phoni['frequency'])*THz+float(phonj['frequency'])*THz)
					Imag_Susceptibility_Tensor[nqpoint,iphon,jphon,:,:,:]=np.transpose(DeltaSum[:,None,None]*np.conj(Chi[None,nqpoint,iphon,jphon,:,None])*(Chi[None,nqpoint,iphon,jphon,None,:]),(1,2,0))	
					

	#i=1
	#j=4
	Imag_Susceptibility_Tensor=63.321006445655684*cmath.pi/2*Imag_Susceptibility_Tensor

	#np.save('Diamond161616_Imag_susceptibility_tensor.npy',Imag_Susceptibility_Tensor)
	'''fig, ax = plt.subplots(figsize=(3,2))
	for nqpoint, qpointn in enumerate(meshdata['phonon']):
		nmode=0
		for iphon, phoni in enumerate(meshdata['phonon'][nqpoint]['band']):
			for jphon, phonj in enumerate(meshdata['phonon'][nqpoint]['band']):
				if float(phoni['frequency']) > 0 and float(phonj['frequency'])>0:
					photonenergyelem=(float(phoni['frequency'])+float(phonj['frequency']))
					phononenergy1FracElem=float(phoni['frequency'])#/photonenergyelem
					#print('mode 1 energy fraction:',phononenergy1FracElem)
					photonenergy[nqpoint,nmode]=photonenergyelem
					phononenergy1Fraction[nqpoint,nmode]=phononenergy1FracElem					

					fraclist.append(phononenergy1FracElem)
					Zs[nqpoint,nmode]=506.56805156524547*cmath.pi/2*Imag_susceptibility_Tensor[nqpoint,iphon,jphon,0,0]	
					nmode+=1
		ZsSort=np.argsort(Zs[nqpoint,:])
		#print(ZsSort)
		plt.scatter(phononenergy1Fraction[nqpoint,ZsSort],photonenergy[nqpoint,ZsSort]*THz/(100*c),c=Zs[nqpoint,ZsSort],s=5)
			
		plt.xlabel('$\omega_1/\omega$')
		plt.ylabel('$\omega~(\mathrm{cm}^{-1})$')


	
	for i in range(3):
		for j in range(EigenLength):
			if i!=j:
				plt.figure()
				plt.plot(freqRange/c*10**-2,Imag_Susceptibility_Tensor[:,i,j,0,0,:].sum(axis=0))
				#plt.title(str(i+1)+str(j+1))
				plt.ylabel('$\chi$')
				plt.xlabel('wave-number $\mathrm{cm}^{-1}$')
				plt.savefig(str('./DiamondSusceptSumcont/Diamond')+str(i+1)+str(j+1)+str('SusceptContribution.png'),format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
				plt.close()
				#plt.show()
			
	Imag_Susceptibility_TA_LO=Imag_Susceptibility_Tensor[:,2,21:23,0,0,:]
	print(np.shape(Imag_Susceptibility_TA_LO))	
	plt.plot(freqRange/c*10**-2,np.transpose(Imag_Susceptibility_Tensor[:,j,0,0,:].sum(axis=0)))
	plt.show()'''


	Zs=np.zeros((weightsum,EigenLength*EigenLength),dtype=float)
	photonenergy=np.zeros((weightsum,EigenLength*EigenLength),dtype=float)
	phononenergy1Fraction=np.zeros((weightsum,EigenLength*EigenLength),dtype=float)
	fraclist=[]	
	#BinWidthomega1=(max(38.6800606247)-min(3.5256222713))/10
	
	bin_number=21	

	BinWidthomega=(2*38.6800606247)/bin_number

	BinWidthomega1=1/bin_number
	
	omegabinlist = [[] for _ in range(bin_number)]
	omegabinlistarr = np.empty(bin_number, object)
	omegabinlistarr[:]= omegabinlist

	omega1binlist = [[] for _ in range(bin_number)]	
	omega1binlistarr = np.empty(bin_number, object)
	omega1binlistarr[:]= omega1binlist

	SusceptBin=np.zeros((bin_number,bin_number),dtype=float)

	for bin_number_omega1 in range(bin_number):
		for bin_number_omega in range(bin_number):
			for nqpoint, qpointn in enumerate(meshdata['phonon']):
				nmode=0
				for iphon, phoni in enumerate(meshdata['phonon'][nqpoint]['band']):
					for jphon, phonj in enumerate(meshdata['phonon'][nqpoint]['band']):
						photon_freq=float(phoni['frequency'])+float(phonj['frequency'])
						phonon1_fraction=float(phoni['frequency'])/photon_freq
						Suscept_Elem=Imag_susceptibility_Tensor[nqpoint,iphon,jphon,0,0]
				
						#print('photon',BinWidthomega*bin_number,photon_freq)
						if photon_freq > BinWidthomega*bin_number_omega and photon_freq < BinWidthomega*(bin_number_omega+1):
							omegabinlist[bin_number_omega].append(photon_freq)
							SusceptBin[bin_number_omega,bin_number_omega1]+=Suscept_Elem

						if phonon1_fraction > BinWidthomega1*(bin_number_omega1) and phonon1_fraction < BinWidthomega1*(bin_number_omega1+1):
							omega1binlist[bin_number_omega1].append(phonon1_fraction)
							SusceptBin[bin_number_omega,bin_number_omega1]+=Suscept_Elem

			#omegabinlist[bin_number_omega]=np.array(omegabinlist[bin_number_omega],dtype=float)
		#omega1binlist[bin_number_omega1]=np.array(omega1binlist[bin_number_omega1],dtype=float)
	contouromega=np.linspace(0,2*38.6800606247,bin_number)
	contouromega1=np.linspace(0,1,bin_number)
	
	SusceptSum=SusceptBin.sum(axis=1)
	print((SusceptBin/SusceptSum[:,np.newaxis]).sum(axis=1))

	fig, ax = plt.subplots(figsize=(3,2))
	plt.contourf(contouromega1,contouromega*THz/(100*c),SusceptBin/SusceptSum[:,np.newaxis])
	cbar=plt.colorbar()
	cbar.set_label('Im($\chi_x$)')
	plt.xlabel('$\omega_1/\omega$')
	plt.ylabel('$\omega~(\mathrm{cm}^{-1})$')
	plt.show()
	fig.savefig('DiamondModeContributionContour.pdf',format='pdf',dpi=400,bbox_inches='tight',pad_inches=0.2)

	#print((omegabinlist))

	'''fig, ax = plt.subplots(figsize=(3,2))
	for nqpoint, qpointn in enumerate(meshdata['phonon']):
		nmode=0
		for iphon, phoni in enumerate(meshdata['phonon'][nqpoint]['band']):
			for jphon, phonj in enumerate(meshdata['phonon'][nqpoint]['band']):
				if float(phoni['frequency']) > 0 and float(phonj['frequency'])>0:
					photonenergyelem=(float(phoni['frequency'])+float(phonj['frequency']))
					phononenergy1FracElem=float(phoni['frequency'])/photonenergyelem
					#print('mode 1 energy fraction:',phononenergy1FracElem)
					photonenergy[nqpoint,nmode]=photonenergyelem
					phononenergy1Fraction[nqpoint,nmode]=phononenergy1FracElem					

					fraclist.append(phononenergy1FracElem)
					Zs[nqpoint,nmode]=506.56805156524547*cmath.pi/2*Imag_susceptibility_Tensor[nqpoint,iphon,jphon,0,0]	
					nmode+=1
		#modesort=np.argsort(photonenergy[nqpoint,:])
		#photonenergy[nqpoint,:]=photonenergy[nqpoint,modesort[1]]
		#phononenergy1Fraction[nqpoint,:]=phononenergy1Fraction[nqpoint,modesort[1]]
		ZsSort=np.argsort(Zs[nqpoint,:])
		#print(ZsSort)
		plt.scatter(phononenergy1Fraction[nqpoint,ZsSort],photonenergy[nqpoint,ZsSort]*THz/(100*c),c=Zs[nqpoint,ZsSort],s=5)
			
		plt.xlabel('$\omega_1/\omega$')
		plt.ylabel('$\omega~(\mathrm{cm}^{-1})$')
					
	#print(np.argsort(photonenergy,axis=1))
	#print(photonenergy[0,np.argsort(photonenergy)],phononenergy1Fraction[0,np.argsort(photonenergy)])
	
	cbar=plt.colorbar()
	cbar.set_label('Im($\chi_x(\omega_1(\mathbf{q}),\omega_2(\mathbf{q}),\omega)$)')	
	plt.show()
	fig.savefig('DiamondModeContributionScatter.pdf',format='pdf',dpi=400,bbox_inches='tight',pad_inches=0.2)'''
	
	'''for i in range(EigenLength):
		for j in range(EigenLength):
			plt.scatter(Edist[:,i,j]/eV,506.56805156524547*cmath.pi/2*Imag_Susceptibility_Tensor[:,i,j])
	plt.xlabel('phonon energy (eV)')
	plt.ylabel('$\chi''$')
	plt.show()'''

	'''for i in range(EigenLength):
		for j in range(EigenLength):
			Susceptmax=Imag_Susceptibility_Tensor[:,i,j]

			Susceptmax=506.56805156524547*cmath.pi/2*Susceptmax[dist_Indices]
			plt.scatter(Qdist,Susceptmax)
			plt.xlabel('distance from $\Gamma$')
			plt.ylabel('$\chi''$')
			plt.savefig(str('./Suscept_result/DiamondSuseptScatterModes')+str(i)+str(j)+str('Frequency')+str(freqRange[156]/THz)+str('.png'),format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)

			fig=plt.figure(figsize = (3,2))
			ax = fig.add_subplot(projection='3d')
	
			img=ax.scatter(Q[:,0],Q[:,1],Q[:,2],c=Susceptmax,cmap=plt.hot())
			fig.colorbar(img)
			ax.set_xlabel('$\mathbf{Q}_x$')
			ax.set_ylabel('$\mathbf{Q}_y$')
			ax.set_zlabel('$\mathbf{Q}_z$')
		plt.show()'''




	

	"""Normalizing density of states with respect to density of states without BE functions"""				
	print('dipole element sum:',Chielem.sum(axis=0).sum(axis=0).sum(axis=0)/weightsum)
	for xs in range(3):
		for ys in range(3):
			if LinewidthSimulate==True:
				Imag_Susceptibility[:,xs,ys]=cmath.pi*Imag_Susceptibility[:,xs,ys]/(2)

			else:
				Tetrahedral_Shift=(shift_factor/(DeltaSumMod))/THz	
				Tetrahedral_Shift[np.isnan(Tetrahedral_Shift)]=0
				Imag_Susceptibility[:,xs,ys]=cmath.pi*Imag_Susceptibility[:,xs,ys]*Tetrahedral_Shift/(2)
	
	return Imag_Susceptibility


def get_ChiN_single_phonon_VASP(M,cell_divisions,meshdata,PhononProperties,NCell=1,testEigenvector=False,testChi=False):
	ChiN=np.zeros((1,3*len(cell_divisions[0,:]),3),dtype=complex)
	print(np.shape(ChiN))#1 qpoint, 3Natom modes, 3 directions
 
	GammaEigenvectors=np.zeros(	(3*len(cell_divisions[0,:]),len(cell_divisions[0,:]),3),dtype=float)
	GammaBorn=np.zeros((len(cell_divisions[0,:]),3,3),dtype=complex)

	"""iterates over the considered atom, the displaced atom, and the direction of the electric field, contraction over other directions (and the atomic indices) is done via dot product"""
	
	for iphon, phoni in enumerate(PhononProperties['Frequencies']):
		Evec=PhononProperties['Eigenvectors'][iphon,:,:]
		eigenvectors=Evec
		eigenvectorsneg=eigenvectors
		#print('Eigenvectors Shape:',np.shape(eigenvectors))

		evs=np.zeros(30)
		
		for a in range(len(eigenvectors[:,0])):
			evs[a]=eigenvectorsneg[a].dot(eigenvectors[a])
		print('Eigenvector Sum:',np.sum(evs))
		
		for atom in range(len(M[0,:,0,0])):#sum over k_1
			GammaBorn[atom,:,:]=M[0,atom,0,0]
		
			for EField_direction in range(len(M[0,0,:,0])):#sum over alpha_1
				if testEigenvector==True:
					print('Eigenvector for atoms ',atom+1,':',eigenvectors[atom])
					#print('Orthogonality Check:',eigenvectorsneg[atom].dot(eigenvectors[atom+1]))
					#np.savez(str('./Data_Files/GaN_Born_and_Eigenvector/phononEigenvectorMode'+str(iphon)+str('.npz')),eigenvectors,'eigenvector')
				magnitudeAtom=np.sqrt(np.sum(evs))
				eigenvectors[atom,:]=eigenvectors[atom,:]/(magnitudeAtom)
				
			
				EvecMass=eigenvectors[atom,:]/math.sqrt(float(meshdata['points'][atom]['mass'])*amu)					

				if testEigenvector==True:
					print('atom scaled eigenvector:',eigenvectors[atom]/math.sqrt(float(meshdata['points'][atom]['mass'])*amu))
					#print('M element:',M[mindistanceindex,atom,EField_direction,:])
					np.savez(str('./Data_Files/GaN_Born_and_Eigenvector/phononDipoleElement'+str('.npz')),M,'BornCharge')				
	
				ChiN[0,iphon,EField_direction]+=(M[0,atom,EField_direction,:].dot(EvecMass)	)
				
				if testChi==True:
					print('mass of atom:',float(meshdata['points'][atom]['mass']))
					print('Product of Fourier Transformed Dipole moment element and eigenvectors ',(M[0,atom,EField_direction,:].dot(EvecMass)	)	)
	
		
	
	
		ChiN[0,iphon,:]=1/np.sqrt(hbar*2*NCell*len(cell_divisions[:,0])*2*cmath.pi*np.absolute(float(PhononProperties['Frequencies'][iphon]))*THz)*ChiN[0,iphon,:]
		#ChiN[0,iphon,0]=ChiN[0,iphon,1]
		#if np.sum(ChiN[0,iphon,:])/3<10**2:
			#ChiN[0,iphon,:]=np.zeros(3)
	#np.savez(str('./Data_Files/Born_and_Eigenvector/phononEigenvectorMode'+str(iphon)+str('.npz')),GammaEigenvectors,'eigenvector')
	#np.savez(str('./Data_Files/Born_and_Eigenvector/phononDipoleElement'+str('.npz')),GammaBorn,'BornCharge')

	#print(ChiN)
	return ChiN


def get_Imag_Susceptibility_single_phonon_VASP(Chi,PhononProperties,Linewidth,freqRange,alpha,shift_factor,NCell=1,freqUnit=THz,testQpoint=False):
	Imag_Susceptibility=np.zeros((len(freqRange),3),dtype=complex)
	
	#EigenLength=6	
	EigenLength=(len(PhononProperties['Frequencies']))

	ChiMod=np.zeros((EigenLength,3))
	ChiMod2=np.zeros((EigenLength,3,len(freqRange)))

	DOSDelta=np.zeros((EigenLength,3,len(freqRange)),dtype=complex)
	DeltaSumMod=np.zeros((len(freqRange)),dtype=complex)
	DeltaSum=np.zeros((len(freqRange)),dtype=complex)
	
	
	
		
	"""For non-cubic systems, sum must be taken for all direction combinations, not just alpha-alpha (handbook der physik)"""
	IR_Active_Modes=np.linspace(1,90,90)
	
	ChiMod=np.zeros((EigenLength,3))		
	#print(IR_Active_Range)
	#print(2*cmath.pi*Frequencies[IR_Active_Range]/THz)
	#IR_Active_Modes=np.array([17,18,19,20,21,24,35,53,55,68])
	
	for iphon, phoni in enumerate(PhononProperties['Frequencies']):
		if iphon in IR_Active_Modes:
			
				ChiMod[iphon,:]=Chi[0,iphon,:].real**2+Chi[0,iphon,:].imag**2
				delta=2*float(PhononProperties['Frequencies'][iphon])*THz/((float(PhononProperties['Frequencies'][iphon])*THz)**2-(freqRange+1j*Linewidth)**2)
				

				DeltaSum=delta			

				
				"""Combining dipole moments and relevant delta function term for q,i,j"""
				ChiSum1=ChiMod[iphon,0]*DeltaSum
				ChiSum2=ChiMod[iphon,1]*DeltaSum
				ChiSum3=ChiMod[iphon,2]*DeltaSum
						
				"""Contributing q,i,j term to the overall susceptibility"""
				Imag_Susceptibility[:,0]+=ChiSum1
				Imag_Susceptibility[:,1]+=ChiSum2
				Imag_Susceptibility[:,2]+=ChiSum3

	"""Normalizing density of states with respect to density of states without BE functions"""				
	magnitude=1
	
	
	Tetrahedral_Shift=1
	
	Imag_Susceptibility[:,0]=cmath.pi*Imag_Susceptibility[:,0]*Tetrahedral_Shift/(2*NCell)
	Imag_Susceptibility[:,1]=cmath.pi*Imag_Susceptibility[:,1]*Tetrahedral_Shift/(2*NCell)
	Imag_Susceptibility[:,2]=cmath.pi*Imag_Susceptibility[:,2]*Tetrahedral_Shift/(2*NCell)
	#print('TDOS Test:',((DeltaSum/magnitude)/TDOS))	
	#print('Modified TDOS magnitude:',scipy.integrate.simps(DeltaSum/magnitude,freqRange))	

	Imag_Susceptibility_Single_Phonon=np.imag(Imag_Susceptibility)
	return Imag_Susceptibility_Single_Phonon

def get_Susceptibility_Linewidth(Chi,meshdata,Temperature,freqRange,Linewidth,alpha,NCell=1,freqUnit=THz,n=2,testQpoint=False):
	Susceptibility=np.zeros((len(freqRange),3),dtype=complex)
	EigenLength=(len(meshdata['phonon'][0]['band']))
	ChiMod=np.zeros((EigenLength,EigenLength,3))
	DOSDelta=np.zeros((EigenLength,EigenLength,3,len(freqRange)))
	DeltaSumMod=np.zeros((len(freqRange)))
	
	T=Temperature
	weightsum=np.prod(np.array(meshdata['mesh'],dtype=int))
	
		
	"""Frequencies are updated for each q-point, but chi still needs to be specified at a q-point"""
	"""For non-cubic systems, sum must be taken for all direction combinations, not just alpha-alpha (handbook der physik)"""
	"""Linewidths are distributed among modes and q-points """
	
	for nqpoint, qpointn in enumerate(meshdata['phonon']):# for silicon at least, the opposite q-points have the same frequency modes, eigenvectors are complex conjugate 
		Frequencies=[]
		DeltaSum=np.zeros((len(freqRange)))#delta function term with bose einstein functions for phonon branches as q ,(i,j)
		ChiMod=np.zeros((EigenLength,EigenLength,3))#Dipole moment element for q-point	
		if testQpoint==True:
			print('q-point number:',nqpoint)
		for b in range(len(meshdata['phonon'][nqpoint]['band'])):
			Frequencies.append(float(meshdata['phonon'][nqpoint]['band'][b]['frequency'])*THz)
		Frequencies=np.array(Frequencies)			
		for iphon, phoni in enumerate(meshdata['phonon'][nqpoint]['band']):
			for jphon, phonj in enumerate(meshdata['phonon'][nqpoint]['band']):
				"""check added since overtones should not contribute to IR spectra (literature)"""
				fsum=float(phoni['frequency'])*THz+float(phonj['frequency'])*THz
				fdiff=float(phoni['frequency'])*THz-float(phonj['frequency'])*THz
				BESum=(DS.BEFunc(float(phoni['frequency'])*THz,T)+1/2)+(DS.BEFunc(float(phonj['frequency'])*THz,T)+1/2)
				BEDiff=(DS.BEFunc(float(phoni['frequency'])*THz,T)+1/2)-(DS.BEFunc(float(phonj['frequency'])*THz,T)+1/2)

				if iphon!=jphon:
						#*int(qpointn['weight'])/weightsum
					ChiMod[iphon,jphon,:]=Chi[nqpoint,iphon,jphon,:].real**2+Chi[nqpoint,iphon,jphon,:].imag**2
					if testQpoint==True:
						print('Bose Einstein Term:',BEFSum([float(phoni['frequency'])*freqUnit,float(phonj['frequency'])*freqUnit],Temperature))

					deltaplus=(2*fsum*BESum	)/(	(fsum)**2-(freqRange+1j*(Linewidth[nqpoint, iphon]+Linewidth[nqpoint, jphon]	)	)**2)*int(qpointn['weight'])/weightsum

					deltaminus=(2*fdiff*BEDiff)/(	(fdiff)**2-(freqRange+1j*(Linewidth[nqpoint, iphon]+Linewidth[nqpoint, jphon]	)	)**2)*int(qpointn['weight'])/weightsum
					#if float(phoni['frequency'])*freqUnit+float(phonj['frequency'])*freqUnit					

	
					DeltaSum=deltaplus/(2*cmath.pi)+deltaminus/(2*cmath.pi)	

					"""term to deal with normalizing DOS"""
					
						
					"""Combining dipole moments and relevant delta function term for q,i,j"""
					ChiSum1=ChiMod[iphon,jphon,0]*DeltaSum
					ChiSum2=ChiMod[iphon,jphon,1]*DeltaSum
					ChiSum3=ChiMod[iphon,jphon,2]*DeltaSum
						
					"""Contributing q,i,j term to the overall susceptibility"""
					Susceptibility[:,0]+=ChiSum1
					Susceptibility[:,1]+=ChiSum2
					Susceptibility[:,2]+=ChiSum3

	"""Normalizing density of states with respect to density of states without BE functions"""				
	magnitude=1
	Tetrahedral_Shift=1
	
	Susceptibility[:,0]=cmath.pi*Susceptibility[:,0]*Tetrahedral_Shift/(2*1)
	Susceptibility[:,1]=cmath.pi*Susceptibility[:,1]*Tetrahedral_Shift/(2*1)
	Susceptibility[:,2]=cmath.pi*Susceptibility[:,2]*Tetrahedral_Shift/(2*1)
	#print('TDOS Test:',((DeltaSum/magnitude)/TDOS))	
	#print('Modified TDOS magnitude:',scipy.integrate.simps(DeltaSum/magnitude,freqRange))	
	
	return Susceptibility




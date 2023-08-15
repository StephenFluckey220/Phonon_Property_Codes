import numpy as np
from numpy.linalg import solve,inv
import MultiphononParameters as mp
from MultiphononParameters import *
import cmath
import math
import scipy
from scipy.signal import hilbert, savgol_filter
from scipy import integrate
import argparse as ag
import matplotlib.pyplot as plt
import Read_Phonon_Properties as rp
import DOS_Approx as DS
import Phonon_Dipole as PD
import Absorption as ab
import yaml
from yaml import load, dump, BaseLoader, SafeLoader, CBaseLoader
import phonopy as php
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh
from phonopy.structure.tetrahedron_method import TetrahedronMethod
import phonopy._phonopy as phonoc
import os
import csv
from MUnitCellCalc import *
import vasprun as vaspr
import pymatgen as mg
import pymatgen.io.vasp as VASP

"""Two Phonon: DipoleMomentCalc.py --dimZ 1 --dimY 1 --dimX 1 --freqRangeF 60 --NAT 30 --NCell 6 --disprange 90 --material ../../Sapphire/Relax/ --meshfile mesh555 --points 1500 --alpha 0.17 --Name Al2O3 --savePlot --freqRangeI 30 --epsilon 3.08 --smooth
"""


"""python DipoleMomentCalc.py --dimZ 2 --dimY 2 --dimX 2 --freqRangeF 80 --NAT 8 --disprange 24 --material ../C-Diamond/DIpole/ --meshfile mesh888 --points 2500 --alpha 0.16 --Name C --LinewidthSimulate --epsilon 5.3"""

"""python DipoleMomentCalc.py --dimZ 4 --dimY 4 --dimX 4 --freqRangeF 80 --NAT 2 --disprange 6 --material ../C-Diamond/DIpole/Primitive/Diamond_Input_Files_VASP/Diamond4x4x4/ --meshfile mesh999VASP --points 500 --alpha 0.16 --Name C --epsilon 5.3"""


"""One Phonon: python DipoleMomentCalc.py --dimZ 2 --dimY 2 --dimX 2 --freqRangeF 55 --NAT 10 --disprange 30 --material ../../Sapphire/Primitive/Unit/ --meshfile mesh555VASP --points 2500 --alpha 0.04 --Name Al2O3 --LinewidthSimulate --plotSusceptibility --singlePhonon  --plotAbsorption"""

def get_CSV_Absorption(absorptiondata):
	nuRef=[]
	alphaRef=[]
	csv.register_dialect('excel', delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
	with open(absorptiondata, newline='') as csvfile:
		Nreader = csv.reader(csvfile, delimiter=',')
		data=list(Nreader)
		for l in range(1,len(data)-1):
			dataarr=np.array(data[l])
			alphaRef.append(float(dataarr[1]))
			nuRef.append(float(dataarr[0]))
	return nuRef, alphaRef
	

def get_direction(count):
	if count==1:
		direction='a'
	elif count==2:
		direction='b'
	elif count==3:
		direction='c'
	return direction

def get_alpha2index(direction):
	if direction=='a':
		alpha2index=0
	elif direction=='b':
		alpha2index=1
	elif direction=='c':
		alpha2index=2
	return alpha2index


def categorize_unit_cells(nat,dimX,dimY,dimZ):
	num_cells=(nat*dimX*dimY*dimZ)//nat
	cell_divisions=np.zeros((num_cells,nat),dtype=int)#rows for each unit cell, columns for each atom in cell
	for n in range(nat):	
		for r in range(num_cells):
			cell_divisions[r,n]=r+1+n*dimX*dimY*dimZ
			
	return cell_divisions

def categorize_unit_cells_SC(nat,dimX,dimY,dimZ):
	num_cells=int(dimX*dimY*dimZ)
	cell_divisions=np.zeros((num_cells,nat//num_cells),dtype=int)#rows for each unit cell, columns for each atom in cell
	for n in range(nat//num_cells):	
		for r in range(num_cells):
			cell_divisions[r,n]=r+1+n*dimX*dimY*dimZ
			
	return cell_divisions

def get_linshift(freq,Gammas,unit=THz):
	shifted_frequency=1#placeholder until lineshifing data for relevant materials is present. 

	return shifted_frequency

def get_TransformationMatrix(dispadata,negative):
	TransformationMatrix=np.zeros((3,3))
	if negative==True:
		TransformationMatrix[0,:]=dispatom[0]['displacement']
			
		TransformationMatrix[1,:]=dispatom[2]['displacement']
			
		TransformationMatrix[2,:]=dispatom[4]['displacement']
		
	else:
		TransformationMatrix[0,:]=dispatom[0]['displacement']
			
		TransformationMatrix[1,:]=dispatom[1]['displacement']
			
		TransformationMatrix[2,:]=dispatom[2]['displacement']
		
	TransformationMatrix=TransformationMatrix.T#might need to be a .T here

	return TransformationMatrix

def Wavenumber_to_Wavelength(freqcm):
    return 10000/freqcm 

def Wavelength_to_Wavenumber(wavelengthum):
    return wavelengthum * 10000 

def ThreeD_Scatter(NAT,DimX,DimY,DimZ,CellCoordinates,CellCoordinatesFractional,BornDiff,lattice_vectors):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	n = NAT*NATSCell
	RDiff=[]
	print(lattice_vectors)
	FractionalCoordinates=np.array(CellCoordinatesFractional).astype(float)
	coord_trim=[]
	nvecs=2
	#lat_vec=[range(-nvecs,nvecs),range(-nvecs,nvecs),range(-nvecs,nvecs)]
	#lat_vecs=[[np.kron(lat_vec[0],lat_vec[1]**0,lat_vec[2]**0)],
	#	[np.kron(lat_vec[0]**0,lat_vec[1],lat_vec[2]**0)],
	#	[np.kron(lat_vec[0]**0,lat_vec[1]**0,lat_vec[2])]]
	#print((lat_vecs))
	#for coord in FractionalCoordinates:
	#	coord2=coord+lat_vecs
	#	coord_trim.append(coord2[argmin(np.norm(coord2))])

	FoldedCoordinates1=np.array(np.where(FractionalCoordinates[:,0]>0.5)).T
	FoldedCoordinates2=np.array(np.where(FractionalCoordinates[:,1]>0.5)).T
	FoldedCoordinates3=np.array(np.where(FractionalCoordinates[:,2]>0.5)).T
	#print(FoldedCoordinates1,FoldedCoordinates2,FoldedCoordinates3)	
	for coord in range(len(FractionalCoordinates[:,0])):
		if coord in FoldedCoordinates1:
			FractionalCoordinates[coord,0]=np.absolute(FractionalCoordinates[coord,0]-1.0)

		if coord in FoldedCoordinates2:
			FractionalCoordinates[coord,1]=np.absolute(FractionalCoordinates[coord,1]-1.0)
		
		if coord in FoldedCoordinates3:
			FractionalCoordinates[coord,2]=np.absolute(FractionalCoordinates[coord,2]-1.0)
	FoldedDirectCoordinates=np.zeros(np.shape(FractionalCoordinates))
	#print(FractionalCoordinates)
	for vectors in range(len(FractionalCoordinates[:,0])):
		#print(lattice_vectors,FractionalCoordinates[vectors,:])
		FoldedDirectCoordinates[vectors,:]=lattice_vectors.dot(FractionalCoordinates[vectors,:])
	#print(FoldedDirectCoordinates,CellCoordinates)
		
	xyz=np.array(CellCoordinates).astype(float)
	dist=np.sqrt(FoldedDirectCoordinates[:,0]**2+FoldedDirectCoordinates[:,1]**2+FoldedDirectCoordinates[:,2]**2)#Assumes atom 1 is at the origin
	
	#print(dist)
	img=ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=BornDiff*angstrom/eV,cmap=plt.hot())
	fig.colorbar(img)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()
	
	plt.figure()
	plt.scatter(dist,BornDiff*angstrom/eV,color='k')
	#plt.ylim(2.9999995*10**-4,3.0000005*10**-4)
	plt.xlabel('distance ($\dot{A}$)')
	plt.ylabel('$dZ^*/dU$ (eV/$\dot{A}$)')
	plt.show()

"""Creates individual elements of dipole moment tensor by taking the derivative of the born effective charge tensor"""
"""of atom s' in direction alpha with respect to displacement in direction beta on atom s"""
"""Inputs are the BEC tensor element with the displacement, the corresponding original BEC tensor element and the displacement in that direction"""


if __name__=='__main__':
	parser=ag.ArgumentParser()  
	parser.add_argument('--material',default='./Si3N4',type=str,help='path to directory containing Born effective charge calculations')
	parser.add_argument('--meshfile',default='mesh',type=str,help='name of file containing phonon eigenstates at each q-point on the grid')
	parser.add_argument('--pathLoad',default='./Dipole_Data/',type=str,help='name of file containing phonon eigenstates at each q-point on the grid')
	parser.add_argument('--Name',default='Si3N4',type=str,help='Name of material')
	parser.add_argument('--disprange',default=12,type=int,help='number of displacements for which Born effective charges are calculated')
	parser.add_argument('--negative',nargs='?',default=False,const=True,help='if negative displacements are present in addition to positive')
	parser.add_argument('--smooth',nargs='?',default=False,const=True,help='determines whether Savgol smoothing is used on final results')
	parser.add_argument('--Supercell',nargs='?',default=False,const=True,help='whether the input data corresponds to a unit cell or a supercell')
	parser.add_argument('--singlePhonon',nargs='?',default=False,const=True,help='determines whether Savgol smoothing is used on final results')
	parser.add_argument('--testDipole',nargs='?',default=False,const=True,help='prints dipole moments')
	parser.add_argument('--testBorn',nargs='?',default=False,const=True,help='prints born effective charges and change in born effective charges')
	parser.add_argument('--testFile',nargs='?',default=False,const=True,help='prints filename at each displacement')
	parser.add_argument('--testChi',nargs='?',default=False,const=True,help='prints the dipole element multiplied by the eigenvectors and the sum for each phonon branch combination')
	parser.add_argument('--TempCompare',nargs='?',default=False,const=True,help='')
	parser.add_argument('--testTransformation',nargs='?',default=False,const=True,help='')
	parser.add_argument('--testEigenvector',nargs='?',default=False,const=True,help='')
	parser.add_argument('--plotSusceptibility',nargs='?',default=False,const=True,help='')
	parser.add_argument('--plotAbsorption',nargs='?',default=False,const=True,help='')
	parser.add_argument('--savePlot',nargs='?',default=False,const=True,help='')
	parser.add_argument('--testBEF',nargs='?',default=False,const=True,help='displays Bose Einstein functions')
	parser.add_argument('--testPhase',nargs='?',default=False,const=True,help='displays phase factors')
	parser.add_argument('--testQpoint',nargs='?',default=False,const=True,help='displays various parameters at each q-point')
	parser.add_argument('--testDOS',nargs='?',default=False,const=True,help='plot the relevant density of states in both smearing and tetrahedron methods')
	parser.add_argument('--CellSum',nargs='?',default=False,const=True)
	parser.add_argument('--LinewidthSimulate',nargs='?',default=False,const=True)
	parser.add_argument('--amplitude',default=0.1,type=float,help='magnitude of displacments for each displaced atom (same as chosen amplitude in phonopy/3py input)')
	parser.add_argument('--freqRangeI',default=0,type=float,help='initial frequency in tested frequency range')
	parser.add_argument('--epsilon',default=None,type=float,help='high frequency dielectric constant')
	parser.add_argument('--freqRangeF',default=30,type=float,help='final frequency in tested frequency range')
	parser.add_argument('--Temperature',default=298,type=float,help='temperature at which susceptibility is calculated')
	parser.add_argument('--Thickness',default=10,type=float,help='temperature at which susceptibility is calculated')
	parser.add_argument('--zfill',default=2,type=int,help='number of leading zeros in disp-{zfill}x subdirectories. Only needs to be changed for very high number of displacements')
	parser.add_argument('--alpha',default=0.01,type=float,help='size of smearing parameter for Gaussian smearing density-of-states calculation')
	parser.add_argument('--NAT',default=4,type=int,help='number of atoms for which Born charge changes and phonon eigenstates are calculated')
	parser.add_argument('--NCell',default=1,type=float,help='deprecated argument: number of primitive cells')
	parser.add_argument('--NATUnitCell',default=2,type=int,help='number of atoms for which Born charge changes and phonon eigenstates are calculated')
	parser.add_argument('--phononNumber',default=2,type=int,help='placeholder variable: if code is adapted for more than two phonons, this will control which calculations are performed')
	parser.add_argument('--points',default=100,type=int,help='number of discrete frequency points')
	parser.add_argument('--dimX',default=1,type=int,help='size of supercell in the x-direction')
	parser.add_argument('--dimY',default=1,type=int,help='size of supercell in the y-direction')
	parser.add_argument('--dimZ',default=1,type=int,help='size of supercell in the z-direction')
	parser.add_argument('--window',default=121,type=int,help='size of window for Savgol smoothing')
	parser.add_argument('--polyorder',default=3,type=int,help='polynomial order for Savgol smooting')
	args=parser.parse_args()
	
	os.system("echo Hello from the other side!")
	
	dispdata=yaml.load(open(str(args.material)+str('/phonopy_disp.yaml')),Loader=BaseLoader)
	meshdata=yaml.load(open(str(args.material)+str(args.meshfile)+str('.yaml')),Loader=BaseLoader)

	disprange=args.disprange
	alpha=args.alpha*THz
	material=args.material
	thickness=args.Thickness
	NAT=args.NAT
		
	name=args.Name
	
	amplitude=args.amplitude*angstrom
	freqRange=np.linspace(args.freqRangeI,args.freqRangeF,args.points)*THz

	"""results in inverse meters need to be divided by 100 to get results in inverse cm"""
	freqcm=freqRange/c*10**-2
	freqcm=np.linspace(min(freqRange)/c*10**-2,max(freqRange)/c*10**-2,args.points)
	print(max(freqRange)/c*10**-2,max(freqcm))


	"""Individual susceptibility contribution plots"""
	Epsilon0=args.epsilon
	
	Imag_Susceptibility_Tensor_121212=np.absolute(np.load('Si_Imag_susceptibility_tensor.npy'))
	print(np.shape(Imag_Susceptibility_Tensor_121212)	)
	Imag_Susceptibility_Tensor_121212_Real=-np.imag(hilbert(3*Imag_Susceptibility_Tensor_121212[:,:,:,0,0,:])) + Epsilon0
	ImChi_Tot=Imag_Susceptibility_Tensor_121212_Real+3j*Imag_Susceptibility_Tensor_121212[:,:,:,0,0,:]
	k121212=np.imag(np.sqrt(ImChi_Tot)).sum(axis=0).sum(axis=0).sum(axis=0)

	plt.plot(freqcm,4*cmath.pi*k121212*freqcm)
	plt.show()

	for q in range(int(meshdata['nqpoint'])):	
		for iphon, phoni in enumerate(meshdata['phonon'][q]['band'][:]):
			plt.scatter(float(phoni['frequency'])*np.ones(6),Imag_Susceptibility_Tensor_121212[q,iphon,:,0,0,150])
		plt.show()

	
	#ImChi_TA_LO_Tot=Imag_Susceptibility_Tensor_TA_LO_Real+3j*Imag_Susceptibility_Tensor_TA_LO[:,:,:,0,0,:]
	k_Tot=np.imag(np.sqrt(ImChi_Tot))
	print(np.shape(k_Tot)	)

	k_A_A=k_Tot[:,:3,:3,:].sum(axis=0).sum(axis=0).sum(axis=0)
	k_TA_LO=k_Tot[:,:2,3:,:].sum(axis=0).sum(axis=0).sum(axis=0)
	k_LO_TA=k_Tot[:,3:,:2,:].sum(axis=0).sum(axis=0).sum(axis=0)
	k_O_O=k_Tot[:,2:,2:,:].sum(axis=0).sum(axis=0).sum(axis=0)

	k_TA_TO=k_Tot[:,:2,[3],:].sum(axis=0).sum(axis=0).sum(axis=0)
	k_TO_TA=k_Tot[:,[3],:2,:].sum(axis=0).sum(axis=0).sum(axis=0)

	plt.plot(freqcm,4*cmath.pi*k_A_A*freqcm,linestyle='--')
	plt.plot(freqcm,4*cmath.pi*(k_TA_LO+k_LO_TA)*freqcm,linestyle='--')
	plt.plot(freqcm,4*cmath.pi*k_O_O*freqcm,linestyle='--')
	plt.plot(freqcm,4*cmath.pi*(k_A_A+k_O_O+k_TA_LO+k_LO_TA)*freqcm,linestyle='--')
	plt.plot(freqcm,4*cmath.pi*k121212*freqcm,color='r')
	plt.plot(freqcm,4*cmath.pi*(k_TO_TA+k_TA_TO)*freqcm,linestyle='--')
	plt.ylabel('absorption $(cm^{-1})$')
	plt.xlabel('wave-number $(cm^{-1})$')
	
	plt.legend(['A-A','A-O','O-O','A-A+O-O+A-O','all modes','TA-TO'])
	plt.show()

	Epsilon0=5.7
	Imag_Susceptibility_Tensor_121212=np.load('Diamond161616_Imag_susceptibility_tensor.npy')
	Imag161616testAbsorp=np.load('./Dipole_Results/C298Ktwo_phonon_absorption_NC161616.npy')
	Imag161616testFreq=np.load('./Dipole_Results/Cfrequency_range_inverse_cm888.npy')


	Imag_Suscept=(Imag_Susceptibility_Tensor_121212[:,:,:,0,0,:]+Imag_Susceptibility_Tensor_121212[:,:,:,1,1,:]+Imag_Susceptibility_Tensor_121212[:,:,:,2,2,:])

	Imag_Susceptibility_Tensor_121212_Real=-np.imag(hilbert(Imag_Suscept[:,:,:,:])) + Epsilon0


	Imag_Susceptibility_Tensor_Real=-np.imag(hilbert(Imag_Suscept[:,:,:,:].sum(axis=0).sum(axis=0).sum(axis=0)	)) + Epsilon0
	ImChi_Tot=Imag_Susceptibility_Tensor_Real+1j*Imag_Suscept[:,:,:,:].sum(axis=0).sum(axis=0).sum(axis=0)

	plt.plot(freqcm,Imag_Suscept[:,:,:,:].sum(axis=0).sum(axis=0).sum(axis=0))
	plt.show()

	Imag_Susceptibility_Tensor_A_A_Real=-np.imag(hilbert(Imag_Suscept[:,:3,:3,:].sum(axis=0).sum(axis=0).sum(axis=0)	)) + Epsilon0
	ImChi_Tot_A_A=Imag_Susceptibility_Tensor_A_A_Real+1j*Imag_Suscept[:,:3,:3,:].sum(axis=0).sum(axis=0).sum(axis=0)

	Imag_Susceptibility_Tensor_O_A_Real=-np.imag(hilbert(Imag_Suscept[:,2:,:2,:].sum(axis=0).sum(axis=0).sum(axis=0)	)) + Epsilon0
	ImChi_Tot_O_A=Imag_Susceptibility_Tensor_O_A_Real+1j*Imag_Suscept[:,2:,:2,:].sum(axis=0).sum(axis=0).sum(axis=0)

	Imag_Susceptibility_Tensor_A_O_Real=-np.imag(hilbert(Imag_Suscept[:,:2,2:,:].sum(axis=0).sum(axis=0).sum(axis=0)	)) + Epsilon0
	ImChi_Tot_A_O=Imag_Susceptibility_Tensor_A_O_Real+1j*Imag_Suscept[:,:2,2:,:].sum(axis=0).sum(axis=0).sum(axis=0)

	Imag_Susceptibility_Tensor_O_O_Real=-np.imag(hilbert(Imag_Suscept[:,2:,2:,:].sum(axis=0).sum(axis=0).sum(axis=0)	)) + Epsilon0
	ImChi_Tot_O_O=Imag_Susceptibility_Tensor_O_O_Real+1j*Imag_Suscept[:,2:,2:,:].sum(axis=0).sum(axis=0).sum(axis=0)

	k_Tot=np.imag(np.sqrt(ImChi_Tot))
	k_Tot_A_A=np.imag(np.sqrt(ImChi_Tot_A_A))
	k_Tot_O_A=np.imag(np.sqrt(ImChi_Tot_O_A))
	k_Tot_A_O=np.imag(np.sqrt(ImChi_Tot_A_O))
	k_Tot_O_O=np.imag(np.sqrt(ImChi_Tot_O_O))

	alpha_A_A=4*cmath.pi*k_Tot_A_A*freqcm
	alpha_A_O=4*cmath.pi*k_Tot_A_O*freqcm
	alpha_O_A=4*cmath.pi*k_Tot_O_A*freqcm
	alpha_O_O=4*cmath.pi*k_Tot_O_O*freqcm
	
	np.save('./Dipole_Results/Contribution_freqcm.npy',freqcm)
	np.save('./Dipole_Results/Diamond_A_A_absorption.npy',alpha_A_A)
	np.save('./Dipole_Results/Diamond_O_A_absorption.npy',alpha_O_A)
	np.save('./Dipole_Results/Diamond_A_O_absorption.npy',alpha_A_O)
	np.save('./Dipole_Results/Diamond_O_O_absorption.npy',alpha_O_O)

	plt.plot(freqcm,4*cmath.pi*k_Tot_A_A*freqcm)
	plt.plot(freqcm,4*cmath.pi*(k_Tot_A_O+k_Tot_O_A)*freqcm)
	plt.plot(freqcm,4*cmath.pi*k_Tot_O_O*freqcm)
	plt.plot(freqcm,4*cmath.pi*(k_Tot_O_A+k_Tot_A_O+k_Tot_O_O+k_Tot_A_A)*freqcm)
	#plt.plot(Imag161616testFreq,Imag161616testAbsorp/100)
	plt.legend(['A-A','A-O','O-O','A-A+O-O+A-O','all modes'])
	plt.show()


	ImChi_Tot=Imag_Susceptibility_Tensor_121212_Real+3j*Imag_Susceptibility_Tensor_121212[:,:,:,0,0,:]
	k121212=np.imag(np.sqrt(ImChi_Tot)).sum(axis=0).sum(axis=0).sum(axis=0)
	plt.plot(freqcm,4*cmath.pi*k121212*freqcm)
	plt.show()

	k_Tot=np.imag(np.sqrt(ImChi_Tot))
	k_tot=k_Tot.sum(axis=0).sum(axis=0).sum(axis=0)

	k_A_A=k_Tot[:,:3,:3,:].sum(axis=0).sum(axis=0).sum(axis=0)
	k_A_O=k_Tot[:,:2,3:,:].sum(axis=0).sum(axis=0).sum(axis=0)
	k_O_A=k_Tot[:,3:,:2,:].sum(axis=0).sum(axis=0).sum(axis=0)
	k_O_O=k_Tot[:,2:,2:,:].sum(axis=0).sum(axis=0).sum(axis=0)

	
	

	plt.plot(freqcm,4*cmath.pi*k_A_A*freqcm,linestyle='--')
	plt.plot(freqcm,4*cmath.pi*(2*k_O_A)*freqcm,linestyle='--')
	plt.plot(freqcm,4*cmath.pi*k_O_O*freqcm,linestyle='--')
	plt.plot(freqcm,4*cmath.pi*(k_O_O+k_O_A+k_A_O+k_A_A)*freqcm,linestyle='--')
	plt.plot(freqcm,4*cmath.pi*k_tot*freqcm)
	plt.legend(['A-A','A-O','O-O','A-A+O-O+A-O','all modes'])
	plt.ylabel('absorption $(cm^{-1})$')
	plt.xlabel('wave-number $(cm^{-1})$')
	plt.show()
	
	"""		"""

	if (freqRange[1]-freqRange[0])/THz > 1/5*alpha/THz:
		print('alpha:',alpha/THz, 'too small for frequency step size of:',(freqRange[1]-freqRange[0])/THz)

	
	Volume=np.linalg.det(np.array(meshdata['lattice'],dtype=float)*10**-10)
	print('Volume prefactor:',(hbar/(epsilon0*Volume*((args.dimX*args.dimY*args.dimZ)	)	)	))
	if args.Supercell:
		VolNorm=(hbar/(epsilon0*Volume*((1)	)	)	)
	else:
		VolNorm=(hbar/(epsilon0*Volume*((args.dimX*args.dimY*args.dimZ)	)	)	)

	np.save(str(name)+str('frequency_range_THz')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),freqRange/THz)
	np.save(str(name)+str('frequency_range_inverse_cm')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),freqcm)

	if args.Supercell:
		cell_divisions=categorize_unit_cells_SC(args.NAT,args.dimX,args.dimY,args.dimZ)
	else:
		cell_divisions=categorize_unit_cells(args.NAT,args.dimX,args.dimY,args.dimZ)
	
	print(cell_divisions)
	
	qyaml=rp.get_phonon_grid_properties_from_mesh(meshdata)

	dispatom=[]
	dispyaml={'Cell Parameters':{},'Rotation Matrix':[],'Displacements':{'Displaced Atom':[],'Displacement':[],'Displaced Born':[],'Displacement Direction':[]},'Phonon Parameters': {},'BornOrig':[]}
	cellyaml={'Cell Vectors':[],'Atom':[],'Mass':[],'Fractional Coordinates':[],'Cartesian Coordinates':[],'Frequencies':[],'Eigenvectors':[]}
	dispyaml['Phonon Parameters']=qyaml

	cellyaml['Cell Vectors']=np.array(dispdata['supercell']['lattice']).astype(float)

	if args.Supercell:
		NATSCell=args.NAT
	else:
		NATSCell=args.NAT*(args.dimX*args.dimY*args.dimZ)
	print('Atoms in Supercell:',NATSCell)
	print('Temperature:',args.Temperature)
	print('q-point mesh:',dispyaml['Phonon Parameters']['q-mesh'])

	for x in range(NATSCell):
		cellyaml['Mass'].append(dispdata['supercell']['points'][x]['mass'])
		cellyaml['Fractional Coordinates'].append(np.array(dispdata['supercell']['points'][x]['coordinates']))
		cellyaml['Fractional Coordinates'][x]=cellyaml['Fractional Coordinates'][x].astype(float)
		cellyaml['Cartesian Coordinates'].append(np.dot(cellyaml['Cell Vectors'], cellyaml['Fractional Coordinates'][x]))


	
	newatoms=np.linspace(1,NAT,NAT)
	
	for n in range(1,NAT):
		newatoms[n]=NATSCell//NAT*n
	newatoms[1:]=newatoms[1:]+1
	print('new indices of unit cell atoms in supercell:',newatoms)

			
	Masses=np.array(cellyaml['Mass'],dtype=float)#mass in AMU
	
	BornOrig,EpsilOrig=rp._read_born_and_epsilon_from_OUTCAR(args.material+str('/Undisplaced/OUTCAR'),NAT)
	
	if args.epsilon==None:
		Epsilon0=np.sum(EpsilOrig)/3
	else:
		Epsilon0=args.epsilon
	print('Static dielectric constant:',Epsilon0)

		
	dispyaml['Cell Parameters']=cellyaml
	dispyaml['BornOrig']=BornOrig.tolist()
	
	#dispyamldata=yaml.load(open('store_file.yaml'),Loader=BaseLoader)
	
	alpha2index=1#used to define direction of displacement
	for n in range(0,disprange):
		Born=[]
		
		dispatom.append({'atom':dispdata['displacements'][n]['atom'],'displacement':np.array(dispdata['displacements'][n]['displacement']),'BornOrig':Born	}	)
		dispyaml['Displacements']['Displaced Atom'].append(dispdata['displacements'][n]['atom']	)
		dispyaml['Displacements']['Displacement'].append(dispdata['displacements'][n]['displacement']	)
		dispyaml['Displacements']['Displacement Direction'].append(get_direction(alpha2index))		
		alpha2index+=1
		if alpha2index==4:
			alpha2index=1
	
	
	TransformationMatrix=get_TransformationMatrix(dispdata,args.negative)
	
	"""Prints the transformation matrix and the products with displacements along a,b,c to ensure they correspond to displacements in x,y,z and in that order"""
	print(TransformationMatrix)
	print(solve(TransformationMatrix,np.array(dispatom[0]['displacement'],dtype='d')))
	print(solve(TransformationMatrix,np.array(dispatom[1]['displacement'],dtype='d')))
	print(solve(TransformationMatrix,np.array(dispatom[2]['displacement'],dtype='d')))

	dispyaml['Rotation Matrix']=TransformationMatrix.tolist()
	print('Determinant cell volume',np.linalg.det(np.array(meshdata['lattice'],dtype=float)))

	"""Files necessary for calculation of two phonon DOS via tetrahedron method"""
	grid_mapping_table=np.loadtxt(args.material+str(name)+str('grid_mapping_table')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.txt'))
	grid_address=np.loadtxt(args.material+str(name)+str('grid_address')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.txt'))
	relative_grid_address=np.load(args.material+str(name)+str('relative_grid_address')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'))

	shift_factor=np.ones(len(freqRange))
	shift_factor800=np.ones(len(freqRange))
	shift_factor2000=np.ones(len(freqRange))

	'''DOS=DS.run_tetrahedron_method_dos(meshdata,freqRange,grid_address,grid_mapping_table,relative_grid_address,coef=None,plot=False)
	if args.testDOS and args.singlePhonon:		
		plt.plot(freqcm,DOS/args.NCell*(100*c/THz),color='k')
		plt.xlim(min(freqcm),max(freqcm))
		plt.xlabel('wave-number ($cm^{-1}$)')
		plt.ylabel('DOS ($cm^{-1}$/Unit Cell)')
		plt.show()'''

	
	"""Test of single phonon M"""
	if args.singlePhonon:
				
		meshdataGamma=meshdata#yaml.load(open('../../Sapphire/Orig/Sapphire_relax/Undisplaced/mesh555.yaml'),Loader=CBaseLoader)
		M_single_phonon,MUnitCell_single_phonon,MPhase_single_phonon,MPhaseUnitCell_single_phonon=PD.get_M_single_phonon(dispyaml,cellyaml,dispyaml['Phonon Parameters'],cell_divisions,material,args.testFile,args.testBorn,args.testTransformation,args.testDipole,args.testPhase,zfill=args.zfill)

		print('Single Phonon M shape:',np.shape(MPhaseUnitCell_single_phonon))
	
		MFourierPhase_single_phonon = MPhaseUnitCell_single_phonon[:,:,:,:,:].sum(axis=1)
		#print('Super cell summed dipole moment (eV):',MFourierPhase_single_phonon[0,:,:,:]/eV)


		#filename='/work/spf150030/Sapphire/LCALCEPS/OUTCAR'
		filename='/work/spf150030/Sapphire/Primitive/Unit/Undisplaced/OUTCAR'
		#filename='/work/spf150030/GaN/LCALCEPS/OUTCAR'
		'''PhononPropertiesGamma=rp._read_phonon_properties_from_OUTCAR(filename,args.NAT)
		MFourierPhase_single_phonon_Test=np.ones(np.shape(MFourierPhase_single_phonon))	
		
		ChiN_Vasp=ab.get_ChiN_single_phonon_VASP(MFourierPhase_single_phonon_Test,cell_divisions,meshdataGamma,PhononPropertiesGamma,NCell=1,testEigenvector=False,testChi=False)
		Susceptibility_single_phonon_VASP=np.absolute(ab.get_Imag_Susceptibility_single_phonon_VASP(ChiN_Vasp,PhononPropertiesGamma,0.05*THz,freqRange,alpha,shift_factor,NCell=args.NCell))
		SusceptibilitymodCubic_single_phonon_VASP=VolNorm*Susceptibility_single_phonon_VASP
		plt.plot(2*cmath.pi*freqRange/THz,SusceptibilitymodCubic_single_phonon_VASP)
		#plt.plot(2*cmath.pi*freqRange/THz,single_phonon_n)
		#plt.xlim(60,120)
		#plt.ylim(0,100)
		plt.legend(['x','y','z'])
		plt.show()'''


		ChiOne_single_phonon = ab.get_ChiN_single_phonon(MFourierPhase_single_phonon,cell_divisions,meshdataGamma,testEigenvector=args.testEigenvector,testChi=args.testEigenvector,NCell=args.NCell)

		print('Born Charge Sum (charge neutrality check)',np.sum(M_single_phonon))

		shift_factor=DOS
		shift_factor[np.isnan(shift_factor)]=0

		Susceptibility_single_phonon=np.absolute(ab.get_Imag_Susceptibility_single_phonon_linewidth(ChiOne_single_phonon,meshdataGamma,0.025*THz,freqRange,alpha,shift_factor,NCell=args.NCell))
		SusceptibilitymodCubic_single_phonon=VolNorm*Susceptibility_single_phonon.sum(axis=1)
		single_phonon_n=(-np.imag(hilbert(SusceptibilitymodCubic_single_phonon))	+(Epsilon0))

		plt.plot(2*cmath.pi*freqRange/THz,VolNorm*Susceptibility_single_phonon)
		#plt.plot(2*cmath.pi*freqRange/THz,single_phonon_n)
		#plt.xlim(60,120)
		#plt.ylim(0,100)
		plt.legend(['x','y','z'])
		plt.show()

		SusceptibilitymodCubic_single_phonon[np.isnan(SusceptibilitymodCubic_single_phonon)]=0
		np.save(str(name)+str('singe_phonon_susceptibility')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),SusceptibilitymodCubic_single_phonon)
		
		if args.plotSusceptibility:

			#vr=VASP.Vasprun(str('/work/spf150030/Sapphire/LCALCEPS/vasprun.xml'))
			#vr=VASP.Vasprun(str('/work/spf150030/GaN/LCALCEPS/vasprun.xml'))
			vr=VASP.Vasprun(str('/work/spf150030/Sapphire/Primitive/Unit/Undisplaced/vasprun.xml'))			

			energies = vr.dielectric[0]
			e1 = np.array(vr.dielectric[1])
			e1x = e1[:,0]
			e1y = e1[:,1]
			e1z = e1[:,2]
			e2 = np.array(vr.dielectric[2])
			e2x = e2[:,0]
			e2y = e2[:,1]
			e2z = e2[:,2]
			etot = (e1x+e1y+e1z)#/3
			etoti = (e2x+e2y+e2z)#/3
			#plt.plot(etoti)
			#plt.plot(-np.imag(hilbert(etoti))	+math.sqrt(3.08),color='k')
			#plt.plot(np.linspace(0,185,2000),e2x)
			#plt.plot(np.linspace(0,185,2000),e2y)
			#plt.plot(np.linspace(0,185,2000),e2z)
			plt.plot(np.linspace(0,185,2000),etoti,color='r')
			#plt.xlim(60,120)#min(freqRange)/THz,max(freqRange/THz))
			#plt.show()


			plt.plot(2*cmath.pi*freqRange/THz,SusceptibilitymodCubic_single_phonon,color='k')
			plt.plot(2*cmath.pi*freqRange/THz,single_phonon_n,color='y')
			plt.ylabel('$\chi''^{(1)}(\omega)$')
			plt.xlabel('Frequency ($2\pi$THz)')
			
			plt.legend(['$Im(\epsilon)_{total}$ VASP','Calculated','$Re(\epsilon)$'])
			plt.show()
			TotalSuscept_single_phonon=single_phonon_n+1j*SusceptibilitymodCubic_single_phonon
			TotalSuscept_Vasp=etot+1j*etoti


			plt.plot(freqcm,np.imag(np.sqrt(TotalSuscept_single_phonon)),color='k')
			#plt.plot(freqcm,np.real(np.sqrt(TotalSuscept_single_phonon)),color='r')
			plt.ylabel('$k$, $n$')
			plt.xlabel('wave-number ($\mathrm{cm^{-1}}$)')
			plt.legend(['$k$','$n$'])
			plt.show()



			SinglePhononAbsorptionVasp=(	(2*2*cmath.pi*np.linspace(0,30,2000)*THz*np.imag(np.sqrt(TotalSuscept_Vasp)	)	)/(np.sqrt(Epsilon0)*c))
			plt.plot(np.linspace(0,185,2000),SinglePhononAbsorptionVasp/100,color='r')
		SinglePhononAbsorption=((2*2*cmath.pi*freqRange*np.imag(np.sqrt(TotalSuscept_single_phonon)	)	)/(np.sqrt(Epsilon0)*c))
		print(np.shape(SinglePhononAbsorption))
		np.save(str(name)+str('singe_phonon_absorption_wavenumber')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),SinglePhononAbsorption)
		plt.plot(2*cmath.pi*freqRange/THz,SinglePhononAbsorption/100,color='k')
		plt.ylabel('Absorption $cm^{-1}$')
		plt.xlabel('Frequency ($2\pi$THz)')#$cm^{-1}$')
		plt.xlim(0,130)#min(freqcm),max(freqcm))#(min(freqcm)/10**-2,max(freqcm)/10**-2)
		#plt.ylim(0,None)
		plt.legend([' VASP','Calculated','$Re(\chi)$'])
		if args.savePlot:
			plt.savefig(str(args.Name)+str('Single Phonon Absorption')+str('Temperature.pdf'),format='pdf',dpi=400,bbox_inches='tight',pad_inches=0.2)
		if args.plotAbsorption:
			plt.show()
			T=np.exp(-SinglePhononAbsorption*(thickness*10**-6))
			plt.plot(freqcm,T)
			plt.ylabel('R,T,$\epsilon$')
			plt.xlabel('Frequency $cm^{-1}$')
			plt.xlim(min(freqcm),max(freqcm))
			plt.ylim(0,1)
			plt.show()


		print('Beginning two-phonon calculation')

	""""""

	"""Calculate 2-phonon Density of States"""
	
	TDOS=DS.get_TDOS_Tetrahedron(meshdata,freqRange,grid_address, grid_mapping_table,relative_grid_address,Overtones=False,Temperature=args.Temperature,BEFMod=False,coefPlus=None,coefMinus=None,plot=args.testDOS,NCell=args.NCell)
	np.save(str(name)+str('TDOS')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),TDOS.sum(axis=0).sum(axis=0)[:,0])

	shift_factor=np.ones(len(freqRange))#TDOS.sum(axis=0).sum(axis=0)[:,0]
	shift_factor[np.isnan(shift_factor)]=0

	if args.TempCompare:
		TDOS800=DS.get_TDOS_Tetrahedron(meshdata,freqRange,grid_address, grid_mapping_table,relative_grid_address,Overtones=False,Temperature=800,BEFMod=False,coefPlus=None,coefMinus=None,)
		shift_factor800=TDOS.sum(axis=0).sum(axis=0)[:,0]

		TDOS2000=DS.get_TDOS_Tetrahedron(meshdata,freqRange,grid_address, grid_mapping_table,relative_grid_address,Overtones=False,Temperature=2000,BEFMod=False,coefPlus=None,coefMinus=None,)
		shift_factor2000=TDOS2000.sum(axis=0).sum(axis=0)[:,0]

	""" test to ensure density of states sum gives results similar to Si 2-phonon DOS"""
	
	if args.testDOS:
		TDOS2=DS.get_TDOS(meshdata,freqRange,alpha,plot=False,BEFMod=args.testBEF)
		plt.plot(freqcm,TDOS.sum(axis=0).sum(axis=0)[:,0]*(100*c/THz),color='r')
		plt.plot(freqRange/THz,TDOS2*THz*(100*c/THz),color='k')
		plt.xlabel('wave-number ($cm^{-1}$)')
		plt.ylabel('TDOS ($cm^{-1}$/Unit Cell)')
		plt.xlim(0,max(freqRange)/THz)
		#plt.legend(['Tetrahedron','Smearing'])
		#plt.plot(freqRange/THz,TDOS2*shift_factor,color='r')
		#plt.savefig(str(args.Name)+str('TDOS')+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2])+str('.pdf'),format='pdf',dpi=400,bbox_inches='tight',pad_inches=0.2)
		plt.show()
	"""	"""


	with open(r'store_file.yaml', 'w') as file:
    		documents = yaml.dump(dispyaml, file)

	try:
		M=np.load(str(args.pathLoad)+str(name)+str('Dipole_Moment')+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('.npy'))
		MUnitCell=np.load(str(args.pathLoad)+str(name)+str('Dipole_Moment_Unit_Cell')+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('.npy'))
		MPhase=np.load(str(args.pathLoad)+str(name)+str('Dipole_Moment_Phase')+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2])+str('.npy'))
		MPhaseUnitCell=np.load(str(args.pathLoad)+str(name)+str('Dipole_Moment_Phase_Unit_Cell')+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2])+str('.npy'))
		print('Dipole moment files found, loading now...')
	except IOError:
		print('No dipole moment file found, calculating second order dipole moments')
	
		M,MUnitCell,MPhase,MPhaseUnitCell=PD.get_Ms(dispyaml,cellyaml,dispyaml['Phonon Parameters'],cell_divisions,material,args.testFile,args.testBorn,args.testTransformation,args.testDipole,args.testPhase,zfill=args.zfill)

		np.save(str(args.pathLoad)+str(name)+str('Dipole_Moment')+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('.npy'),M)
		np.save(str(args.pathLoad)+str(name)+str('Dipole_Moment_Unit_Cell')+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('.npy'),MUnitCell)
		np.save(str(args.pathLoad)+str(name)+str('Dipole_Moment_Phase')+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2])+str('.npy'),MPhase)
		np.save(str(args.pathLoad)+str(name)+str('Dipole_Moment_Phase_Unit_Cell')+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2])+str('.npy'),MPhaseUnitCell)
	
	print('Unit Cell dividied:',np.shape(MPhaseUnitCell))
		
	
	MFourierPhase=np.sum(np.sum(MPhaseUnitCell,axis=1),axis=2)
	
	"""Sums over all displacements in the first unit cell (the only displaced atoms) and multiplies by the number of unit cells (possible because of the periodic structure of the material)"""
	"""Effectively N*Sum_(cellindex1) M^(cellindex1, atom1, cellindex2=0,atom2)_(Edirection,direction1,direction2)(q)"""
	MFourierPhaseTest=np.zeros(np.shape(MFourierPhase),dtype=complex)
	if args.testPhase:
			print(MPhaseUnitCell[0,0,:,0,:,:,:]/angstrom)

	for qn, nq in enumerate(np.array(dispyaml['Phonon Parameters']['q-point']).astype(float)):
		for R1 in range(len(MPhaseUnitCell[0,:,0,0,0,0,0,0])):
			for k1 in range(len(MPhaseUnitCell[0,0,:,0,0,0,0,0])):
				for k2 in range(len(MPhaseUnitCell[0,0,0,0,:,0,0,0])):
					for displacement_direction in range(3):
						TMat=TransformationMatrix/np.sqrt(float(dispyaml['Displacements']['Displacement'][0][0])**2+float(dispyaml['Displacements']['Displacement'][0][1])**2+float
		(dispyaml['Displacements']['Displacement'][0][2])**2)
						phase_factor=PD.get_phase_factor(np.array(nq).astype(float),cellyaml['Fractional Coordinates'][k1],(cellyaml['Fractional Coordinates'][k2]))

						#phase_factor_neg=PD.get_phase_factor(-np.array(nq).astype(float),cellyaml['Fractional Coordinates'][k1],(cellyaml['Fractional Coordinates'][k2]))
						
						#print('positive q',MUnitCell[R1,k1,0,k2,:,:,displacement_direction]*phase_factor)
						#print('negative q',MUnitCell[R1,k1,0,k2,:,:,displacement_direction]*phase_factor_neg)

						MFourierPhaseTest[qn,k1,k2,:,:,displacement_direction]+=args.dimX*args.dimY*args.dimZ*MUnitCell[R1,k1,0,k2,:,:,displacement_direction]*phase_factor

	FineQgrid=np.array(meshdata['mesh'],dtype=int)
	dimFine=np.prod(FineQgrid)
	
	cell_indices=np.linspace(1,args.dimX*args.dimY*args.dimZ,args.dimX*args.dimY*args.dimZ,dtype=int)
	print('Cell indices:',cell_indices)
	
	cell_indices_reshape_roll=np.roll(cell_indices.reshape(args.dimX,args.dimY,args.dimZ),(args.dimX//2,args.dimY//2,args.dimZ//2),axis=(0,1,2)).reshape(args.dimX*args.dimY*args.dimZ)
	print(cell_indices_reshape_roll)

	M_Fine=np.zeros((dimFine,args.NAT,args.dimX*args.dimY*args.dimZ,args.NAT,3,3,3),dtype=complex)

	M_Fine[0:args.dimX*args.dimY*args.dimZ,:,:,:,:,:,:]=MUnitCell

	M_Fine[0:args.dimX*args.dimY*args.dimZ,:,:,:,:,:,:]=M_Fine[cell_indices_reshape_roll-1,:,:,:,:,:,:]

	MFourierPhase_Fine=np.zeros(np.shape(MFourierPhase),dtype=complex)

	VolNorm=(hbar/(epsilon0*Volume*((dimFine)	)	)	)

	print('Fine Volume Prefactor:',VolNorm)

	for qn, nq in enumerate(np.array(dispyaml['Phonon Parameters']['q-point']).astype(float)):
		for R1 in range(len(M_Fine[:,0,0,0,0,0,0])):
			for k1 in range(len(M_Fine[0,:,0,0,0,0,0])):
				for k2 in range(len(M_Fine[0,0,0,:,0,0,0])):
					for displacement_direction in range(3):
						TMat=TransformationMatrix/np.sqrt(float(dispyaml['Displacements']['Displacement'][0][0])**2+float(dispyaml['Displacements']['Displacement'][0][1])**2+float
		(dispyaml['Displacements']['Displacement'][0][2])**2)
						phase_factor=PD.get_phase_factor(np.array(nq).astype(float),cellyaml['Fractional Coordinates'][k1],(cellyaml['Fractional Coordinates'][k2]))

						MFourierPhase_Fine[qn,k1,k2,:,:,displacement_direction]+=dimFine*M_Fine[R1,k1,0,k2,:,:,displacement_direction]*phase_factor

	#NCell=dimFine/args.dimX*args.dimY*args.dimZ


	"""Check to ensure that nonzero elements are all in index zero of displaced atoms"""
	'''MFourierPhaseC=np.zeros(np.shape(MFourierPhase),dtype=complex)
	for R1 in range(len(MPhaseUnitCell[0,:,0,0,0,0,0,0])):
		MFourierPhaseC+=args.dimX*args.dimY*args.dimZ*MPhaseUnitCell[:,R1,:,0,:,:,:,:]'''
	print('Fourier transformed:',np.shape(MFourierPhase))
	print(np.mean(MFourierPhase-MFourierPhaseTest))
	print('cell_divisions shape:',np.shape(cell_divisions))	
	
	"""displays selected sections of the dipole moment element"""
	print(MUnitCell[:,0,0,:,:,:,0].sum(axis=0)/angstrom	)

	'''MUnitCellFlat=(np.sort(MUnitCell[:,:,0,:,:,:].flatten()/angstrom))
	plt.plot(MUnitCellFlat)
	plt.ylabel('M ($C/m$)')
	plt.show()'''

	"""Calculates the 2nd order dipole moments as a function of phonon quantum branch index"""
	ChiOne=ab.get_ChiN(MFourierPhase_Fine,cell_divisions,meshdata,NCell=dimFine,testEigenvector=args.testEigenvector,testChi=args.testChi)
	print('Chi Shape',np.shape(ChiOne))



	"""Tetrahedron-based Suscepbibility calculation"""#0.48/300*args.Temperature*THz
	Susceptibility=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,args.Temperature,freqRange,alpha,shift_factor,LinewidthSimulate=True))
	
	'''Susceptibility350=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,350,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility400=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,400,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility450=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,450,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility500=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,500,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility550=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,550,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility800=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,800,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility650=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,650,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility600=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,600,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility700=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,700,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility750=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,750,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])
	Susceptibility2000=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,2000,freqRange,alpha,shift_factor,LinewidthSimulate=True)[200,0,0])

	SusceptTemp=VolNorm*np.array([Susceptibility[200,0,0],Susceptibility350,Susceptibility400,Susceptibility450,Susceptibility500,Susceptibility550,Susceptibility600,Susceptibility650,Susceptibility700,Susceptibility750,Susceptibility800,Susceptibility2000])
	plt.plot([args.Temperature,350,400,450,500,550,600,650,700,750,800,2000],np.sqrt(SusceptTemp))
	plt.show()
	plt.plot(np.diff(SusceptTemp)/50)'''

	if args.TempCompare:
		Susceptibility800=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,800,freqRange,alpha,shift_factor,LinewidthSimulate=True))
		Susceptibility2000=np.absolute(ab.get_Imag_Susceptibility_NonCubic(ChiOne,meshdata,2000,freqRange,alpha,shift_factor,LinewidthSimulate=True))

	print('Noncubic Susceptibility Shape',np.shape(Susceptibility))

	"""Multiplies susceptibility by hbar/Vepsilon0 calculated above"""
	Susceptibilitymod=np.zeros(len(freqRange))
	if args.TempCompare:
		Susceptibilitymod800=np.zeros(len(freqRange))
		Susceptibilitymod2000=np.zeros(len(freqRange))

	for i in range(3):
		Susceptibilitymod+=Susceptibility[:,i,i]
		if args.TempCompare:
			Susceptibilitymod800+=Susceptibility800[:,i,i]
			Susceptibilitymod2000+=Susceptibility2000[:,i,i]
		

	Susceptibilitymod=VolNorm*(Susceptibilitymod)#Susceptibility.sum(axis=1).sum(axis=1)

	if args.TempCompare:
		Susceptibilitymod800=VolNorm*(Susceptibilitymod800)
		Susceptibilitymod2000=VolNorm*(Susceptibilitymod2000)

		Real_SusceptibilityNonCubic800=-np.imag(hilbert(Susceptibilitymod800)) + Epsilon0
		Total_Suscept_Noncubic800=Real_SusceptibilityNonCubic800+1j*Susceptibilitymod800

		Real_SusceptibilityNonCubic2000=-np.imag(hilbert(Susceptibilitymod2000)) + Epsilon0
		Total_Suscept_Noncubic2000=Real_SusceptibilityNonCubic2000+1j*Susceptibilitymod2000

	"""Calculates the real susceptibility by performing a hilbert transform on the imaginary part of the susceptibility"""
	Real_SusceptibilityNonCubic=-np.imag(hilbert(Susceptibilitymod)) + Epsilon0
	Total_Suscept_Noncubic=Real_SusceptibilityNonCubic+1j*Susceptibilitymod

	np.save(str(name)+str(args.Temperature)+str('K')+str('two_phonon_susceptibility_NC')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),Susceptibilitymod)

	"""implements the following equation to calculate the absorption"""
	"""alpha(omega) = (2 k omega)/c"""
	twoPhononAbsorption_NC=((2*2*cmath.pi*freqRange*np.imag(np.sqrt(Total_Suscept_Noncubic)	)	)/(c))#*np.real(np.sqrt(Total_Suscept_Noncubic)	)

	if args.TempCompare:
		twoPhononAbsorption800_NC=((2*2*cmath.pi*freqRange*np.imag(np.sqrt(Total_Suscept_Noncubic800)	))/(c))
		twoPhononAbsorption2000_NC=((2*2*cmath.pi*freqRange*np.imag(np.sqrt(Total_Suscept_Noncubic2000)	))/(c))
		np.save(str(name)+str(800)+str('K')+str('two_phonon_absorption_NC')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),twoPhononAbsorption800_NC)
		np.save(str(name)+str(2000)+str('K')+str('two_phonon_absorption_NC')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),twoPhononAbsorption2000_NC)

	np.save(str(name)+str(args.Temperature)+str('K')+str('two_phonon_absorption_NC')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),twoPhononAbsorption_NC)


	plt.figure()
	plt.plot(freqcm,twoPhononAbsorption_NC/(100),color='k')
	plt.legend(['Calculated Model','Experimental','Speculative 800 K', 'Speculative 2000 K'])
	if args.TempCompare:
		plt.plot(freqcm,twoPhononAbsorption800_NC/(100),color='b')
		plt.plot(freqcm,twoPhononAbsorption2000_NC/(100),color='y')
		plt.legend([str(args.Temperature),'800','2000'])
	plt.ylabel('Absorption ($cm^{-1}$)')
	plt.xlabel('Frequency ($cm^{-1}$)')
	plt.xlim(min(freqcm),max(freqcm))
	plt.ylim(10**-2,None)
	plt.show()

	Cref_wavenumber, Cref_absorption=get_CSV_Absorption('Diamond Multiphonon Absorption (1).csv')
	Cref_wavenumber_model, Cref_absorption_model=get_CSV_Absorption('Diamond Model Absorption.csv')

	plt.plot(freqcm,twoPhononAbsorption_NC/(100),color='k')
	plt.plot(np.array(Cref_wavenumber),np.array(Cref_absorption),color='r')
	plt.plot(np.array(Cref_wavenumber_model),np.array(Cref_absorption_model),color='b')
	#plt.plot(np.array(nuref),np.array(alpharef),color='g')
	plt.xlim(min(freqcm),max(freqcm))
	plt.ylim(10**-2,None)
	plt.title(str(name)+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2]))
	plt.legend(['Calculated Model','Reference','$Thomas~et~al$', 'Speculative 2000 K'])
	plt.show()



	LinewidthSimulate=args.LinewidthSimulate
	alpha=0.48/300*args.Temperature*0.60	
	alphas=np.ones((int(meshdata['nqpoint']),3*int(meshdata['natom']),3*int(meshdata['natom'])),dtype=float)*args.alpha*THz

	SusceptibilityCubic=np.absolute(ab.get_Imag_Susceptibility(ChiOne,meshdata,args.Temperature,freqRange,alphas,shift_factor,n=2,NCell=args.NCell,LinewidthSimulate=LinewidthSimulate,testQpoint=args.testQpoint))#*args.dimX*args.dimY*args.dimZ/np.prod(np.array(dispyaml['Phonon Parameters']['q-mesh'],dtype=int))
	
	if args.TempCompare:


		Susceptibility800=np.absolute(ab.get_Imag_Susceptibility(ChiOne,meshdata,800,freqRange,alphas,shift_factor800,n=2,LinewidthSimulate=False))

		Susceptibility2000=np.absolute(ab.get_Imag_Susceptibility(ChiOne,meshdata,2000,freqRange,alphas,shift_factor2000,n=2,LinewidthSimulate=False))
	
	
	
	
	SusceptibilitymodCubic=VolNorm*SusceptibilityCubic.sum(axis=1)/3#np.sqrt(SusceptibilityCubic[:,0]**2+SusceptibilityCubic[:,1]**2+SusceptibilityCubic[:,2]**2)#
	Real_SusceptibilityCubic=-np.imag(hilbert(SusceptibilitymodCubic)) + Epsilon0

	if args.TempCompare:
		Susceptibility800mod=VolNorm*Susceptibility800.sum(axis=1)
		Susceptibility2000mod=VolNorm*Susceptibility2000.sum(axis=1)

	np.save(str(name)+str(args.Temperature)+str('K')+str('two_phonon_susceptibility')+str(meshdata['mesh'][0])+str(meshdata['mesh'][1])+str(meshdata['mesh'][2])+str('.npy'),SusceptibilitymodCubic)

	if args.plotSusceptibility:
		plt.plot(freqcm,Susceptibilitymod,color='k')
		plt.plot(freqcm,Real_SusceptibilityNonCubic-Epsilon0,color='r')
		plt.ylabel('$\chi^{''(2)}(\omega)$')
		plt.xlabel('wave-number ($\mathrm{cm^{-1}}$)')
		plt.legend(['Im($\chi$)','Re($\chi$)'])
		plt.xlim(0,max(freqcm))
		#plt.ylim()
		plt.savefig(str(args.Name)+str('Susceptibility')+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2])+str('.pdf'),format='pdf',dpi=400,bbox_inches='tight',pad_inches=0.2)
		plt.show()
	Total_Suscept=Real_SusceptibilityCubic+1j*SusceptibilitymodCubic

	if args.plotSusceptibility:
		plt.plot(freqcm,np.imag(np.sqrt(Total_Suscept)),color='k')
		plt.plot(freqcm,np.real(np.sqrt(Total_Suscept)),color='r')
		plt.ylabel('$k$, $n$')
		plt.xlabel('wave-number ($\mathrm{cm^{-1}}$)')
		plt.legend(['$k$','$n$'])
		plt.title(str(name)+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2]))
		plt.show()

	"""implements the following equation to calculate the absorption"""
	"""alpha(omega) = (2 k omega)/c"""

	twoPhononAbsorption=((2*2*cmath.pi*freqRange*np.imag(np.sqrt(Total_Suscept)	)	)/(c))#*np.real(np.sqrt(Total_Suscept))

	#np.savetxt('Two-phonon_absorption_121212_Fine_grid.txt',twoPhononAbsorption)
	two_phonon_fine=np.loadtxt('Two-phonon_absorption_121212_Fine_grid.txt')
		
	if args.TempCompare:
		twoPhononAbsorption800=((2*2*cmath.pi*freqRange*np.sqrt(Susceptibility800mod))/(c))#*(100*c/THz)
		twoPhononAbsorption2000=((2*2*cmath.pi*freqRange*np.sqrt(Susceptibility2000mod))/(c))#*(100*c/THz)
		

	if args.smooth:	
		p=args.polyorder
		w=args.window
		twoPhononAbsorption=savgol_filter(twoPhononAbsorption,w,p)
		if args.TempCompare:		
			twoPhononAbsorption800=savgol_filter(twoPhononAbsorption800,w,p)
			twoPhononAbsorption2000=savgol_filter(twoPhononAbsorption2000,w,p)


	"""Plotting absorption and comparing to experimental results"""
	plt.semilogy(freqcm,twoPhononAbsorption/100,color='k')

	#plt.semilogy(freqcm,twoPhononAbsorptionNonCubic/100,color='b')

	if args.TempCompare:
		plt.semilogy(freqcm,twoPhononAbsorption800/100,color='r')
		plt.semilogy(freqcm,twoPhononAbsorption2000/100,color='y')
	
	
	if args.singlePhonon:
		plt.semilogy(freqcm,SinglePhononAbsorption/100,color='b')
		plt.semilogy(freqcm,twoPhononAbsorption/100+SinglePhononAbsorption/100,color='g')

	plt.title(str(name)+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2]))
	plt.ylabel('Absorption ($cm^{-1}$)')
	plt.xlabel('Frequency ($cm^{-1}$)')
	plt.xlim(min(freqcm),max(freqcm))#(min(freqcm)/10**-2,max(freqcm)/10**-2)
	plt.ylim(10**-2,None)
	if args.TempCompare:
		plt.legend([ str(args.Temperature)+str('K'),'800 K','2000 K',str('1-phonon'),'$ 1-phonon+2-phonon$'])
	else:	
		plt.legend([ str(args.Temperature)+str('K'),str('1-phonon'),'$ 1-phonon+2-phonon$'])
	if args.savePlot:
		plt.savefig(str(args.Name)+str('Absorption')+str('Temperature.pdf'),format='pdf',dpi=400,bbox_inches='tight',pad_inches=0.2)
	if args.plotAbsorption:
		plt.show()

	nuref,alpharef=get_CSV_Absorption('Si 2-Phonon Model Absorption.csv')
	nuexp,alphaexp=get_CSV_Absorption('Si 2-Phonon Experimental Absorption.csv')

	if args.TempCompare:
		Ratio800=np.mean(twoPhononAbsorption800[twoPhononAbsorption800!=0]/twoPhononAbsorption[twoPhononAbsorption!=0])#[0:len(twoPhononAbsorption800)//len(nuref):]
		Ratio2000=np.mean(twoPhononAbsorption2000[twoPhononAbsorption2000!=0]/twoPhononAbsorption[twoPhononAbsorption!=0])#[0:len(twoPhononAbsorption2000)//len(nuref):]
	

	plt.figure()
	plt.plot(freqcm,twoPhononAbsorption/(100),color='b')
	plt.plot(np.array(nuexp),np.array(alphaexp),color='r')
	#plt.plot(freqcm,twoPhononAbsorption_NC/(100),color='k')
	#plt.plot(freqcm,two_phonon_fine/(100),color='k')
	#plt.plot(np.array(Cref_wavenumber),np.array(Cref_absorption),color='r')
	
	#plt.plot(np.array(nuref),np.array(alpharef),color='g')
	plt.xlim(min(freqcm),max(freqcm))
	plt.ylim(10**-2,20)
	plt.title(str(name)+str(args.dimX)+str(args.dimY)+str(args.dimZ)+str('q_grid')+str(dispyaml['Phonon Parameters']['q-mesh'][0])+str(dispyaml['Phonon Parameters']['q-mesh'][1])+str(dispyaml['Phonon Parameters']['q-mesh'][2]))
	plt.legend(['Calculated Model','Experimental','Calculated Model Approx', 'Speculative 2000 K'])
	plt.show()


	T2=np.exp(-twoPhononAbsorption*(thickness*10**-6))
	plt.plot(freqcm,T2)
	plt.ylabel('R,T,$\epsilon$')
	plt.xlabel('Frequency $cm^{-1}$')
	plt.xlim(min(freqcm),max(freqcm))
	#plt.ylim(0,1)
	plt.show()
	
	
	np.savetxt('FreqAbsorp.txt',freqcm)
	np.savetxt('twoPhononAbsorption.txt',twoPhononAbsorption/100)

	ThreeD_Scatter(NAT,args.dimX,args.dimY,args.dimZ,np.array(cellyaml['Cartesian Coordinates'],dtype=float)*angstrom,cellyaml['Fractional Coordinates'],np.absolute(M[0,:,0,0,0]),np.array(cellyaml['Cell Vectors']))		

	

import Read_Phonon_Properties as rp
import numpy as np
from numpy.linalg import solve,inv
import MultiphononParameters as mp
from MultiphononParameters import *
import cmath
import math
import scipy
import matplotlib.pyplot as plt
import yaml
from yaml import load, dump, BaseLoader, SafeLoader
import time
import os


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

def get_phase_factor(q,R1,R2):
	phase_factor=cmath.exp(1j*q.dot((R1-R2)))
	return phase_factor

def get_phase_factor_single_phonon(q,R1):
	phase_factor=cmath.exp(1j*q.dot((R1)))
	return phase_factor

def get_phase_factor_phonopy(q,R_1,R_2,F):#positions in fractional coordinates
	primitiveconversion=2*cmath.pi*np.dot(R_2-R_1,F)
	phase_factor_phonopy=cmath.exp(1j*np.dot(q,primitiveconversion))
	
	return phase_factor_phonopy

def get_Unit_Cell_M(M,cell_divisions,testCellDivisions=False,PhaseCalc=False, testMatches=False,n=2):
	
	"""Test is only for silicon, general test may be implemented later"""
	Matches1=np.full((len(cell_divisions[:,0]),len(cell_divisions[0,:])),False)
	Matches2=np.full((len(cell_divisions[:,0]),len(cell_divisions[0,:])),False)
	
	
	if len(cell_divisions[0,:])==1:
		print('Unit Cell Calculation')

	if PhaseCalc==True:
		"""indices of MUnitCell correspond (in 2 phonon case) to unit cell-1, atom in unit cell-1, unit cell-2, atom in unit cell-2 , remaining are the same as standard M"""
		MUnitCell=np.zeros((len(M[:,0,0,0,0,0]),len(cell_divisions[:,0]),len(cell_divisions[0,:]),len(cell_divisions[:,0]),len(cell_divisions[0,:]),3,3,3),dtype=complex)
		for qphon in range(len(M[:,0,0,0,0,0])):
			for CellIndexOne in range(len(cell_divisions[:,0])):
				for j in range(len(cell_divisions[0,:])):
					for l in range(len(cell_divisions[:,0])):
						for m in range(len(cell_divisions[0,:])):
							for k in range(len(cell_divisions[0,:])*len(cell_divisions[:,0])):
								for n in range(len(cell_divisions[0,:])*len(cell_divisions[:,0])):
									#print(k,cell_divisions[i,j],n,cell_divisions[l,m])
									if k+1==cell_divisions[CellIndexOne,j] and n+1==cell_divisions[l,m]:
										Matches1[CellIndexOne,j]=True
										Matches2[l,m]=True
										#print(k,cell_divisions[i,j])
										#print(n,cell_divisions[l,m])
										MUnitCell[:,CellIndexOne,j,l,m,:,:,:]=M[:,k,n,:,:,:]

	else:
		"""indices of MUnitCell correspond (in 2 phonon case) to unit cell-1, atom in unit cell-1, unit cell-2, atom in unit cell-2 , remaining are the same as standard M"""
		MUnitCell=np.zeros((len(cell_divisions[:,0]),len(cell_divisions[0,:]),len(cell_divisions[:,0]),len(cell_divisions[0,:]),3,3,3),dtype=complex)

		for CellIndexOne in range(len(cell_divisions[:,0])):
			for j in range(len(cell_divisions[0,:])):
				for l in range(len(cell_divisions[:,0])):
					for m in range(len(cell_divisions[0,:])):
						for k in range(len(cell_divisions[0,:])*len(cell_divisions[:,0])):
							for n in range(len(cell_divisions[0,:])*len(cell_divisions[:,0])):
								if k+1==cell_divisions[CellIndexOne,j] and n+1==cell_divisions[l,m]:
									Matches1[CellIndexOne,j]=True
									Matches2[l,m]=True
									MUnitCell[CellIndexOne,j,l,m,:,:,:]=M[k,n,:,:,:]
	if testMatches==True and n==2:						
		print(Matches1,Matches2)
	return MUnitCell


def get_Unit_Cell_M_single_phonon(M,cell_divisions,testCellDivisions=False,PhaseCalc=False, testMatches=False,n=2):
	
	"""Test is only for silicon, general test may be implemented later"""
	Matches1=np.full((len(cell_divisions[:,0]),len(cell_divisions[0,:])),False)
	
	
	if len(cell_divisions[0,:])==1:
		print('Unit Cell Calculation')

	if PhaseCalc==True:
		"""indices of MUnitCell correspond (in 2 phonon case) to unit cell-1, atom in unit cell-1, unit cell-2, atom in unit cell-2 , remaining are the same as standard M"""
		MUnitCell=np.zeros((len(M[:,0,0,0]),len(cell_divisions[:,0]),len(cell_divisions[0,:]),3,3),dtype=complex)
		for qphon in range(len(M[:,0,0,0])):
			for CellIndexOne in range(len(cell_divisions[:,0])):
				for j in range(len(cell_divisions[0,:])):
					for k in range(len(cell_divisions[0,:])*len(cell_divisions[:,0])):
						if k+1==cell_divisions[CellIndexOne,j]:
							Matches1[CellIndexOne,j]=True
							MUnitCell[:,CellIndexOne,j,:,:]=M[:,k,:,:]

	else:
		"""indices of MUnitCell correspond (in 2 phonon case) to unit cell-1, atom in unit cell-1, unit cell-2, atom in unit cell-2 , remaining are the same as standard M"""
		MUnitCell=np.zeros((len(cell_divisions[:,0]),len(cell_divisions[0,:]),3,3),dtype=complex)

		for CellIndexOne in range(len(cell_divisions[:,0])):
			for j in range(len(cell_divisions[0,:])):
				for k in range(len(cell_divisions[0,:])*len(cell_divisions[:,0])):
					if k+1==cell_divisions[CellIndexOne,j]:
						Matches1[CellIndexOne,j]=True
						MUnitCell[CellIndexOne,j,:,:]=M[k,:,:]
						print('Supercell atom:',k+1,'unit cell:',CellIndexOne+1,'Atom in unit cell:',j+1)
						print('Dipole element(eV):',MUnitCell[CellIndexOne,j,:,:]/eV)
	if testMatches==True and n==2:						
		print(Matches1)
	return MUnitCell


def get_Ms(displacement_parameters,cell_parameters,phonon_parameters,cell_divisions,material,testFile=False,testBorn=False,testTransformation=False,testDipole=False,testPhase=False,testUnitCell=False,unitCellSum=False,zfill=2):
	"""Indices of M standard M are the considered atom, the displaced atom, the electric field direction, the considered direction, and the displaced direction, respectively"""

	""""This function calculates the position dependent 2nd order dipole moments and their Fourier transform by the following relations"""
	"""M^(atom1, atom2)_(Edirection, direction1, direction2)(Position1, Position2)=d^3(Energy)/dE_Edirection du^atom1_direction1(Position1) du^atom2_direction2(Position2)"""
	"""M^(atom1, atom2)_(Edirection, direction1, direction2)(q)=M^(atom1, atom2)_(Edirection, direction1, direction2)(Position1, Position2) e^(q dot (Position1-Position2))"""
	"""These are also recategorized from supercell atoms to (unit cell, atom) index structure """

	saveBornChange=True
	disprange=len(displacement_parameters['Displacements']['Displacement'])
	NAT=len(cell_divisions[0,:])
	NATSCell=NAT*len(cell_divisions[:,0])
	M=np.zeros((NATSCell,NATSCell,3,3,3),dtype=float)

	"""MPhase is the same as M, but with the phase factor for each q point considered, so a leading index of that q-point is added"""
	MPhase=np.zeros((len(phonon_parameters['q-point']),NATSCell,NATSCell,3,3,3),dtype=complex)
	MUnitCellSum=np.zeros((NAT,NAT,3,3,3),dtype=complex)
	MUnitCellSumPhase=np.zeros(np.shape(MUnitCellSum))

	TransformationMatrix=np.array(displacement_parameters['Rotation Matrix'])
	BornOrig=np.array(displacement_parameters['BornOrig'])
	#if testBorn==True:
	#	print('Undisplaced Born Effective Charges',BornOrig)
		
	if testTransformation==True:
		print(TransformationMatrix)


	for n in range(1,disprange+1):
		num=str(n)
		"""May need to adjust zfill based on the number of displacements"""
		filename=str(material)+str('/disp-0')+str(num.zfill(zfill))+str('/OUTCAR')
		if testFile==True:
			print(filename)		
		#try:
		#	Born,Epsil
		#except IOError:
		Born,Epsil=rp._read_born_and_epsilon_from_OUTCAR(filename,NAT)
		
		displacement_parameters['Displacements']['Displaced Born'].append(Born.tolist())
		amplitude=np.sqrt(float(displacement_parameters['Displacements']['Displacement'][disprange-1][0])**2+float(displacement_parameters['Displacements']['Displacement'][disprange-1][1])**2+float
		(displacement_parameters['Displacements']['Displacement'][disprange-1][2])**2)

		ScaledTmat=(TransformationMatrix)
		if testTransformation == True:
			print('Displacement Amplitude:',amplitude)
			print('Scaled Transformation Matrix',ScaledTmat)

		#print('Displacement distance:',amplitude)
		#qcount=0
		#for qn, nq in enumerate(np.array(phonon_parameters['q-point']).astype(float)):
		for atom in range(len(Born[:,0,0])):
			if testBorn==True and qcount==0:			
				print('Born for atom',atom+1,':',Born[atom,:,:],'displacement',n)
				print('Change in Born Effective Charge Tensor:',Born[atom,:,:]-BornOrig[atom,:,:])
				if saveBornChange==True:
					np.save('./Data_Files/BornDisplacement'+str(disprange+1)+str('atom')+str(atom)+str('.npy'),Borns)
					
				
			DipoleMomentElement=-solve(ScaledTmat,	(Born[atom,:,:]-BornOrig[atom,:,:])	)*eV/(angstrom)#
					

			M[atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,:,:,int(get_alpha2index(displacement_parameters['Displacements']['Displacement Direction']
			[n-1]))]=DipoleMomentElement

			if testDipole==True and qcount==0 and testPhase==False:
				print('Rotated Born Charge Change (C/angstrom) from displacement by',displacement_parameters['Displacements']['Displacement'][n-1],'A','of atom',displacement_parameters
				['Displacements']['Displaced Atom'][n-1],':')
				print(DipoleMomentElement/angstrom)
				print('alpha2 index:',int(get_alpha2index(displacement_parameters['Displacements']['Displacement Direction'][n-1])))
				print('original atom:',atom+1)
				print('displaced atom:',int(displacement_parameters['Displacements']['Displaced Atom'][n-1]))
				print('M^{k_1,k_2}_{Edirection,alpha1,alpha2} No Phase (C/angstrom)')
				print('Edirection = x')
				print(M[atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,0,:,:]/angstrom)
				print('Edirection = y')
				print(M[atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,1,:,:]/angstrom)
				print(f"Edirection = z")
				print(M[atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,2,:,:]/angstrom,"\n")


			
				"""Units of q-point are assumed to be of inverse angstrom scale (inverse of cell parameter coordinates)"""
				#phase_factor=get_phase_factor(np.array(nq).astype(float)/angstrom,cell_parameters['Cartesian Coordinates'][atom]*angstrom,ScaledTmat.dot(cell_parameters['Cartesian Coordinates'][int
				#(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1]*angstrom))
				 
				"""Applying phase factor to calculated dipole moment"""
				#MPhase[qn,atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,:,:,int(get_alpha2index(displacement_parameters['Displacements']['Displacement Direction']
				#[n-1]))]=DipoleMomentElement*phase_factor

				#qcount+=1

				'''if testPhase==True:
					if int(get_alpha2index(displacement_parameters['Displacements']['Displacement Direction'][n-1]))==2:
						print('q-point:',nq,cell_parameters['Fractional Coordinates'][atom],cell_parameters['Fractional Coordinates'][int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1],'Phase factor:',phase_factor)
						print('M^{k_1,k_2}_{Edirection,alpha1,alpha2} No Phase')
						print('Edirection = x')
						print(M[atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,0,:,:])
						print('Edirection = y')
						print(M[atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,1,:,:])
						print(f"Edirection = z")
						print(M[atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,2,:,:],"\n")

						print('Dipole moment with phase (angstrom units):')
						#print(MPhase[qn,atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,:,:,int(get_alpha2index(displacement_parameters['Displacements']['Displacement Direction'][n-1]))])
						print('Ex')
						print(MPhase[qn,atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,0,:,:])
						print('Ey')
						print(MPhase[qn,atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,1,:,:])
						print('Ez')
						print(MPhase[qn,atom,int(displacement_parameters['Displacements']['Displaced Atom'][n-1])-1,2,:,:])
						time.sleep(0.5)'''

	"""Dividing M from N-supercell,N-supercell into N-basis N-cell, N-basis N-cell"""
	MUnitCell=get_Unit_Cell_M(M,cell_divisions,True)
	MUnitCellPhase=get_Unit_Cell_M(MPhase,cell_divisions,PhaseCalc=True,testMatches=testUnitCell)
	
	return M,MUnitCell,MPhase,MUnitCellPhase

def get_M_single_phonon(displacement_parameters,cell_parameters,phonon_parameters,cell_divisions,material,testFile=False,testBorn=False,testTransformation=False,testDipole=False,testPhase=False,testUnitCell=False,unitCellSum=False,zfill=2):
	"""Indices of M standard M are the considered atom, the displaced atom, the electric field direction, the considered direction, and the displaced direction, respectively"""

	NAT=len(cell_divisions[0,:])
	NATSCell=NAT*len(cell_divisions[:,0])
	M=np.zeros((NATSCell,3,3),dtype=float)

	"""MPhase is the same as M, but with the phase factor for each q point considered, so a leading index of that q-point is added"""
	MPhase=np.zeros((len(phonon_parameters['q-point']),NATSCell,3,3),dtype=complex)
	MUnitCellSum=np.zeros((NAT,NAT,3,3),dtype=complex)
	MUnitCellSumPhase=np.zeros(np.shape(MUnitCellSum))

	TransformationMatrix=np.array(displacement_parameters['Rotation Matrix'])
	BornOrig=np.array(displacement_parameters['BornOrig'])
	if testBorn==True:
		print('Undisplaced Born Effective Charges',BornOrig)
		
	if testTransformation==True:
		print(TransformationMatrix)

				
	for qn, nq in enumerate(np.array(phonon_parameters['q-point']).astype(float)):
		for atom in range(len(BornOrig[:,0,0])):
			amplitudes=cell_parameters['Cartesian Coordinates'][atom]	
			DipoleMomentElement=-(BornOrig[atom,:,:])*eV

			if testDipole==True and qn==0:
				print('Dipole Moment (eV units):')
				print(DipoleMomentElement/eV)
				print('original atom:',atom+1)
				

			M[atom,:,:]=DipoleMomentElement
			
			"""Units of q-point are assumed to be of inverse angstrom scale (inverse of cell parameter coordinates)"""
			phase_factor=get_phase_factor_single_phonon(np.array(nq).astype(float)/angstrom,cell_parameters['Cartesian Coordinates'][atom]*angstrom)
				 
			"""Applying phase factor to calculated dipole moment"""
			MPhase[qn,atom,:,:]=DipoleMomentElement*phase_factor
			if testPhase==True:
				print('q-point:',nq,'Phase factor:',phase_factor)
				print('Dipole moment with phase (eV units):')
				print(MPhase[qn,atom,:,:]/eV)

	"""Dividing M from N-supercell,N-supercell into N-basis N-cell, N-basis N-cell"""
	MUnitCell=get_Unit_Cell_M_single_phonon(M,cell_divisions,False)
	MUnitCellPhase=get_Unit_Cell_M_single_phonon(MPhase,cell_divisions,PhaseCalc=True,testMatches=testUnitCell)#time permitting, write single phonon unit cell functions
	
	return M,MUnitCell,MPhase,MUnitCellPhase




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
import os

def _read_born_and_epsilon_from_OUTCAR(filename,nat):#if I ever make this an object, nat will be absorbed into self
    with open(filename) as outcar:
        borns = []
        epsilon = []
        directlatticevectors=[]
        cartesianpositions=[]
        fractionalpositions=[]
        num_atom=nat
        while True:
            line = outcar.readline()
            #print(line)
            if not line:
                break

            if "NIONS" in line:
                num_atom = int(line.split()[11])       

            if "MACROSCOPIC STATIC DIELECTRIC TENSOR" in line:
                epsilon = []
                outcar.readline()
                epsilon.append([float(x) for x in outcar.readline().split()])
                epsilon.append([float(x) for x in outcar.readline().split()])
                epsilon.append([float(x) for x in outcar.readline().split()])

            if "BORN EFFECTIVE CHARGES" in line:
                outcar.readline()
                line = outcar.readline()
                if "ion" in line:
                    for i in range(num_atom):
                        born = []
                        born.append([float(x) for x in outcar.readline().split()][1:])
                        born.append([float(x) for x in outcar.readline().split()][1:])
                        born.append([float(x) for x in outcar.readline().split()][1:])
                        outcar.readline()
                        borns.append(born)
        borns = np.array(borns, dtype="double")
        epsilon = np.array(epsilon, dtype="double")

    return borns, epsilon

def _read_phonon_properties_from_OUTCAR(filename):#if I ever make this an object, nat will be absorbed into self
	with open(filename) as outcar:
		frequency = []
		eigenvector = []
		qpoint=[]

		PhononProperties={'Masses':[],'Eigenvectors':[],'Frequencies':[],'qpoint':[]}
		while True:
			line = outcar.readline()
			if not line:
				break

			if "NIONS" in line:
				num_atom = int(line.split()[11])

			if "Eigenvectors and eigenvalues of the dynamical matrix" in line:
                
				outcar.readline()
				print(outcar.readline())
				print(line)

	return PhononProperties

def get_phonon_grid_properties_from_mesh(meshdata):
	qyaml={'q-mesh':[],'q-point':[],'eigenstates':{'Frequencies':[],'Eigenvectors':{'atom':[],'Eigenvector':[]}}}#should be natoms frequencies and eigenvectors for each frequency
	qyaml['q-mesh']=np.array(meshdata['mesh'],dtype=int)
	for q in range(int(meshdata['nqpoint'])):
		#print(len(meshdata['phonon'][q]['band']))
		qyaml['q-point'].append(meshdata['phonon'][q]['q-position'])
		for b in range(len(meshdata['phonon'][q]['band'])):
			qyaml['eigenstates']['Frequencies'].append(float(meshdata['phonon'][q]['band'][b]['frequency'])*THz)#only ever one band for Si, may not be the case for all studied materials
			#print(meshdata['phonon'][q]['band'][b]['frequency'])
			for n in range(int(meshdata['natom'])):
				Ev=np.zeros(3,dtype=complex)
				qyaml['eigenstates']['Eigenvectors']['atom'].append(int(n+1))
				for direction in range(3):
					#print('q-point index:',q,'band index:',b,'atom:',n,'direction:',direction)
					Ev[direction]=float(meshdata['phonon'][q]['band'][b]['eigenvector'][n][direction][0])+1j*float(meshdata['phonon'][q]['band'][b]['eigenvector'][n][direction][1])
				qyaml['eigenstates']['Eigenvectors']['Eigenvector'].append(Ev)
	return qyaml

def read_phonon_properties_from_dyn2(filename,nat):
	with open(filename) as dyn:
		#also return q-point for fourier transform
		PhononProperties={'Masses':[],'Eigenvectors':[],'Frequencies':[],'qpoint':[]}
		masses=[16.00,16.00]
		eigenvectors=[]
		frequencies=[]
		qpoint=[]
		line = dyn.readlines()
	
		"""Frequency unit can be changed, but the read characters would need to be adjusted appropriately"""	
		#print('line length:',len(line))
		for s in range(len(line)):
			if 'q = ' in line[s]:
				if qpoint == []:
					qpoint.append([float(x) for x in line[s][14:54].split()])
					#qpoint.append(np.array(line[s][14:54]).tolist())
					#print((qpoint))
			if 'freq (' in line[s]:
				frequencies.append(float(line[s][25:34])*THz)
			if 'freq (' in line[s-1]:
				for m in range(0,nat-1):				
					eigenvectors.append(np.array(line[s+m][2:63]))
					eigenvectors.append(np.array(line[s+m+1][2:63]))
					#print(eigenvectors[m])
					#eigenvectors[m].remove(' ')
		PhononProperties['Masses']=masses
		PhononProperties['qpoint']=qpoint
		PhononProperties['Eigenvectors']=eigenvectors
		PhononProperties['Frequencies']=frequencies

	return PhononProperties

def _read_phonon_properties_from_OUTCAR(filename,NAT):#if I ever make this an object, nat will be absorbed into self
	with open(filename) as outcar:
		frequency = []
		eigenvector = []
		Eigenvectors=np.zeros((3*NAT,NAT,3))
		frequencymode=0
		frequencies=[]

		PhononProperties={'Masses':[],'Eigenvectors':[],'Frequencies':[]}
		while True:
			line = outcar.readline()
			#print(line[0:15])
			if not line:
				break

			if "NIONS" in line:
				num_atom = int(line.split()[11])
				#print('atoms',num_atom)
			if "f  =" in line or "f/i=" in line:
				
				frequency.append(line[12:21])
				frequencymode+=1
			if "dx" in line:
				born = []
				born.append([float(x) for x in outcar.readline().split()][3:])
				        			
				#print(line)
				#eigline=outcar.readline()
				#print(eigline)
				for I in range(num_atom-1):
					#eigline2=outcar.readline()
					#born = []
					born.append([float(x) for x in outcar.readline().split()][3:])#[1:])
				evec=np.array(born,dtype=float)	
				Eigenvectors[frequencymode-1,:,:]=evec
	
		frequency=np.array(frequency,dtype=float)
		#print(Eigenvectors[2,:,:])
		#print((frequency),frequencymode)
		PhononProperties['Eigenvectors']=Eigenvectors
		PhononProperties['Frequencies']=frequency
			
	return PhononProperties




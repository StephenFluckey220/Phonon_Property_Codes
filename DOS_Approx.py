import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve,inv
import MultiphononParameters as mp
from MultiphononParameters import *
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh
from phonopy.structure.tetrahedron_method import TetrahedronMethod
import cmath
import math
import scipy
import sys
import warnings
from scipy import integrate

"""for R1 in range(len(MPhaseUnitCell[0,:,0,0,0,0,0,0])):
		phase_factor=pd.get_phase_factor(np.array(q).astype(float),cellyaml['Fractional Coordinates'][atom],cellyaml['Fractional Coordinates'][int
				(dispyaml['Displacements']['Displaced Atom'][n-1])-1])
		MFourierPhaseTest+=args.dimX*args.dimY*args.dimZ*MUnitCell[q,R1,k1,0,k2,:,:,displacement_direction]*phase_factor"""

"""solve(ScaledTmat,	(Born[atom,:,:]-BornOrig[atom,:,:])	)*eV/(angstrom)"""




def BEFunc(omega,T,mu=0):
	nBE=1/(math.exp((hbar*2*cmath.pi*abs(omega))/(k*T))-1)
	if nBE < 0:
		print('BE function less than zero')
	return nBE

def BEFSum(omegas,T,n=2):
	if n==2:
		nBESum=((BEFunc(omegas[0],T)+1/2)+(BEFunc(omegas[1],T)+1/2))+((BEFunc(omegas[0],T)+1/2)-(BEFunc(omegas[1],T)+1/2))
	return nBESum

def diracApprox(alpha,vrange,method='Gaussian',plot=False):
	delta=np.zeros(len(vrange))
	if method == 'Gaussian':
		delta=np.exp(-(1/2)*vrange**2/alpha**2)/(np.sqrt(alpha**2*2*cmath.pi))
	if method == 'Lorentzian': 
		delta=1/(cmath.pi)*(alpha**2)/(alpha**2+vrange**2)
	if plot == True:
		plt.figure()
		plt.plot(vrange,delta)
		plt.show()
	return delta

def diracApprox_single_phonon(alpha,freq,vrange,method='Gaussian',plot=False):
	delta=np.zeros(len(vrange))
	if method == 'Gaussian':
		delta=np.exp(-(1/2)*((2*cmath.pi*vrange)**2-(2*cmath.pi*freq)**2)**2/(2*cmath.pi*alpha)**2)/(np.sqrt(2*cmath.pi*(2*cmath.pi*alpha)**2))
	if method == 'Lorentzian': 
		delta=1/(cmath.pi)*(alpha**2)/(alpha**2+(vrange**2-freq**2)**2)
	if plot == True:
		plt.figure()
		plt.plot(vrange,delta)
		plt.show()
	return delta

def get_delta_function_sum(omegas,freqRange,alpha,n=2):
	if n==2:
		deltaplus=diracApprox(alpha,omegas[0]+omegas[1]-freqRange)
		deltaminus=diracApprox(alpha,omegas[0]-omegas[1]-freqRange)
		delta_function_sum=deltaplus+deltaminus
		#print(delta_function_sum)
	
	return delta_function_sum

"""Calculates TDOS presumably per unit cell"""
def get_TDOS(meshdata,freqRange,alpha,BEFMod=False,T=298,plot=False):
	DeltaSum=np.zeros((len(freqRange)),dtype=float)
	#meshdata['phonon'][q]['band'][0]['frequency']
	DeltaSumMod=np.zeros((len(freqRange)),dtype=float)
	weights=[]
	for n in range(int(meshdata['nqpoint'])):
		weights.append(meshdata['phonon'][n]['weight'])
	weights=np.array(weights,dtype=int)
	print(np.sum(weights))

	#DOS_Tensor=np.zeros((int(meshdata['nqpoint']),6,6,len(freqRange)),dtype=float)

	"""Iterates over qpoint and mode combinations for each qpoint"""
	for qphon in range(int(meshdata['nqpoint'])):
		#print('1st q-point:',qphon)
			for iphon, phoni in enumerate(meshdata['phonon'][qphon]['band']):#Phonon['Frequencies']):
				#print('iphon:',iphon)
				for jphon, phonj in enumerate(meshdata['phonon'][qphon]['band']):#Phonon['Frequencies']):
					#print('jphon:',jphon)
					if BEFMod==True:
						deltaplus=np.array(diracApprox(alpha,float(phoni['frequency'])*THz+float(phonj['frequency'])*THz-freqRange))*((BEFunc(float(phoni['frequency'])*THz,T)+1/2)+(BEFunc(float(phonj['frequency'])*THz,T)+1/2))*weights[qphon]/np.sum(weights)

						deltaminus=np.array(diracApprox(alpha,float(phoni['frequency'])*THz-float(phonj['frequency'])*THz-freqRange))*((BEFunc(float(phoni['frequency'])*THz,T)+1/2)-(BEFunc(float(phonj['frequency'])*THz,T)+1/2))*weights[qphon]/np.sum(weights)
						
						DeltaSum+=deltaplus+deltaminus

						#DeltaSum+=get_delta_function_sum(np.array([float(phoni['frequency'])*THz,float(phonj['frequency'])*THz]),freqRange,alpha)*weights[qphon]/np.sum(weights)*BEFSum(						np.array([float(phoni['frequency'])*THz,float(phonj['frequency'])*THz]),T,n=2)
						DeltaSumMod+=get_delta_function_sum(np.array([float(phoni['frequency'])*THz,float(phonj['frequency'])*THz]),freqRange,alpha)*weights[qphon]/np.sum(weights)
					elif iphon >=0 and jphon >=0:
						DeltaSum+=get_delta_function_sum(np.array([float(phoni['frequency'])*THz,float(phonj['frequency'])*THz]),freqRange,alpha)*weights[qphon]/np.sum(weights)
						DeltaSumMod=DeltaSum
						#DOS_Tensor[qphon,iphon,jphon,:]=get_delta_function_sum(np.array([float(phoni['frequency'])*THz,float(phonj['frequency'])*THz]),freqRange,alpha)*weights[qphon]/np.sum(weights)

	magnitude=1#scipy.integrate.simps(DeltaSumMod,freqRange)	#normalization factor should only take DOS into account, not modification from Bose-Einstein functions		
	TDOS=1/magnitude*DeltaSum
	print('smearing magnitude',magnitude)

	
	'''plt.plot(freqRange/c*10**-2,DOS_Tensor[:,:3,:3,:].sum(axis=0).sum(axis=0).sum(axis=0),linestyle='--')
	plt.plot(freqRange/c*10**-2,DOS_Tensor[:,2:,2:,:].sum(axis=0).sum(axis=0).sum(axis=0),linestyle='--')
	plt.plot(freqRange/c*10**-2,2*DOS_Tensor[:,2:,:3,:].sum(axis=0).sum(axis=0).sum(axis=0),linestyle='--')
	plt.plot(freqRange/c*10**-2,DOS_Tensor[:,:,:,:].sum(axis=0).sum(axis=0).sum(axis=0))
	plt.legend(['A-A','O-A','O-O','total'])
	plt.show()'''

	if plot==True:
		print('TDOS magnitude:',scipy.integrate.simps(TDOS,freqRange))
		plt.figure()
		plt.xlabel('Frequency (THz)')
		plt.ylabel('TDOS ($cm^{-1}$/Unit Cell)')
		plt.plot(freqRange/THz,TDOS)
		plt.savefig('Si222_TDOS.svg',format='svg')
		plt.xlim(0,max(freqRange)/THz)
		plt.show()

	#np.save('Si3N4TDOSCmInverseCell.npy',TDOS)
	return TDOS

def get_Green_two_phonon(meshdata,freqRange,alpha,temperature):#calculates two phonon imaginary part of green function at specified temperature and points on a q-grid
	Green_two_phonon=np.zeros(len(int(meshdata['nqpoint'])),3*int(meshdata['natom']),3*int(meshdata['natom']),len(freqRange))
	weights=[]
	for n in range(int(meshdata['nqpoint'])):
		weights.append(meshdata['phonon'][n]['weight'])
	weights=np.array(weights,dtype=int)

	for qphon in range(int(meshdata['nqpoint'])):
		for iphon, phoni in enumerate(meshdata['phonon'][qphon]['band']):#Phonon['Frequencies']):
			for jphon, phonj in enumerate(meshdata['phonon'][qphon]['band']):
				if iphon!=jphon:	
					deltaplus=np.array(diracApprox(alpha,float(phoni['frequency'])*THz+float(phonj['frequency'])*THz-freqRange))*((BEFunc(float(phoni['frequency'])*THz,T)+1/2)+(BEFunc(float(phonj['frequency'])*THz,T)+1/2))*weights[qphon]/np.sum(weights)

					deltaminus=np.array(diracApprox(alpha,float(phoni['frequency'])*THz-float(phonj['frequency'])*THz-freqRange))*((BEFunc(float(phoni['frequency'])*THz,T)+1/2)-(BEFunc(float(phonj['frequency'])*THz,T)+1/2))*weights[qphon]/np.sum(weights)

					Green_two_phonon[qphon,iphon,jphon,:]=deltaplus+deltaminus
								

	return Green_two_phonon

def get_DOS(meshdata,freqRange,alpha,BEFMod=False,T=298,plot=False):
	DeltaSum=np.zeros((len(freqRange)),dtype=float)
	#meshdata['phonon'][q]['band'][0]['frequency']
	DeltaSumMod=np.zeros((len(freqRange)),dtype=float)
	weights=[]
	for n in range(int(meshdata['nqpoint'])):
		weights.append(meshdata['phonon'][n]['weight'])
	weights=np.array(weights,dtype=int)
	print(np.sum(weights))

	for qphon in range(int(meshdata['nqpoint'])):
		for iphon, phoni in enumerate(meshdata['phonon'][qphon]['band']):#Phonon['Frequencies']):
				if BEFMod==True:
					deltaplus=np.array(diracApprox(alpha,float(phoni['frequency'])*THz-freqRange))*((BEFunc(float(phoni['frequency'])*THz,T)+1/2))*weights[qphon]/np.sum(weights)
					deltaplusMod=np.array(diracApprox(alpha,float(phoni['frequency'])*THz-freqRange))*weights[qphon]/np.sum(weights)
					DeltaSum+=deltaplus
					DeltaSumMod+=deltaplusMod
				else:
					deltaplus=np.array(diracApprox(alpha,float(phoni['frequency'])*THz-freqRange))*weights[qphon]/np.sum(weights)
					DeltaSum+=deltaplus
					DeltaSumMod=DeltaSum
	magnitude=1	
	DOS=1/magnitude*DeltaSum
	print('smearing magnitude',magnitude)
	if plot==True:
		print('TDOS magnitude:',scipy.integrate.simps(DOS,freqRange))
		plt.figure()
		plt.xlabel('Frequency (THz)')
		plt.ylabel('DOS (THz/$nm^3$)')
		plt.plot(freqRange/THz,DOS*THz/(4))
		#plt.savefig('Si222_DOS.svg',format='svg')
		plt.xlim(5,max(freqRange)/THz)
		plt.show()

	return DOS


"""Input frequency assumes units of THz"""
def run_tetrahedron_method_dos(
    meshdata,
    frequency_points,
    grid_address,
    grid_mapping_table,
    relative_grid_address,
    coef=None,
    plot=False,
    NCell=1
):
    mesh=np.array(meshdata['mesh'], dtype="int_")
    
    frequencies=np.zeros((int(meshdata['nqpoint']),(3*int(meshdata['natom']))	))
    frequency_points=frequency_points/THz

    for qphon in range(int(meshdata['nqpoint'])):
          n=0
          for iphon, phoni in enumerate(meshdata['phonon'][qphon]['band']):
             frequencies[qphon,n]=(float(phoni['frequency']))
             n+=1
    """Return (P)DOS calculated by tetrahedron method in C."""
    try:
        import phonopy._phonopy as phonoc
    except ImportError:
        import sys

        print("Phonopy C-extension has to be built properly.")
        sys.exit(1)

    if coef is None:
        _coef = np.ones((frequencies.shape[0], 1, frequencies.shape[1]), dtype="double")
    else:
        _coef = np.array(coef, dtype="double", order="C")
    arr_shape = frequencies.shape + (len(frequency_points), _coef.shape[1])
    dos = np.zeros(arr_shape, dtype="double")

    #print('grid mapping table:',np.array(grid_mapping_table, dtype="int_", order="C"))
    #print('frequency points:',np.shape(frequency_points))
    

    phonoc.tetrahedron_method_dos(
        dos,
        np.array(mesh, dtype="int_"),
        frequency_points,
        frequencies,
        _coef,
        np.array(grid_address, dtype="int_", order="C"),
        np.array(grid_mapping_table, dtype="int_", order="C"),
        relative_grid_address,
    )
    if plot==True:
        plt.plot(dos[:, :, :, 0].sum(axis=0).sum(axis=0) /( np.prod(mesh)*NCell),color='k')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('DOS (THz/ Unit Cell)')
        #plt.xlim(min(frequency_points)/THz,max(frequency_points))
        plt.show()
    #print(np.shape(dos))
    if coef is None:
        return dos[:, :, :, 0].sum(axis=0).sum(axis=0) / np.prod(mesh)
    else:
        return dos.sum(axis=0).sum(axis=0) / np.prod(mesh)


def run_tetrahedron_method_dos_Gamma(
    meshdata,
    frequency_points,
    grid_address,
    grid_mapping_table,
    relative_grid_address,
    coef=None,
    plot=False
):
    mesh=np.array(meshdata['mesh'], dtype="int_")
    
    frequencies=np.zeros((1,(3*int(meshdata['natom']))	))
    frequency_points=frequency_points/THz

    
    n=0
    for iphon, phoni in enumerate(meshdata['phonon'][0]['band']):
       frequencies[0,n]=(float(phoni['frequency']))
       n+=1
    """Return (P)DOS calculated by tetrahedron method in C."""
    try:
        import phonopy._phonopy as phonoc
    except ImportError:
        import sys

        print("Phonopy C-extension has to be built properly.")
        sys.exit(1)

    if coef is None:
        _coef = np.ones((frequencies.shape[0], 1, frequencies.shape[1]), dtype="double")
    else:
        _coef = np.array(coef, dtype="double", order="C")
    arr_shape = frequencies.shape + (len(frequency_points), _coef.shape[1])
    dos = np.zeros(arr_shape, dtype="double")

    #print('grid mapping table:',np.array(grid_mapping_table, dtype="int_", order="C"))
    #print('frequency points:',np.shape(frequency_points))
    

    phonoc.tetrahedron_method_dos(
        dos,
        np.array(mesh, dtype="int_"),
        frequency_points,
        frequencies,
        _coef,
        np.array(grid_address, dtype="int_", order="C"),
        np.array(grid_mapping_table, dtype="int_", order="C"),
        relative_grid_address,
    )
    if plot==True:
        plt.plot(dos[:, :, :, 0].sum(axis=0).sum(axis=0) / np.prod(mesh))
        plt.xlabel('Frequency (THz)')
        plt.ylabel('TDOS (THz/ Unit Cell)')
        plt.show()
    #print(np.shape(dos))
    if coef is None:
        return dos[:, :, :, 0].sum(axis=0).sum(axis=0) / np.prod(mesh)
    else:
        return dos.sum(axis=0).sum(axis=0) / np.prod(mesh)

def run_tetrahedron_method_dos_BEF(
    mesh,
    frequency_points,
    frequencies,
    grid_address,
    grid_mapping_table,
    relative_grid_address,
    Temperature,
    coef=None,
):
    """Return (P)DOS calculated by tetrahedron method in C."""
    try:
        import phonopy._phonopy as phonoc
    except ImportError:
        import sys

        print("Phonopy C-extension has to be built properly.")
        sys.exit(1)

    if coef is None:
        _coef = np.ones((frequencies.shape[0], 1, frequencies.shape[1]), dtype="double")
    else:
        _coef = np.array(coef, dtype="double", order="C")
    arr_shape = frequencies.shape + (len(frequency_points), _coef.shape[1])
    dosBEF = np.zeros(arr_shape, dtype="double")

    #print('grid address:',np.shape(grid_address),np.array(grid_address, dtype="int_", order="C"))
    print('grid mapping table:',np.array(grid_mapping_table, dtype="int_", order="C"))
    print('frequency points:',np.shape(frequency_points))
    
    BEFMod=np.ones((frequencies.shape[0],frequencies.shape[1]))
    print('BEFMod shape:',np.shape(BEFMod))
    for nqpoint in range(len(frequencies[:,0])):
        for nmode in range(len(frequencies[0,:])):
            BEFMod[nqpoint,nmode]=BEFunc(frequencies[nqpoint,nmode]*THz,Temperature)         
  
    phonoc.tetrahedron_method_dos(
        dosBEF,
        np.array(mesh, dtype="int_"),
        frequency_points,
        frequencies,
        _coef,
        np.array(grid_address, dtype="int_", order="C"),
        np.array(grid_mapping_table, dtype="int_", order="C"),
        relative_grid_address,
    )
 
    if coef is None:
        for nqpoint in range(frequencies.shape[0]):
            for fmode in range(frequencies.shape[1]):
                for fpoint in range(len(frequency_points)):
                    dosBEF[nqpoint, fmode,fpoint, 0]=dosBEF[nqpoint, fmode,fpoint, 0]*BEFMod[nqpoint,fmode]
        #dosBEF=dosBEF[:, :,:, 0]*BEFMod
        return dosBEF
    else:
        print('not none:',np.shape(dosBEF))
        for nqpoint in range(frequencies.shape[0]):
            for fmode in range(frequencies.shape[1]):
                for fpoint in range(len(frequency_points)):
                    dosBEF[nqpoint, fmode,fpoint]=dosBEF[nqpoint, fmode,fpoint]*BEFMod[nqpoint,fmode]
        #dosBEF=dosBEF*BEFMod
        return dosBEF

def get_TDOS_Tetrahedron(meshdata,freqRange,grid_address, grid_mapping_table,relative_grid_address,NCell=1,BEFMod=False,Overtones=True,Temperature=298,plot=False,coefPlus=None,coefMinus=None,Supercell=False):
	DeltaSum=np.zeros((len(freqRange)),dtype=float)
	DeltaSumMod=np.zeros((len(freqRange)),dtype=float)
	weights=[]
	mesh=np.array(meshdata['mesh'], dtype="int_")
	for n in range(int(meshdata['nqpoint'])):
		weights.append(meshdata['phonon'][n]['weight'])
	weights=np.array(weights,dtype=int)
	freqsPlus=np.zeros((int(meshdata['nqpoint']),(3*int(meshdata['natom']))**2	))
	freqsMinus=np.zeros((np.shape(freqsPlus)))
	T=Temperature
	"""Assuming input is in Hz as base code uses this, but tetrahedron method requires units of THz"""
	frequency_points=freqRange/THz

	BEFModPlus=np.ones(freqsPlus.shape)
	BEFModMinus=np.ones(freqsMinus.shape)

	DOSSmearTest=np.zeros((len(freqRange)),dtype=float)
	deltaplus=np.zeros((int(meshdata['nqpoint']),(3*int(meshdata['natom']))**2,len(freqRange)	)	)
	deltaminus=np.zeros(np.shape(deltaplus))

	for qphon in range(int(meshdata['nqpoint'])):
		n=0
		for iphon, phoni in enumerate(meshdata['phonon'][qphon]['band']):
			for jphon, phonj in enumerate(meshdata['phonon'][qphon]['band']):
	
				if iphon>=0*NCell or jphon >=0*NCell:
					freqsPlus[qphon,n]=(float(phoni['frequency'])+float(phonj['frequency']))
					freqsMinus[qphon,n]=(float(phoni['frequency'])-float(phonj['frequency']))
					if BEFMod==True:
						BEFModPlus[qphon,n]=(BEFunc(float(phoni['frequency'])*THz,T)+1/2)+(BEFunc(float(phonj['frequency'])*THz,T)+1/2)
						BEFModMinus[qphon,n]=(BEFunc(float(phoni['frequency'])*THz,T)+1/2)-(BEFunc(float(phonj['frequency'])*THz,T)+1/2)
	

				n+=1

	"""Return (P)DOS calculated by tetrahedron method in C."""
	try:
		import phonopy._phonopy as phonoc
	except ImportError:
		import sys
		print("Phonopy C-extension has to be built properly.")
		sys.exit(1)

	"""Sum Contribution to TDOS with Bose Einstein contributions"""
	if coefPlus is None:
		_coefPlus = np.ones((freqsPlus.shape[0], 1, freqsPlus.shape[1]), dtype="double")
	else:
		_coefPlus = np.array(coefPlus, dtype="double", order="C")

	arr_shapePlus = freqsPlus.shape + (len(frequency_points), _coefPlus.shape[1])
	dosPlus = np.zeros(arr_shapePlus, dtype="double")

	phonoc.tetrahedron_method_dos(dosPlus,np.array(mesh, dtype="int_"),frequency_points,freqsPlus,_coefPlus,np.array(grid_address, dtype="int_", order="C"),np.array(grid_mapping_table, dtype="int_",order="C"),
        relative_grid_address,)

	dosPlusMod=dosPlus

	if coefPlus is None:
		for nqpoint in range(freqsPlus.shape[0]):
			for fmode in range(freqsPlus.shape[1]):
				for fpoint in range(len(frequency_points)):
					if fmode%7==0 and Overtones==False:
						dosPlus[nqpoint, fmode,fpoint, 0]=0
					else:
						dosPlus[nqpoint, fmode,fpoint, 0]=dosPlus[nqpoint, fmode,fpoint, 0]*BEFModPlus[nqpoint,fmode]
					
	else:
		print('not none:',np.shape(dosPlus))
		for nqpoint in range(freqsPlus.shape[0]):
			for fmode in range(freqsPlus.shape[1]):
				for fpoint in range(len(frequency_points)):
					if fmode%7==0 and Overtones==False:
						dosPlus[nqpoint, fmode,fpoint]=0
					else:
						dosPlus[nqpoint, fmode,fpoint]=dosPlus[nqpoint, fmode,fpoint]*BEFModPlus[nqpoint,fmode]
						

	"""Difference Contribution to TDOS with Bose Einstein functions"""
	if coefMinus is None:
		_coefMinus = np.ones((freqsMinus.shape[0], 1, freqsMinus.shape[1]), dtype="double")
	else:
		_coefMinus = np.array(coefMinus, dtype="double", order="C")	
	
	arr_shapeMinus = freqsMinus.shape + (len(frequency_points), _coefMinus.shape[1])
	dosMinus = np.zeros(arr_shapeMinus, dtype="double")
	
	phonoc.tetrahedron_method_dos(dosMinus,np.array(mesh, dtype="int_"),frequency_points,freqsMinus,_coefMinus,np.array(grid_address, dtype="int_", order="C"),np.array(grid_mapping_table, dtype="int_", order="C"),
        relative_grid_address,)

	dosMinusMod=dosMinus

	if coefMinus is None:
		print(np.shape(dosMinus))
		for nqpoint in range(freqsMinus.shape[0]):
			for fmode in range(freqsMinus.shape[1]):
				for fpoint in range(len(frequency_points)):
					if fmode%7==0 and Overtones==False:
						dosMinus[nqpoint, fmode,fpoint, 0]=0
					else:
						dosMinus[nqpoint, fmode,fpoint, 0]=dosMinus[nqpoint, fmode,fpoint, 0]*BEFModMinus[nqpoint,fmode]
	else:
		print('not none:',np.shape(dosMinus))
		for nqpoint in range(freqsMinus.shape[0]):
			for fmode in range(freqsMinus.shape[1]):
				for fpoint in range(len(frequency_points)):
					if fmode%7==0 and Overtones==False:
						dosMinus[nqpoint, fmode,fpoint]=0
					else:
						dosMinus[nqpoint, fmode,fpoint]=dosMinus[nqpoint, fmode,fpoint]*BEFModMinus[nqpoint,fmode]

	DeltaSum=dosPlus+dosMinus
	"""Normalizing Two phonon DOS"""
	DeltaSumMod=(dosPlusMod+dosMinusMod).sum(axis=0).sum(axis=0)
	if Supercell==True:
		TDOSPlot=1/(np.prod(mesh))*DeltaSum.sum(axis=0).sum(axis=0)/NCell**2
	else:
		TDOSPlot=1/(np.prod(mesh))*DeltaSum.sum(axis=0).sum(axis=0)
	TDOS=DeltaSum/(np.prod(mesh))#*THz converts from units of THz to units of Hz

	#np.save(str('DiamondTDOSPlusMinus')+str('.npy'),TDOSPlot*(100*c/THz)*10**-2*THz*100)

	if plot==True:
		print('TDOS magnitude:',scipy.integrate.simps(TDOSPlot[:,0],freqRange))
		plt.figure()
		plt.xlabel('Frequency ($cm^{-1}$)')
		plt.ylabel('TDOS ($cm^{-1}$/ $\mathrm{Unit~Cell}^2$)')
		plt.plot(freqRange*(100*c/THz)/THz*1000,TDOSPlot*(100*c/THz),color='k')
		plt.savefig('TDOS.pdf',format='pdf')
		plt.xlim(min(freqRange)*(100*c/THz)*1000/THz,max(freqRange)*(100*c/THz)*1000/THz)#max(freqRange)*(100*c/THz)*1000/THz)
		plt.ylim(0,None)
		plt.savefig(str('DiamondTDOSPlusMinus'),format='pdf',dpi=400,bbox_inches='tight',pad_inches=0.2)
		plt.show()
	return TDOS

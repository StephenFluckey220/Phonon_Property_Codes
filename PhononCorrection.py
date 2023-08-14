import yaml
import numpy as np
from yaml import load, dump, BaseLoader, SafeLoader
import argparse as ag
import random as r
from Phonon_Freq import *
import vasprun as vr
#from pprint import pprint
#from vasprun import vasprun
import matplotlib.pyplot as plt
import pymatgen as mg
import pymatgen.io.vasp as VASP


parser=ag.ArgumentParser()  
parser.add_argument('--material',default='/work/spf150030/Phonon Scripts/Si3N4',type=str)
parser.add_argument('--nac',nargs='?',default=False, const=True)
parser.add_argument('--vasp',nargs='?',default=False, const=True)
parser.add_argument('--save',nargs='?',default=False, const=True)
parser.add_argument('--TestDielectric',nargs='?',default=False, const=True)
parser.add_argument('--TestReflectance',nargs='?',default=False, const=True)
parser.add_argument('--TestTransmittance',nargs='?',default=False, const=True)
parser.add_argument('--TestEmissivity',nargs='?',default=False, const=True)
parser.add_argument('--plot',nargs='?',default=False, const=True)
parser.add_argument('--logPlot',nargs='?',default=False, const=True)
parser.add_argument('--gruneisen',nargs='?',default=False, const=True)
parser.add_argument('--dielectric',nargs='?',default=False, const=True)
parser.add_argument('--emissivity',nargs='?',default=False, const=True)
parser.add_argument('--setLinewidth',nargs='?',default=False, const=True)
parser.add_argument('--twopi',nargs='?',default=False, const=True)
parser.add_argument('--TOLinewidth',default=0.1,type=float)
parser.add_argument('--LOLinewidth',default=0.1,type=float)
parser.add_argument('--freqRangei',default=0,type=float)
parser.add_argument('--freqRangef',default=100,type=float)
parser.add_argument('--freqPoints',default=200,type=int)
parser.add_argument('--thickness',default=10,type=float)
args=parser.parse_args()

plt.grid(False)

MatName=str(args.material)+str('/qpoints.yaml')
MatName_lo=str((args.material)+str('/NAC')+str('/qpoints.yaml'))


#try:
MatIrrepName=str(args.material)+str('/irreps.yaml')
MatIrrepName_lo=str(args.material)+str('/NAC/irreps.yaml')
MatLinewidthName=str(args.material)+str('/linewidth.dat')
#except IOError

'''read q-points, irreducible representations, and linewidths from files'''
data = yaml.load(open(MatName),Loader=BaseLoader)
data_lo = yaml.load(open(MatName_lo),Loader=BaseLoader)
#linewidths=np.load(MatLinewidthName,unpack=True)
thickness=args.thickness*10**-6

irreps_data=yaml.load(open(MatIrrepName),Loader=BaseLoader)
if args.nac or args.dielectric or args.emissivity:
	irreps_data_lo=yaml.load(open(MatIrrepName_lo),Loader=BaseLoader)

rf=open(str(args.material)+str('/POSCAR'),'r')
vasp=str(args.material)+str('/vasprun.xml')

vr=VASP.Vasprun(vasp)
#print(vr.dielectric)
print(np.shape(data['phonon'][0]['dynamical_matrix']))#['q-position']['dynamical_matrix']))
'''
gd = h5py.File("gamma_detail-mxxx-gx.hdf5")
temp_index = 30 # index of temperature
temperature = gd['temperature'][temp_index]
gamma_tp = gd['gamma_detail'][:].sum(axis=-1).sum(axis=-1)
weight = gd['weight'][:]
gamma = np.dot(weight, gamma_tp[temp_index])
'''
freqRange=np.linspace(args.freqRangei,args.freqRangef,args.freqPoints)#2.8 micrometers: 107 THz, 3.5 micrometers: 85.65 THz, 8 micrometers :37.47 THz, 10 micrometers: 29.97 THz

Epsilon0=3.76# 7.5 #Si3N4 Static Dielectric Constant #Epsilon0=8.9 #GaN static dielectric constant
lambdRange=(3*10**8)/(15.633302*freqRange*10**12)

		
chone='Au'
chtwo='T1u'
IR_Active_characters=[chone,chtwo]
print(IR_Active_characters)
IR_Active_indices_Si3N4=[12,14,15,20,21,26,32,33,37,38]# Si3N4 IR-Active modes at zero temperature
IR_Active_indices_BeO=[5,9,10]
IR_Active_indices_diamond=['unknown']
IR_Active_indices_BN=[6,8,9]
IR_Active_indices_Si=[3,4,5]
IR_Active_indices_GaN=[6,7,8] #GaN IR-active modes at zero temperature
IR_Active_indices_SiO2=[3,4,8,12,13,14,15,16] #SiO2 IR-active modes at zero temperature
#TODamping=0.1*np.ones(len(IR_Active_indices))#Damping values and static dielectric constant set to test. 
#LODamping=0.1*np.ones(len(IR_Active_indices))

P=Phonon_Effects(data,irreps_data,True,vasp)
if args.nac or args.dielectric or args.emissivity:
	P_LO=Phonon_Effects(data_lo,irreps_data_lo,True,vasp)

dynmat=P.Dynamical_Matrix()
IR_Active_Modes=[]
freq_IR_Modes=[]

if args.nac or args.dielectric or args.emissivity:
	freq_IR_Modes_LO=[]

print('Irreps normal modes:',P.irrep_data['normal_modes'][1]['frequency'])

'''Find IR Active Modes from Character Table'''
for n in range(len(P.irrep_data_indices)):
	for i in range(len(IR_Active_characters)):
		if P.irrep_data['normal_modes'][n]['ir_label']==IR_Active_characters[i]:
			for m in range(len(P.irrep_data_indices[n])):		
				IR_Active_Modes.append(int(np.array(P.irrep_data_indices[n][m])))
				freq_IR_Modes.append(float(P.irrep_data['normal_modes'][n]['frequency']))
				if args.nac or args.dielectric or args.emissivity:
					freq_IR_Modes_LO.append(float(P_LO.irrep_data['normal_modes'][n]['frequency']))						

IR_Active_Modes=IR_Active_Modes[3:]	
IRAM=IR_Active_Modes
Irrep_TO_Modes=freq_IR_Modes[3:]
Irrep_LO_Modes=freq_IR_Modes_LO[3:]
print('Irrep Modes:',freq_IR_Modes)
print('Irrep LO Modes:',freq_IR_Modes_LO)


'''Get Dynamical Matrix and eigenfrequencies'''
TO_Modes=P.Frequencies(dynmat)[IR_Active_indices_GaN]# 3 lowest frequency modes are acoustic

print('TO Modes:',TO_Modes)

if args.nac or args.gruneisen or args.dielectric or args.emissivity:
	dynmat_LO=P_LO.Dynamical_Matrix()
	#print('LO Dyamical Matrix Dimensions:',np.shape(dynmat_LO))
	LO_Modes=P.Frequencies(dynmat_LO)[IR_Active_indices_GaN] #3 lowest modes are acoustic
	print('LO Modes:',LO_Modes)

CellVolume=P.CellRead(rf)
CellVolume_lo=CellVolume

TODamping=np.ones(len(TO_Modes))#Damping values and static dielectric constant set to test. 
LODamping=np.ones(len(LO_Modes))

if args.setLinewidth:
	TODamping=args.TOLinewidth*np.ones(len(TO_Modes))#Damping values and static dielectric constant set to test. 
	LODamping=args.LOLinewidth*np.ones(len(LO_Modes))
else:
	"""GaN 1 K Linewidths"""
	TODamping=2*[1.517845923958673e-02,1.265825677342176e-02,1.265825677342176e-02]
	LODamping=2*[1.517845923958673e-02,1.265825677342176e-02,1.174468894743755e-02]

	TOLineshift=[-1.004828289363014e-01,-1.011494156040974e-01,-1.011494156040974e-01]
	LOLineshift=[-1.004828289363014e-01,-1.011494156040974e-01,-9.337330344896723e-02]

	TOShiftedDamping=2*[1.532106688469257e-02,1.265825677342176e-02,1.265825677342176e-02]
	LOShiftedDamping=2*[1.532106688469257e-02,1.265825677342176e-02,1.747764929177660e-02]

	"""GaN 300 K Linewidths"""
	#TODamping=2*[6.429899991863343e-02,3.814086219647143e-02,3.814086219647143e-02]
	#LODamping=2*[6.429899991863343e-02,3.814086219647143e-02,3.181523482643671e-02]

	#TOLineshift=[-9.148774755427792e-02,-9.669369085760182e-02,-9.669369085760182e-02]
	#LOLineshift=[-9.148774755427792e-02,-9.669369085760182e-02,-8.240902543672211e-02]

	#TOShiftedDamping=2*[1.517845923958673e-02,3.814086219647143e-02,3.814086219647143e-02]
	#LOShiftedDamping=2*[1.517845923958673e-02,3.814086219647143e-02,4.465692130789726e-02]

	"""GaN 800 K Linewidths"""
	#TODamping=2*[1.786955775964182e-01,1.015789783152607e-01,1.015789783152607e-01]
	#LODamping=2*[1.786955775964182e-01,1.015789783152607e-01,8.305640486908146e-02]

	#TOLineshift=[-1.311481814783189e-01,-1.470417547887715e-01,-1.470417547887715e-01]
	#LOLineshift=[-1.311481814783189e-01,-1.470417547887715e-01,-1.121795555923655e-01]

	#TOShiftedDamping=2*[1.864533833724715e-01,1.015789783152607e-01,1.015789783152607e-01]
	#LOShiftedDamping=2*[1.864533833724715e-01,1.015789783152607e-01,1.154441902165703e-01]

if args.TestDielectric:
	print("Difference:",TO_Modes-LO_Modes)
#4 parameters needed for full emissvity description: dynamical matrices with and without non-analytical corrections, material thickness, mode linewidths



# 4.61/1.36*
	
if args.dielectric:
	
	TO_Broadening=TOShiftedDamping
	LO_Broadening=LOShiftedDamping
	print('Shifted TO Modes:',TO_Modes+TOLineshift)
	print('Shifted LO Modes:',LO_Modes+LOLineshift)
	
	eps,d_TO,d_LO=P.DielectricFun(TO_Broadening,LO_Broadening,TO_Modes+TOLineshift,LO_Modes+LOLineshift,freqRange,Epsilon0,True,True,args.TestDielectric)
	if args.twopi:
		freqRange=freqRange
	#print(d_TO)
	if args.plot:
		plt.figure(1)
		plt.grid(False)
		if args.logPlot:
			plt.semilogy(freqRange,eps.real)
			plt.semilogy(freqRange,np.absolute(eps.imag))
			plt.title('$\epsilon$')#  $\gamma_{TO}$=0.35, $\gamma_{LO}$=0.4')
			plt.ylabel('T')
			plt.xlabel('Wavelength ($\mu m$)')
		else:
			plt.plot(freqRange,eps.real)
			plt.plot(freqRange,np.absolute(eps.imag))
			plt.plot(freqRange,Epsilon0*np.ones(len(freqRange)))
			plt.xlim(args.freqRangei,args.freqRangef)
			plt.scatter(TO_Modes,np.zeros(len(TO_Modes)),color='r')
			#plt.scatter(LO_Modes,np.zeros(len(LO_Modes)),color='g')
			if args.twopi:
				plt.xlabel('Frequency($2\pi$THz)',fontsize=24)
			else:
				plt.xlabel('Frequency(THz)',fontsize=24)
		plt.title('Dielectric')#$\gamma_{TO}$=0.35, $\gamma_{LO}$=0.4')
		plt.ylabel('$\epsilon$',fontsize=24)
		
		plt.legend(['Real $\epsilon$','Imag $\epsilon$','$\epsilon_0$','$\Omega_{TO}$','$\Omega_{LO}$'])
		name=str('Dielectric Function.png')#+str(TO_Broadeningn)+str(LO_Broadeningn)+'.png')
		if args.save:
			plt.savefig(name,format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
		plt.show()
		 
	


if args.emissivity and not args.vasp:
	TO_Broadening=TODamping
	LO_Broadening=LODamping
	
	eps,d_TO,d_LO=P.DielectricFun(TO_Broadening,LO_Broadening,TO_Modes+TOLineshift,LO_Modes+LOLineshift,freqRange,Epsilon0,True,True,args.TestDielectric)
			
	
	R,n,k=P.Reflectance(eps,freqRange,args.TestReflectance)
	if args.plot:
		plt.figure(2)
		plt.grid(False)
		if args.logPlot:
			plt.semilogy(freqRange,R)
			plt.title('Reflectance')#  $\gamma_{TO}$=0.35, $\gamma_{LO}$=0.4')
			plt.ylabel('R')
			plt.xlabel('Wavelength ($\mu m$)')
		else:
			plt.plot(freqRange,R)
			plt.scatter(TO_Modes,np.zeros(len(TO_Modes)),color='r')
			#plt.scatter(LO_Modes,np.zeros(len(LO_Modes)),color='g')
			plt.title('Reflectance')# $\gamma_{TO}$=0.35, $\gamma_{LO}$=0.4')
			plt.ylabel('R')
			if args.twopi:
				plt.xlabel('Frequency($2\pi$THz)')
			else:
				plt.xlabel('Frequency(THz)')
			plt.legend(['Reflectance','$\Omega_{TO}$','$\Omega_{LO}$'])
			plt.xlim(args.freqRangei,args.freqRangef)
		name=str('Reflectance.png')#+str(TO_Broadeningn)+str(LO_Broadeningn)+'.png')
		if args.save:
			plt.savefig(name,format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
		plt.show()
	testeps=2*np.ones(len(freqRange))
	T,n,k=P.Transmittance(eps,freqRange,thickness,args.TestTransmittance)
	if args.plot:
		plt.xlim(args.freqRangei,args.freqRangef)
		plt.figure(3)
		plt.grid(False)
		if args.logPlot:
			plt.semilogy(freqRange,T)
			plt.title('Transmittance')#  $\gamma_{TO}$=0.35, $\gamma_{LO}$=0.4')
			plt.ylabel('T')
			plt.xlabel('Wavelength ($\mu m$)')
		else:
			plt.plot(freqRange,T)
			plt.scatter(TO_Modes,np.ones(len(TO_Modes)),color='r')
			#plt.scatter(LO_Modes,np.zeros(len(LO_Modes)),color='g')
			plt.title('Transmittance')#  $\gamma_{TO}$=0.35, $\gamma_{LO}$=0.4')
			plt.ylabel('T')
			plt.xlim(args.freqRangei,args.freqRangef)
			if args.twopi:
				plt.xlabel('Frequency($2\pi$THz)')
			else:
				plt.xlabel('Frequency(THz)')
			
		plt.legend(['Transmittance','$\Omega_{TO}$','$\Omega_{LO}$'])
		name=str('Transmittance.png')#+str(TO_Broadeningn)+str(LO_Broadeningn)+'.png')
		
		if args.save:
			plt.savefig(name,format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
			
		plt.show()
	

	E=P.emissivity(freqRange,R,T,args.TestEmissivity)
	print('Index of refraction and Extinction coefficient corresponding to minimum emissivity: n=',n[np.where(E==min(E))],'k=',k[np.where(E==min(E))])
	if args.plot:
		plt.figure(4)
		plt.grid(False)
		if args.logPlot:
			plt.semilogy(freqRange,E)
			plt.title('Emissivty')#  $\gamma_{TO}$=0.35, $\gamma_{LO}$=0.4')
			plt.ylabel('$\epsilon$')
			plt.xlabel('Frequency (THz)')
		else:
			plt.plot(freqRange,E)
			plt.scatter(TO_Modes,np.zeros(len(TO_Modes)),color='r')
			#plt.scatter(LO_Modes,np.zeros(len(LO_Modes)),color='g')
			plt.title('Emissivity')#  $\gamma_{TO}$=0.35, $\gamma_{LO}$=0.4')
			plt.ylabel('$\epsilon$')
			if args.twopi:
				plt.xlabel('Frequency($2\pi$THz)')
			else:
				plt.xlabel('Frequency(THz)')
			plt.legend(['$\epsilon$','$\Omega_{TO}$','$\Omega_{LO}$'])
			name=str('Emissivity.png')#+str(TO_Broadeningn)+str(LO_Broadeningn)+'.png')
			plt.xlim(0,30)
		if args.save:
			plt.savefig(name,format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
		plt.show()
if args.TestEmissivity:
	for l in range(len(freqRange)):
		print('R+T:',R[l]+T[l],'Frequency:',freqRange[l])
		print('R+T+E:',R[l]+T[l]+E[l],'Frequency:',freqRange[l])
	

if args.vasp:
	DielectricFun, EnergyRange=P.VaspDielectric(args.plot)
	if args.TestDielectric:
		plt.figure(1)
		plt.plot(EnergyRange, DielectricFun.real)
		plt.plot(EnergyRange, DielectricFun.imag)
		plt.show()

	if args.emissivity:
		R,n,k=P.Reflectance(DielectricFun,EnergyRange,args.TestReflectance)
		T,n,k=P.Transmittance(DielectricFun,EnergyRange,thickness,args.TestTransmittance)
		eps=E=P.emissivity(EnergyRange,R,T,args.TestEmissivity)

		if args.plot:
			
			plt.figure()
			plt.plot(np.array(EnergyRange)/6.28,R)
			plt.xlim(args.freqRangei,args.freqRangef)
			plt.grid('on')
			plt.legend(['R','$\Omega_{TO}$','$\Omega_{LO}$'])
			plt.xlabel('Frequency(THz)')
			plt.ylabel('R')
			plt.savefig('ReflectanceVasp.png',format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
			
			plt.figure(3)
			plt.plot(np.array(EnergyRange)/6.28,T)
			plt.xlim(args.freqRangei,args.freqRangef)
			plt.grid('on')
			plt.legend(['T','$\Omega_{TO}$','$\Omega_{LO}$'])
			plt.xlabel('Frequency(THz)')
			plt.ylabel('T')
			plt.savefig('TransmittanceVasp.png',format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)

			plt.figure(4)
			plt.plot(np.array(EnergyRange)/6.28,eps)
			plt.xlim(args.freqRangei,args.freqRangef)
			plt.grid('on')
			plt.legend(['$\epsilon$','$\Omega_{TO}$','$\Omega_{LO}$'])
			plt.xlabel('Frequency(THz)')
			plt.ylabel('$\epsilon$')
			plt.savefig('EmissivityVasp.png',format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)
			
			plt.figure()
			plt.plot(np.array(EnergyRange)/6.28,R)
			plt.plot(np.array(EnergyRange)/6.28,T)
			plt.plot(np.array(EnergyRange)/6.28,eps)
			#plt.plot(np.array(EnergyRange)/6.28,R+T+eps)
			plt.xlim(args.freqRangei,args.freqRangef)
			plt.grid('on')
			plt.legend(['R','T','$\epsilon$','R+T+$\epsilon$'])
			plt.xlabel('Frequency(THz)')
			plt.ylabel('R,T,$\epsilon$')
			plt.savefig('CombinedPlotVasp.png',format='png',dpi=400,bbox_inches='tight',pad_inches=0.2)


			
#volume=
#print(dynmat_data)
#print("width",len(dynmat_data[0][0,:]))
#print("height",len(dynmat_data[0][:,0]))


#if args.save:
#	np.save('Frequencies.npy',frequencies)


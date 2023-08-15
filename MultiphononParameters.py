#multiphonon parameters 
import cmath

pi=cmath.pi#unit predeclarations 
J=1
kg=1
s=1
meter=1#10**10
K=1

THz=10**12*1/meter
angstrom=10**-10
eV=1.602e-19*J
hbar=1.05*10**(-34)*J*s
hbar2 = 1#1.05 * 10 ** (-34)*J*s
h=6.63*10**(-34)*J*s
m0= 9.31*10**(-31)*kg
k = 1.38*10**-23*meter**2*kg*s**-1*K**-1
c=3.00*10**8*meter*s**-1
amu=1.66*10**-27*kg
epsilon0=8.87*10**-12*kg**-1*meter**-3*s**4

def get_RotationMatrix_x(theta):
	RotationMatrix_x=np.zeros((3,3))
	RotationMatrix_x=[[1,0,0],[0,cmath.cos(theta),-cmath.sin(theta)],[0,cmath.sin(theta),cmath.cos(theta)]]

	return RotationMatrix_x

def get_RotationMatrix_z(theta):
	RotationMatrix_z=np.zeros((3,3))
	RotationMatrix_z=[[cmath.cos(theta),0,cmath.sin(theta)],[0,1,0],[-cmath.sin(theta),0,cmath.cos(theta)]]

	return RotationMatrix_z

def get_RotationMatrix_y(theta):
	RotationMatrix_y=np.zeros((3,3))
	RotationMatrix_y=[[cmath.cos(theta),-cmath.sin(theta),0],[cmath.sin(theta),cmath.cos(theta),0],[0,0,1]]

	return RotationMatrix_y
	



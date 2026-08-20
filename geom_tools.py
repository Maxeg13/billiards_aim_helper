import numpy as np

def l1P(x):
	res = np.sqrt(x[0]*x[0] + x[1]*x[1])
	return res

def l2P(x):
	res = x[0]*x[0] + x[1]*x[1]
	return res

def createP(x):
	return np.array(x, dtype=np.float32)

def convertToDrawableP(p):
	return np.array(np.around(p), dtype=np.int16)

def rotateP(p, phi):
	mat = np.array([[np.cos(phi), np.sin(phi)], 
	[-np.sin(phi), np.cos(phi)]])
	return (mat@p.reshape((2, 1))).reshape(2)	
	
def vectorMult(x, y):
	return x[0] * y[1] - x[1] * y[0]

def middleP(p1, p2):
	return (p1 + p2)/ 2

def mixingP(p1, p2):
    mp = middleP(p1, p2)
    p1[:], p2[:] = mp[:], mp[:]

def signedDistP(p, line):
	p_ = p - line.p1
	return np.dot(p_, line.getNormalP())

def isAligned(p, line):
	pp = line.p2 - line.p1
	pp_dir = line.getDirP()
	pp_l1 = line.length()
	p_ = p - line.p1
	dot = np.dot(p_, pp_dir)
	aligned = dot > 0 and dot < pp_l1
	return aligned

def isOutsidedP(p, line):
	pp = line.p2 - line.p1
	pp_dir = line.getDirP()
	pp_l1 = line.length()
	p_ = p - line.p1
	dot = np.dot(p_, pp_dir)
	aligned = dot > 0 and dot < pp_l1
	return vectorMult(p_, pp) < 0 and aligned

def getAngleP(A, B):
	# print(f"get angle, A: {A}, B: {B}")
	C = A - B
	sign = 1 if vectorMult(A, B) > 0 else -1
	arg = (1/(2*l1P(A)*l1P(B))) * (l2P(A) + l2P(B) - l2P(C))
	arg = min(1., arg)
	res = np.arccos(arg) * sign
	# print(f"arg: {arg}, res: {res}")
	return res

class Line:
	def __init__(self, p1, p2):
		self.normal = createP([0., 0.])
		self.p1, self.p2  = p1, p2
	def __intersect__(self, line):
		l1, l2p1, l2p2 = self.p2 - self.p1, line.p1 - self.p1, line.p2 - self.p1
		return (vectorMult(l1, l2p1) * vectorMult(l1, l2p2)) < 0
	def length(self):
		p = self.p2 - self.p1
		return np.sqrt(p.dot(p))
	def intersect(self, line):
		return self.__intersect__(line) and line.__intersect__(self)
	def added(self, p):
		self.p1 += p
		self.p2 += p
		return self
	def addP(self, p):
		return Line(self.p1 + p, self.p2 + p)
	def interP(self, line):
		self.pp = self.p2 - self.p1
		pp = line.p2 - line.p1
		mat = np.array([[self.pp[1], -self.pp[0]],
									[pp[1], -pp[0]]])
		c = np.array([[self.p1[0] * self.pp[1] - self.p1[1] * self.pp[0]],
								[line.p1[0] * pp[1] - line.p1[1] * pp[0]]])
		return (np.linalg.inv(mat) @ c).reshape(2)
	def getP0(self):
		return (self.p2 + self.p1) * 0.5
	def rotated(self, dphi):
		p0 = self.getP0()
		p1, p2 = self.p1 - p0, self.p2 - p0
		self.p1, self.p2 = rotateP(p1, dphi) + p0, rotateP(p2, dphi) + p0
		return self
	def rotate(self):
		p0 = self.getP0()
		p1, p2 = self.p1 - p0, self.p2 - p0
		return Line(rotateP(p1, dphi) + p0, rotateP(p2, dphi) + p0)
	def getDirP(self):
		# print("lenght:", l1P(self.p2 - self.p1))
		return (self.p2 - self.p1)/l1P(self.p2 - self.p1)
	def getNormalP(self):
		self.normal = rotateP(self.getDirP(), np.pi/2)
		return self.normal 

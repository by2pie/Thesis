__author__ = 'jankrzywda'


from qutip.states import state_number_enumerate

__author__ = 'jankrzywda'


from qutip import *
numpy.set_printoptions(threshold='nan')
import numpy as np
import matplotlib.pyplot as plt
from cmath import *
import math as mt
from qft import qft
from fractions import gcd
from fractions import Fraction


def U_gate(x, a, N):
    return x**a % N

def contfrac(p, q):
    while q:
        n = p // q
        yield n
        q, p = p - q*n, q

def list2fract(list):
    result = 0
    print(reversed(list[1:]))
    for num in reversed(list[1:]):
        result = float(1/(float(num)+result))
        print(result)
        print('sss'+str(Fraction(result)))
    return Fraction(result)
H = snot(1)
N = 21
eps = 0.25
#L = int(mt.ceil(mt.log(N, 2)))
#t = int(2*L+1+mt.ceil(mt.log(2+1/(2*eps), 2)))
#x = np.random.randint(1, N-1)
L = 5  #drugi
t = 2   #pierwszy
x = 11
psi0 = 3
first_register = tensor([H*basis(2, 0)] * t)            #pierwszy rejestr
                                 #odwrotna transformata fouriera
second_register = tensor([basis(2, 1)] + [basis(2, 0)] * (L-1))        #drugi rejestr
circ = tensor(first_register, second_register)
qbit_state = []
for i, state in enumerate(circ):
    if state > 0:
        qbit_state.append([round(float(state.real), 2), state_index_number([2] * (L+t), i)[:t], state_index_number([2] * (L+t), i)[t:]])


prb_tr = []
temp = 0
#perform U gate:
for state in qbit_state:
    state[2] = state_index_number([2] * L, U_gate(x, state_number_index([2] * t, state[1]), N))
    prb_tr.append(temp + state[0]**2)
    temp += state[0]**2


#masurement:
dice = np.random.random()
measured = []
for i,tresh in enumerate(prb_tr):
    if dice < tresh:
        for qbit in qbit_state:
            if(qbit[2] == qbit_state[i][2]):
                measured.append(qbit[1])
        break
print(qbit_state)
n = len(measured)
for i in range(n):
    measured[i] = [1/n**0.5, measured[i]]
print(measured)
#fourier transform
qf_inv = qft(N=2)

fourierd_qubits = []
for qubits in measured:
    print(qubits)
    fourierd_qubits.append(qubits[0]*qf_inv * state_number_qobj([2] * t, qubits[1]))
print(fourierd_qubits)
#measuring c:
dice = np.random.random()
temp = 0
print(sum(fourierd_qubits))
for i, state in enumerate(sum(fourierd_qubits)):
    prob = (abs(state)**2).real
    temp += prob
    if dice < temp:
        c = i
        break


result = list(contfrac(c, N))
print(result)
print('res:')
print(list2fract(list(contfrac(c, N))))
y = x**(result[-1]/2) % N
print(y)
p = gcd(y + 1, N)
q = gcd(y - 1, N)
print(p, q)



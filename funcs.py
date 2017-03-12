from itertools import product
import math
import numpy as np

#генерация импульса
def Impuls(name,t,args):
    if name == "Berlage":
        A = args[0]
        n = args[1]
        B = args[2]
        f = args[3]
        ImpY = np.array([A * (i ** n) * math.exp(-B * i) * math.cos(2 * math.pi * f * i + math.pi / 2) for i in t])
    if name == "Gauss":
        A = args[0]
        B = args[1]
        f = args[2]
        ImpY = np.array([A * math.exp(-B * i ** 2) * math.sin(2 * math.pi * f * i) for i in t])
    return ImpY
#генерация словаря по параметрам
def DictGenPar(name,siglen,fs,args):

    n = len(args)

    Dict = []
    DictParam = []
    ArgsSize = []

    inds = []
    for comb in product(*[args[i] for i in range(n)]):
        inds.append(comb)

    if name == "Berlage":
        for i in range(4):
            DictParam.append([ inds[j][i] for j in range( len(inds) )] )
        DictParam.append([1]*len(inds))


        Pars = [0]*4;

        Pars[0] = [1]*len(inds)

        Tend = [j / 100 * siglen for j in DictParam[1]]
        Tmax = [DictParam[2][j] / 100 * Tend[j] for j in range(len(DictParam[3]))]
        Pars[1] = [math.log(0.05) / (math.log( Tend[j] / Tmax[j] ) - Tend[j] / Tmax[j] + 1 ) * DictParam[3][j] for j in range(len(Tend))];

        Pars[2] = [Pars[1][j] / (Tmax[j] / fs) for j in range(len(Tmax))]
        Pars[3] = DictParam[0]
    if name == "Gauss":
        for i in range(2):
            DictParam.append([inds[j][i] for j in range(len(inds))])
        DictParam.append([0]*len(inds))
        DictParam.append([inds[j][2] for j in range(len(inds))]);
        DictParam.append([2] * len(inds))

        Pars = [0] * 3;
        Pars[0] = [1]*len(inds);
        Pars[1] = [- math.log(0.05) / ( (math.ceil((round(DictParam[1][j] / 100 * siglen)-1) / 2) / fs) ** 2) *DictParam[3][j] for j in range(len(DictParam[3]))]
        Pars[2] = DictParam[0]

    for i in range(len(inds)):
        G = [0]*siglen
        L = round(DictParam[1][i] / 100 * siglen)

        if name == "Berlage":
            nums = list(range(1, L+1,1))
        else:
            nums = list( range( - math.floor( (L-1)/2 ), math.ceil( (L-1)/2 ) + 1, 1) )

        t = [i / fs for i in nums]
        ImpPars = [ j[i] for j in Pars]
        G[:L:] = Impuls(name,t,ImpPars)

        maxG = max([abs(j) for j in G])
        G = [ k / maxG for k in G]
        sumG2 = 0
        for k in G:
            sumG2+=k**2
        G = [k / (sumG2 ** 0.5) for k in G]

        Dict.append(G)

    rez = []
    rez.append(np.array(Dict,dtype=np.float64))
    rez.append(np.array(DictParam,dtype=np.float64))

    return rez

#согласованное преследование - классическое
def newMP(sig,Dict,iters):
    L = len( sig )
    ( N, M ) = np.shape( Dict )
    ERR = np.zeros(iters)

    sig_work = sig.copy()

    C = np.zeros( (N , L+M-1) )
    R = np.zeros( L+2*(M-1) )

    scales = 1 / (np.sum( Dict ** 2, 1 ) ** 0.5)

    for i in range(N):
        Dict[i,:] = Dict[i,:]*scales[i]

    SEG = [0]*iters

    for z in range(iters):
        for j in range(N):
            a = np.fft.fft(Dict[j,::-1],L+M-1)
            b = np.fft.fft(sig_work,L+M-1)
            C[j,:]=[k.real for k in np.fft.ifft(a*b)]

        num = np.argmax(abs(C))
        (x,y) = np.unravel_index( num, np.shape( C ))
        pos = y - M + 1

        R[y : y + M ] = R[y : y + M ] + C[x, y] * Dict[x, :]
        sig_work = sig - R[M - 1 : M + L-1]
        ERR[z] = 100 / np.sum( sig**2 ) * np.sum( (sig - R[M-1 : M + L-1]) ** 2)

    return ERR

#MP с уточнением

D = np.array( [ [1,6,2],
                [4,0,0]
                        ], dtype = np.float64)

s = np.array( [1,6,2,4], dtype = np.float64 )

np.set_printoptions(2)
print(newMP(s,D,2))
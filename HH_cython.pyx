from math import exp
#cimport numpy as np
import numpy as np
#DTYPE = np.int
#ctypedef np.int_t DTYPE_t
#from cpython cimport array
#import array


def hhModel(params, double[:] Iext, double dt, int[:] fI):

    ## Unwrap params argument: these variables are going to be optimized
    cdef double VT     = params[0]
    #cdef double d      = params[1] # [um]
    #cdef double area   = params[2] # [um^2]
    cdef double area   = 10522 # [um^2]

    # membrane capacitance
    #cdef double Cm     = params[3] * area * 1e-8 # [uF]
    cdef double Cm     = area * 1e-8 # [uF] under the assumption that the cell's capacitance density = 1 uF/cm2

    # equilibrium potentials
    cdef double ENa    = params[4] # [mV]
    cdef double EK     = params[5] # [mV]
    cdef double EL     = params[6] # [mV]
    cdef double ECa    = params[7] # [mV]

    # maximal conductances
    cdef double gNa    = params[8]  * area * 1e-8 # [mS]
    cdef double gK     = params[9]  * area * 1e-8 # [mS]
    cdef double gL     = params[10] * area * 1e-8 # [mS]
    cdef double gCa    = params[11] * area * 1e-8 # [mS]
    cdef double gM     = params[12] * area * 1e-8 # [mS]

    # Na+ activation
    cdef double M1     = params[13]
    cdef double M2     = params[14]
    cdef double M3     = params[15]
    cdef double M4     = params[16]
    cdef double M5     = params[17]
    cdef double M6     = params[18]

    # Na+ deactivation
    cdef double H1     = params[19]
    cdef double H2     = params[20]
    cdef double H3     = params[21]
    cdef double H4     = params[22]
    cdef double H5     = params[23]
    cdef double H6     = params[24]

    # K+ activation
    cdef double N1     = params[25]
    cdef double N2     = params[26]
    cdef double N3     = params[27]
    cdef double N4     = params[28]
    cdef double N5     = params[29]
    cdef double N6     = params[30]

    # slow K+ activation
    cdef double P1     = params[31]
    cdef double P2     = params[32]
    cdef double P3     = params[33]
    cdef double P4     = params[34]
    cdef double tauMax = params[35]

    ## Input paramters
    # I    : a list containing external current steps, your stimulus vector [nA]
    # dt   : a crazy time parameter [ms]
    # Vref : reference potential [mV]

    ############################### VT? Vref?
    def alphaM(double v):       return M1 * (v-VT+M2) / ( exp( (-v+VT-M2) / M3) -1.)
    def betaM(double v):        return M4 * (v-VT+M5) / ( exp( (v-VT+M5) / M6) -1.)
    def alphaH(double v):       return H1 * exp( (-v+VT+H2) / H3)
    def betaH(double v):        return H4 / (1. + exp( (-v+VT+H5) / H6))
    def alphaN(double v):       return N1 * (v-VT-N2) / (exp( (-v+VT+N2) / N3) -1.)
    def betaN(double v):        return N4 * exp( (-v+VT+N5) / N6)


    ## steady-state values and time constants of m,h,n
    def m_infty(double v):      return alphaM(v) / ( alphaM(v) + betaM(v) )
    def h_infty(double v):      return alphaH(v) / ( alphaH(v) + betaH(v) )
    def n_infty(double v):      return alphaN(v) / ( alphaN(v) + betaN(v) )
    def p_infty(double v):      return 1. / (1. + exp( (-v-P1) / P2))
    def p_tau(double v):        return tauMax / (P3 * exp( (v+P1) / P4) + exp( (-v-P1) / P4))

    # assert Iext.dtype == DTYPE
    # numSamples = int(T/dt);
    # DEF numSamples = len(Iext);
    # DEF numSamples = 120000
    cdef int numSamples = Iext.shape[0]
    cdef double[:] t    = np.linspace(0, numSamples*dt, numSamples, endpoint=True)
    cdef int j

    # initial values
    #cdef np.ndarray[np.float64_t] v = np.zeros(numSamples, dtype=np.float64)
    cdef double[:] v = np.zeros(numSamples)
    cdef double[:] m = np.zeros(numSamples)
    cdef double[:] h = np.zeros(numSamples)
    cdef double[:] n = np.zeros(numSamples)
    cdef double[:] p = np.zeros(numSamples)

    # Vref? VT?
    try:
        v[0]  = EL             # initial membrane potential
        m[0]  = m_infty(v[0])  # initial m
        h[0]  = h_infty(v[0])  # initial h
        n[0]  = n_infty(v[0])  # initial n
        p[0]  = p_infty(v[0])  # initial p

    ## calculate membrane response step-by-step
        for j in range(0, numSamples-1):

            # ionic currents: g[mS] * V[mV] = I[uA]
            INa = gNa * m[j]*m[j]*m[j]      * h[j] * (ENa-v[j])
            IK  = gK  * n[j]*n[j]*n[j]*n[j]        * (EK -v[j])
            IL  = gL                               * (EL -v[j])
            IM  = gM  * p[j]                       * (EK -v[j])

            # derivatives: I[uA] / C[uF] * dt[ms] = dv[mV]
            #dv_dt = ( INa + IK + IL + IM + Iext[j]*1e-3 ) / Cm
            dv_dt = ( fI[0]*INa + fI[1]*IK + fI[2]*IL + fI[3]*IM + Iext[j]*1e-3 ) / Cm

            dm_dt = (1-m[j])* alphaM(v[j]) - m[j]* betaM(v[j])
            dh_dt = (1-h[j])* alphaH(v[j]) - h[j]* betaH(v[j])
            dn_dt = (1-n[j])* alphaN(v[j]) - n[j]* betaN(v[j])
            dp_dt = (p_infty(v[j]) - p[j]) / p_tau(v[j])

            # calculate next step
            v[j+1] = (v[j] + dv_dt * dt)
            m[j+1] = (m[j] + dm_dt * dt)
            h[j+1] = (h[j] + dh_dt * dt)
            n[j+1] = (n[j] + dn_dt * dt)
            p[j+1] = (p[j] + dp_dt * dt)
    except (OverflowError, ZeroDivisionError):
        v[-1] = float("NaN")

    return np.asarray(v), np.asarray(t)
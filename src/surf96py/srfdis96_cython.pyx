from libc.stdlib cimport malloc, free
import numpy as np

cdef extern void disp_(int* iiso,int* iunit,int* iflsph,int* idimen,int* icnvel,
    int* mmax, float* d, float* a,float* b,float* rho,float* qa,float* qb,float* etap, float* etas,
    float* frefp,float* frefs,float* refdep,
    int* idispl, int* idispr, int* nsph,
    int* ifunc,int* kmax,int* mode,float* ddc,float* sone,int* igr,float* h,
    float* t,int* iq,int* is1,int* ie,
    float* rtp,float* dtp,float* btp,int* llw,
    int* iverb,
    int* wvtp,int* md,float* prd1,float* prd2,float* vel1,float* vel2)

cdef extern void __srfdis96_module_MOD_getmod_from_file(char* mname,
    char* title, int* iiso, int* iunit,int* iflsph,int* idimen,int* icnvel,
    int* mmax, float* d, float* a,float* b,float* rho,
    float* qa,float* qb,float* etap, float* etas,
    float* frefp,float* frefs,float* refdep)

def compute_fundamental_Rayleigh_phase(periods,
        thk, vp, vs, rho):
    cdef int iiso, iunit, iflsph, idimen, icnvel
    cdef int mmax
    cdef float *d
    cdef float *a
    cdef float *b
    cdef float *rho1
    cdef float *qa
    cdef float *qb
    cdef float *etap
    cdef float *etas
    cdef float *frefp
    cdef float *frefs
    cdef float refdep
    cdef int idispl, idispr, nsph
    cdef int ifunc, kmax, mode, igr, iq, is1, ie, llw
    cdef float ddc, sone, h
    cdef float *t
    cdef float *rtp
    cdef float *dtp
    cdef float *btp
    cdef int *iverb
    cdef int *wvtp
    cdef int *md
    cdef float *prd1
    cdef float *prd2
    cdef float *vel1
    cdef float *vel2

    #cdef char *title = <char*> malloc(81 * sizeof(char))
    #cdef char *mname
    cdef str title, mname

    cdef int NL=200, NP=512
    cdef int i

    iiso = 0
    iunit = 0
    iflsph = 0
    idimen = 1
    icnvel = 0
    igr = 0 # phase
    mode = 1 # mode(funda 1, higher >1)
    idispl = 0
    idispr = 1 # Rayleigh
    nsph = 0
    ifunc = 2 # Rayleigh

    # periods = 1.0/freqs
    kmax = len(periods)
    iq = 1 #
    is1 = 1
    ie = kmax

    iverb = <int*> malloc(2 * sizeof(int))
    iverb[0] = 0
    iverb[1] = 0

    t = <float*> malloc(NP * sizeof(float))
    for i in range(kmax):
        t[i] = periods[i]

    # print(periods)
    # print(len(periods))

    d = <float*> malloc(NL * sizeof(float))
    a = <float*> malloc(NL * sizeof(float))
    b = <float*> malloc(NL * sizeof(float))
    rho1 = <float*> malloc(NL * sizeof(float))
    qa = <float*> malloc(NL * sizeof(float))
    qb = <float*> malloc(NL * sizeof(float))
    etap = <float*> malloc(NL * sizeof(float))
    etas = <float*> malloc(NL * sizeof(float))
    frefp = <float*> malloc(NL * sizeof(float))
    frefs = <float*> malloc(NL * sizeof(float))
    rtp = <float*> malloc(NL * sizeof(float))
    dtp = <float*> malloc(NL * sizeof(float))
    btp = <float*> malloc(NL * sizeof(float))

    # mname = mdl
    # mdl_b = mdl.encode('utf-8')
    # print(mname)
    # title = 'MODxxxxxxxxxxxxxxxx'
    # title_b = title.encode('utf-8')

    # __srfdis96_module_MOD_getmod_from_file(mdl_b,
    #     title_b, &iiso, &iunit, &iflsph, &idimen, &icnvel,
    #     &mmax, d, a, b, rho,
    #     qa, qb, etap, etas,
        # frefp, frefs, &refdep)
    
    mmax = len(thk)
    
    for i in range(mmax):
        d[i] = thk[i]
        a[i] = vp[i]
        b[i] = vs[i]
        rho1[i] = rho[i]
        frefp[i] = 1.0
        frefs[i] = 1.0
        qa[i] = 0
        qa[i] = 0
        etap[i] = 0
        etas[i] = 0
    refdep = 0.0
    llw = 1

    ddc = 0.005
    sone = 0
    h = 0.005
    llw = 1
    
    wvtp = <int*> malloc(NP * sizeof(int))
    md = <int*> malloc(NP * sizeof(int))
    prd1 = <float*> malloc(NP * sizeof(float))
    prd2 = <float*> malloc(NP * sizeof(float))
    vel1 = <float*> malloc(NP * sizeof(float))
    vel2 = <float*> malloc(NP * sizeof(float))

    disp_(&iiso,&iunit,&iflsph,&idimen,&icnvel,
        &mmax, d, a, b, rho1, qa, qb, etap, etas,
        frefp, frefs, &refdep,
        &idispl, &idispr, &nsph,
        &ifunc, &kmax, &mode, &ddc, &sone, &igr, &h,
        t, &iq, &is1, &ie,
        rtp, dtp, btp, &llw,
        iverb,
        wvtp, md, prd1, prd2, vel1, vel2)

    # for i in range(kmax):
    #     print(wvtp[i], md[i], prd1[i],
    #         prd2[i], vel1[i], vel2[i])
    ret = tuple([np.array([prd1[i] for i in range(kmax)]),
        np.array([vel1[i] for i in range(kmax)])])

    free(d)
    free(a)
    free(b)
    free(rho1)
    free(qa)
    free(qb)
    free(etap)
    free(etas)
    free(frefp)
    free(frefs)
    free(rtp)
    free(dtp)
    free(btp)

    free(wvtp)
    free(md)
    free(prd1)
    free(prd2)
    free(vel1)
    free(vel2)
    # free(mname)

    return ret




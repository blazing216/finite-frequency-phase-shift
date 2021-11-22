c----------------------------------------------------------------------c
c                                                                    c
c    COMPUTER PROGRAMS IN SEISMOLOGY                                 c
c    VOLUME IV                                                       c
c                                                                    c
c    PROGRAM: SRFDIS                                                 c
c                                                                    c
c    COPYRIGHT 1986, 1991                                            c
c    D. R. Russell, R. B. Herrmann                                   c
c    Department of Earth and Atmospheric Sciences                    c
c    Saint Louis University                                          c
c    221 North Grand Boulevard                                       c
c    St. Louis, Missouri 63103                                       c
c    U. S. A.                                                        c
c                                                                    c
c----------------------------------------------------------------------c
c
      program srfdis96_0rp
c     0rp stands for fundamental Rayleigh wave phase velocity
c     1-D ISO flat model
c     simplify the program
c
        use srfdis96_module
c       parameter(LER=0,LIN=5,LOT=6)
c       integer NL, NL2, NLAY
c       parameter(NL=200,NLAY=200,NL2=NL+NL)
c       integer NP
c       parameter (NP=512)

c-----
c     LIN - unit for FORTRAN read from terminal
c     LOT - unit for FORTRAN write to terminal
c     LER - unit for FORTRAN error output to terminal
c     NL  - layers in model
c     NP  - number of unique periods
c-----
        double precision twopi,one,onea
        double precision cc,c1,clow,cm,dc,t1
        double precision c(NP),cb(NP)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),t(NP),btp(NL)
        real*4  qbinv(NL), qainv(NL)
        common/modl/ d,a,b,rho,rtp,dtp,btp
        common/para/ mmax,llw,twopi
        integer*4 iverb(2)

c-----
c     common for iget
c-----
        common/isomod/dl(NLAY),va(NLAY),vb(NLAY),rrho(NLAY),
     1      qa(NLAY),qb(NLAY),etap(NLAY),etas(NLAY), 
     2      frefp(NLAY), frefs(NLAY)
        common/depref/refdep
        integer mmax, iunit, iiso, iflsph, idimen, icnvel
        common/modtit/title
        character title*80
c-----
c     machine dependent initialization
c-----
        call mchdep()
c-----
        iverb(1) = 0
        iverb(2) = 0
        open(3,file='tmpsrfi.03',form='unformatted',access='sequential')
        rewind 3
        read(3) idispl,idispr,nsph
c-----
c     get velocity model
c-----
        call getmod(2,'tmpsrfi.17',mmax,title,iunit,iiso,iflsph,
     1      idimen,icnvel,ierr,.false.)
        mlyr = MMAX
        iunit = 0
        nsph = iflsph
c-----
c     save current values
c-----
        do 39 i=1,mmax
            if(qb(i).gt.0.0)then
                qbinv(i) = 1.0/qb(i)
            else
                qbinv(i) =     qb(i)
            endif
            if(qa(i).gt.0.0)then
                qainv(i) = 1.0/qa(i)
            else
                qainv(i) =     qa(i)
            endif
            b(i) = vb(i)
            a(i) = va(i)
            d(i) = dl(i)
            rho(i) = rrho(i)
   39   continue
c-----
c     check for water layer
c-----
        llw=1
        if(b(1).le.0.0) llw=2

        twopi=2.d0*3.141592653589793d0
        one=1.0d-2

        if(nsph.eq.1) call sphere(0,0)
c-----
c     find the extremal velocities to assist in starting search
c-----
        call get_maximum_and_minimum_velocities(betmx,betmn,
     1      jmn,jsol,mmax,a,b)
        write(LOT,*)'betmn, betmx:',betmn, betmx
c-----
c     open file for output dispersion?
c     1(tmpsrfi.06) for love, 2(tmpsrfi.07) for rayleigh
c-----        
        if(idispl.gt.0)then
            open(1,file='tmpsrfi.06',form='unformatted',
     1          access='sequential')
            rewind 1
        endif
        if(idispr.gt.0)then
            open(2,file='tmpsrfi.07',form='unformatted',
     1          access='sequential')
            rewind 2
        endif
c-----
c     Main loop (just two iteration)
c     1 for love, 2 for rayleigh
c     No further operation after the loop
c-----       
        do 2000 ifunc=1,2
c           skip the computation if idisp[l|r] set to 0        
            if(ifunc.eq.1.and.idispl.le.0) go to 2000
            if(ifunc.eq.2.and.idispr.le.0) go to 2000
c           flatten the model if spherical model used
            if(nsph.eq.1) call sphere(ifunc,1)

c           read in something            
            read(3) kmax,mode,ddc,sone,igr,h
            read(3) (t(i),i=1,kmax)
            write(LOT,*) 'kmax,mode,ddc,sone,igr,h'
            write(LOT,*) kmax,mode,ddc,sone,igr,h
            write(LOT,*) 't(1:kmax)'
            write(LOT,*) (t(i), i=1,kmax)

            call write_to_file_before_searching_velocity(ifunc,
     1          mmax,nsph,
     2          btp,dtp,
     3          d,a,b,rho,
     4          kmax,igr,h)

            if(sone.lt. 0.01) sone=2.0
            onea=dble(sone)

            call get_start_value_for_phase_velocity(betmn,jsol,
     1          jmn,a,b,ddc,cc,dc,c1,cm)
        
            call zero_cb_and_c(kmax, cb, c)

            ift=999
c-----
c           secondary main loop (over iq)
c           label 1800: end of the loop
c           label 1700: print no-found warning and end of loop
c           No further operation after the loop
c-----     
            do 1800 iq=1,mode
                read(3) is,ie
                itst=ifunc

c-----
c               third main loop (over k)
c               label 1600: end of the loop
c               No further operation after the loop
c-----                    
        do 1600 k=is,ie
            if(k.ge.ift) go to 1700
            
            t1=dble(t(k))
            if(igr.gt.0)then
                t1a=t1/(1.+h)
                t1b=t1/(1.-h)
                t1=dble(t1a)
            else
                t1a=sngl(t1)
            endif
c-----
c     get initial phase velocity estimate to begin search
c
c     in the notation here, c() is an array of phase velocities
c     c(k-1) is the velocity estimate of the present mode
c     at the k-1 period, while c(k) is the phase velocity of the
c     previous mode at the k period. Since there must be no mode
c     crossing, we make use of these values. The only complexity
c     is that the dispersion may be reversed. 
c
c     The subroutine getsol determines the zero crossing and refines
c     the root.
c-----
            call get_initial_phase_velocity_estimate(
     1          c1,clow,ifirst,
     2          k,is,iq,
     3          cc,c,dc,one,onea)
c-----
c     bracket root and refine it
c-----
            call getsol(t1,c1,clow,dc,cm,betmx,
     1          iret,ifunc,ifirst)
            if(iret.eq.-1)goto 1700
            c(k) = c1
c-----
c     for group velocities 
c     compute another solution for a perturbated frequency
c-----
            if(igr.gt.0) then
c               group velocity
                t1=dble(t1b)
                ifirst = 0
                clow = cb(k) + one*dc
                c1 = c1 -onea*dc
                call getsol(t1,c1,clow,dc,cm,betmx,iret,ifunc,ifirst)
c-----
c     test if root not found at slightly larger period
c-----
                if(iret.eq.-1)then
                    c1 = c(k)
                endif
                cb(k)=c1
            else
c               phase velocity
                c1 = 0.0d+00
            endif

            cc0 = sngl(c(k))
            cc1 = sngl(c1)
c important output
c itst-Love(0)orRayleigh(1)
c iq-mode, fundamental(1), higher modes(>1)
c t1a,t1b,cc0,cc1- period,0,phase velocity,0 (for phase)
c t1a,t1b,cc0,cc1- t-(slightly lower),t+,gv-,gv+ (for group)
c           write(LOT,*) itst,iq,t1a,t1b,cc0,cc1
            write(ifunc) itst,iq,t1a,t1b,cc0,cc1
 1600   continue
c       end of the third loop (over k) 
c       go to 1800 leads to the next loop in the second
c       loop (over iq)
        go to 1800

 1700   if(iq.gt.1) go to 1750

        if(iverb(ifunc).eq.0)then
            iverb(ifunc) = 1
            call print_warning_for_no_zero_found(ifunc,
     1          iq,is,ie,cc,cm,c1,
     2          mmax,d,a,b,rho,
     3          k,t,c)
        endif

 1750   ift=k
        itst=0

        call write_to_file_at_end_of_each_mode(ifunc,k,ie,
     1       itst,iq)

 1800   continue
        close(ifunc,status='keep')
 2000   continue
        close(3,status='keep')
        stop
        end


        subroutine gtsolh(a,b,c)
c-----
c     starting solution
c-----
        real*4 kappa, k2, gk2
        c = 0.95*b
        do 100 i=1,5
            gamma = b/a
            kappa = c/b
            k2 = kappa**2
            gk2 = (gamma*kappa)**2
            fac1 = sqrt(1.0 - gk2)
            fac2 = sqrt(1.0 - k2)
            fr = (2.0 - k2)**2 - 4.0*fac1*fac2
            frp = -4.0*(2.0-k2) *kappa
     1          +4.0*fac2*gamma*gamma*kappa/fac1
     2          +4.0*fac1*kappa/fac2
            frp = frp/b
            c = c - fr/frp
  100   continue
        return
        end

        subroutine getsol(t1,c1,clow,dc,cm,betmx,iret,ifunc,ifirst)
c-----
c     subroutine to bracket dispersion curve
c     and then refine it
c-----
c     t1  - period
c     c1  - initial guess on low side of mode
c     clow    - lowest possible value for present mode in a
c           reversed direction search
c     dc  - phase velocity search increment
c     cm  - minimum possible solution
c     betmx   - maximum shear velocity
c     iret    - 1 = successful
c         - -1= unsuccessful
c     ifunc   - 1 - Love
c         - 2 - Rayleigh
c     ifirst  - 1 this is first period for a particular mode
c         - 0 this is not the first period
c             (this is to define period equation sign
c              for mode jumping test)
c-----
        real*8 wvno, omega, twopi
        real*8 c1, c2, cn, cm, dc, t1, clow
        real*8 dltar, del1, del2, del1st, plmn
        save del1st
c-----
c     to avoid problems in mode jumping with reversed dispersion
c     we note what the polarity of period equation is for phase
c     velocities just beneath the zero crossing at the 
c         first period computed.
c-----
c     bracket solution
c-----
        twopi=2.d0*3.141592653589793d0
        omega=twopi/t1
        wvno=omega/c1
        del1 = dltar(wvno,omega,ifunc)
        if(ifirst.eq.1)del1st = del1
        plmn = dsign(1.0d+00,del1st)*dsign(1.0d+00,del1)
        if(ifirst.eq.1)then
            idir = +1
        elseif(ifirst.ne.1 .and. plmn.ge.0.0d+00)then
            idir = +1
        elseif(ifirst.ne.1 .and. plmn.lt.0.0d+00)then
            idir = -1
        endif
c-----
c     idir indicates the direction of the search for the
c     true phase velocity from the initial estimate.
c     Usually phase velocity increases with period and
c     we always underestimate, so phase velocity should increase
c     (idir = +1). For reversed dispersion, we should look
c     downward from the present estimate. However, we never
c     go below the floor of clow, when the direction is reversed
c-----
 1000   continue
        if(idir.gt.0)then
            c2 = c1 + dc
        else
            c2 = c1 - dc
        endif
        if(c2.le.clow)then
            idir = +1
            c1 = clow
        endif
        if(c2.le.clow)goto 1000
        omega=twopi/t1
        wvno=omega/c2
        del2 = dltar(wvno,omega,ifunc)
        if (dsign(1.0d+00,del1).ne.dsign(1.0d+00,del2)) then
            go to 1300
        endif
        c1=c2
        del1=del2
c   check that c1 is in region of solutions
        if(c1.lt.cm) go to 1700
        if(c1.ge.(betmx+dc)) go to 1700
        go to 1000
c-----
c     root bracketed, refine it
c-----
 1300   call nevill(t1,c1,c2,del1,del2,ifunc,cn)
        c1 = cn
        if(c1.gt.(betmx)) go to 1700
        iret = 1
        return
 1700   continue
        iret = -1
        return
        end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
        subroutine sphere(ifunc,iflag)
c-----
c     Transform spherical earth to flat earth
c
c     Schwab, F. A., and L. Knopoff (1972). Fast surface wave and free
c     mode computations, in  Methods in Computational Physics, 
c         Volume 11,
c     Seismology: Surface Waves and Earth Oscillations,  
c         B. A. Bolt (ed),
c     Academic Press, New York
c
c     Love Wave Equations  44, 45 , 41 pp 112-113
c     Rayleigh Wave Equations 102, 108, 109 pp 142, 144
c
c     Revised 28 DEC 2007 to use mid-point, assume linear variation in
c     slowness instead of using average velocity for the layer
c     Use the Biswas (1972:PAGEOPH 96, 61-74, 1972) density mapping
c
c     ifunc   I*4 1 - Love Wave
c                 2 - Rayleigh Wave
c     iflag   I*4 0 - Initialize
c                 1 - Make model  for Love or Rayleigh Wave
c-----
        parameter(NL=200,NP=512)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
        common/modl/ d,a,b,rho,rtp,dtp,btp
        common/para/ mmax,llw,twopi
        double precision z0,z1,r0,r1,dr,ar,tmp,twopi
        save dhalf
        ar=6370.0d0
        dr=0.0d0
        r0=ar
        d(mmax)=1.0
        if(iflag.eq.0) then
            do 5 i=1,mmax
                dtp(i)=d(i)
                rtp(i)=rho(i)
    5       continue
            do 10 i=1,mmax
                dr=dr+dble(d(i))
                r1=ar-dr
                z0=ar*dlog(ar/r0)
                z1=ar*dlog(ar/r1)
                d(i)=z1-z0
c-----
c               use layer midpoint
c-----
                TMP=(ar+ar)/(r0+r1)
                a(i)=a(i)*tmp
                b(i)=b(i)*tmp
                btp(i)=tmp
                r0=r1
   10       continue
            dhalf = d(mmax)
        else
            d(mmax) = dhalf
            do 30 i=1,mmax
                if(ifunc.eq.1)then
                     rho(i)=rtp(i)*btp(i)**(-5)
                else if(ifunc.eq.2)then
                     rho(i)=rtp(i)*btp(i)**(-2.275)
                endif
   30       continue
        endif
        d(mmax)=0.0
        return
        end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
        subroutine nevill(t,c1,c2,del1,del2,ifunc,cc)
c-----
c   hybrid method for refining root once it has been bracketted
c   between c1 and c2.  interval halving is used where other schemes
c   would be inefficient.  once suitable region is found neville s
c   iteration method is used to find root.
c   the procedure alternates between the interval halving and neville
c   techniques using whichever is most efficient
c-----
c     the control integer nev means the following:
c
c     nev = 0 force interval halving
c     nev = 1 permit neville iteration if conditions are proper
c     nev = 2 neville iteration is being used
c-----
        parameter (NL=200,NP=512)
        implicit double precision (a-h,o-z)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
        dimension x(20),y(20)
        common/modl/ d,a,b,rho,rtp,dtp,btp
        common/para/ mmax,llw,twopi
c-----
c     initial guess
c-----
        omega = twopi/t
        call half(c1,c2,c3,del3,omega,ifunc)
        nev = 1
        nctrl=1
  100 continue
        nctrl=nctrl+1
        if(nctrl.ge.100) go to 1000
c-----
c     make sure new estimate is inside the previous values. If not
c     perform interval halving
c-----
        if(c3 .lt. dmin1(c1,c2) .or. c3. gt.dmax1(c1,c2))then
            nev = 0
            call half(c1,c2,c3,del3,omega,ifunc)
        endif
            s13 = del1 - del3
            s32 = del3 - del2
c-----
c     define new bounds according to the sign of the period equation
c-----
            if(dsign(1.d+00,del3)*dsign(1.d+00,del1) .lt.0.0d+00)then 
                c2 = c3
                del2 = del3
            else
                c1 = c3
                del1 = del3
            endif
c-----
c     check for convergence. A relative error criteria is used
c-----
        if(dabs(c1-c2).le.1.d-6*c1) go to 1000
c-----
c     if the slopes are not the same between c1, c3 and c3
c     do not use neville iteration
c-----
        if(dsign (1.0d+00,s13).ne.dsign (1.0d+00,s32)) nev = 0
c-----
c     if the period equation differs by more than a factor of 10
c     use interval halving to avoid poor behavior of polynomial fit
c-----
        ss1=dabs(del1)
        s1=0.01*ss1
        ss2=dabs(del2)
        s2=0.01*ss2
        if(s1.gt.ss2.or.s2.gt.ss1 .or. nev.eq.0) then
            call half(c1,c2,c3,del3,omega,ifunc)
            nev = 1
            m = 1
        else
            if(nev.eq.2)then
                x(m+1) = c3
                y(m+1) = del3
            else
                x(1) = c1
                y(1) = del1
                x(2) = c2
                y(2) = del2
                m = 1
            endif
c-----
c     perform Neville iteration. Note instead of generating y(x)
c     we interchange the x and y of formula to solve for x(y) when
c     y = 0
c-----
            do 900 kk = 1,m
                j = m-kk+1
                denom = y(m+1) - y(j)
                if(dabs(denom).lt.1.0d-10*abs(y(m+1)))goto 950
                x(j)=(-y(j)*x(j+1)+y(m+1)*x(j))/denom
  900       continue
            c3 = x(1)
            wvno = omega/c3
            del3 = dltar(wvno,omega,ifunc)
            nev = 2
            m = m + 1
            if(m.gt.10)m = 10
            goto 951
  950       continue
            call half(c1,c2,c3,del3,omega,ifunc)
            nev = 1
            m = 1
  951       continue
        endif
        goto 100
 1000 continue
        cc = c3
        return
        end

        subroutine half(c1,c2,c3,del3,omega,ifunc)
        implicit double precision (a-h,o-z)
        c3 = 0.5*(c1 + c2)
        wvno=omega/c3
        del3 = dltar(wvno,omega,ifunc)
        return
        end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
        function dltar(wvno,omega,kk)
c   control the way to P-SV or SH.
c
        implicit double precision (a-h,o-z)
c
        if(kk.eq.1)then
c   love wave period equation
          dltar = dltar1(wvno,omega)
        elseif(kk.eq.2)then
c   rayleigh wave period equation
          dltar = dltar4(wvno,omega)
        endif
        end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
        function dltar1(wvno,omega)
c   find SH dispersion values.
c
        parameter (NL=200,NP=512)
        implicit double precision (a-h,o-z)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
        common/modl/ d,a,b,rho,rtp,dtp,btp
        common/para/ mmax,llw,twopi
c
c   Haskell-Thompson love wave formulation from halfspace
c   to surface.
c
        beta1=dble(b(mmax))
        rho1=dble(rho(mmax))
        xkb=omega/beta1
        wvnop=wvno+xkb
        wvnom=dabs(wvno-xkb)
        rb=dsqrt(wvnop*wvnom)
        e1=rho1*rb
        e2=1.d+00/(beta1*beta1)
        mmm1 = mmax - 1
        do 600 m=mmm1,llw,-1
          beta1=dble(b(m))
          rho1=dble(rho(m))
          xmu=rho1*beta1*beta1
          xkb=omega/beta1
          wvnop=wvno+xkb
          wvnom=dabs(wvno-xkb)
          rb=dsqrt(wvnop*wvnom)
          q = dble(d(m))*rb
          if(wvno.lt.xkb)then
                sinq = dsin(q)
                y = sinq/rb
                z = -rb*sinq
                cosq = dcos(q)
          elseif(wvno.eq.xkb)then
                cosq=1.0d+00
                y=dble(d(m))
                z=0.0d+00
          else
                fac = 0.0d+00
                if(q.lt.16)fac = dexp(-2.0d+0*q)
                cosq = ( 1.0d+00 + fac ) * 0.5d+00
                sinq = ( 1.0d+00 - fac ) * 0.5d+00
                y = sinq/rb
                z = rb*sinq
          endif
          e10=e1*cosq+e2*xmu*z
          e20=e1*y/xmu+e2*cosq
          xnor=dabs(e10)
          ynor=dabs(e20)
          if(ynor.gt.xnor) xnor=ynor
          if(xnor.lt.1.d-40) xnor=1.0d+00
          e1=e10/xnor
          e2=e20/xnor
  600 continue
        dltar1=e1
        return
        end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
        function dltar4(wvno,omga)
c   find P-SV dispersion values.
c
        parameter (NL=200,NP=512)
        implicit double precision (a-h,o-z)
        dimension e(5),ee(5),ca(5,5)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
        common/modl/ d,a,b,rho,rtp,dtp,btp
        common/para/ mmax,llw,twopi
        common/ovrflw/ a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
c
        omega=omga
        if(omega.lt.1.0d-4) omega=1.0d-4
        wvno2=wvno*wvno
        xka=omega/dble(a(mmax))
        xkb=omega/dble(b(mmax))
        wvnop=wvno+xka
        wvnom=dabs(wvno-xka)
        ra=dsqrt(wvnop*wvnom)
        wvnop=wvno+xkb
        wvnom=dabs(wvno-xkb)
        rb=dsqrt(wvnop*wvnom)
        t = dble(b(mmax))/omega
c-----
c   E matrix for the bottom half-space.
c-----
        gammk = 2.d+00*t*t
        gam = gammk*wvno2
        gamm1 = gam - 1.d+00
        rho1=dble(rho(mmax))
        e(1)=rho1*rho1*(gamm1*gamm1-gam*gammk*ra*rb)
        e(2)=-rho1*ra
        e(3)=rho1*(gamm1-gammk*ra*rb)
        e(4)=rho1*rb
        e(5)=wvno2-ra*rb
c-----
c   matrix multiplication from bottom layer upward
c-----
        mmm1 = mmax-1
        do 500 m = mmm1,llw,-1
          xka = omega/dble(a(m))
          xkb = omega/dble(b(m))
          t = dble(b(m))/omega
          gammk = 2.d+00*t*t
          gam = gammk*wvno2
          wvnop=wvno+xka
          wvnom=dabs(wvno-xka)
          ra=dsqrt(wvnop*wvnom)
          wvnop=wvno+xkb
          wvnom=dabs(wvno-xkb)
          rb=dsqrt(wvnop*wvnom)
          dpth=dble(d(m))
          rho1=dble(rho(m))
          p=ra*dpth
          q=rb*dpth
          beta=dble(b(m))
c-----
c   evaluate cosP, cosQ,.... in var.
c   evaluate Dunkin's matrix in dnka.
c-----
          call var(p,q,ra,rb,wvno,xka,xkb,dpth,w,cosp,exa)
          call dnka(ca,wvno2,gam,gammk,rho1)
          do 200 i=1,5
            cr=0.0d+00
            do 100 j=1,5
              cr=cr+e(j)*ca(j,i)
  100     continue
            ee(i)=cr
  200   continue
          call normc(ee,exa)
          do 300 i = 1,5
            e(i)=ee(i)
  300   continue
  500 continue
        if(llw.ne.1) then
c-----
c   include water layer.
c-----
          xka = omega/dble(a(1))
          wvnop=wvno+xka
          wvnom=dabs(wvno-xka)
          ra=dsqrt(wvnop*wvnom)
          dpth=dble(d(1))
          rho1=dble(rho(1))
          p = ra*dpth
          beta = dble(b(1))
          znul = 1.0d-05
          call var(p,znul,ra,znul,wvno,xka,znul,dpth,w,cosp,exa)
          w0=-rho1*w
        dltar4 = cosp*e(1) + w0*e(2)
        else
        dltar4 = e(1)
        endif
        return
        end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        subroutine var(p,q,ra,rb,wvno,xka,xkb,dpth,w,cosp,exa)
c-----
c   find variables cosP, cosQ, sinP, sinQ, etc.
c   as well as cross products required for compound matrix
c-----
c   To handle the hyperbolic functions correctly for large
c   arguments, we use an extended precision procedure,
c   keeping in mind that the maximum precision in double
c   precision is on the order of 16 decimal places.
c
c   So  cosp = 0.5 ( exp(+p) + exp(-p))
c            = exp(p) * 0.5 * ( 1.0 + exp(-2p) )
c   becomes
c       cosp = 0.5 * (1.0 + exp(-2p) ) with an exponent p
c   In performing matrix multiplication, we multiply the modified
c   cosp terms and add the exponents. At the last step
c   when it is necessary to obtain a true amplitude,
c   we then form exp(p). For normalized amplitudes at any depth,
c   we carry an exponent for the numerator and the denominator, and
c   scale the resulting ratio by exp(NUMexp - DENexp)
c
c   The propagator matrices have three basic terms
c
c   HSKA        cosp  cosq
c   DUNKIN      cosp*cosq     1.0
c
c   When the extended floating point is used, we use the
c   largest exponent for each, which is  the following:
c
c   Let pex = p exponent > 0 for evanescent waves = 0 otherwise
c   Let sex = s exponent > 0 for evanescent waves = 0 otherwise
c   Let exa = pex + sex
c
c   Then the modified matrix elements are as follow:
c
c   Haskell:  cosp -> 0.5 ( 1 + exp(-2p) ) exponent = pex
c             cosq -> 0.5 ( 1 + exp(-2q) ) * exp(q-p)
c                                          exponent = pex
c          (this is because we are normalizing all elements in the
c           Haskell matrix )
c    Compound:
c            cosp * cosq -> normalized cosp * cosq exponent = pex + qex
c             1.0  ->    exp(-exa)
c-----
        implicit double precision (a-h,o-z)
        common/ovrflw/   a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
        exa=0.0d+00
        a0=1.0d+00
c-----
c   examine P-wave eigenfunctions
c      checking whether c> vp c=vp or c < vp
c-----
        pex = 0.0d+00
        sex = 0.0d+00
        if(wvno.lt.xka)then
               sinp = dsin(p)
               w=sinp/ra
               x=-ra*sinp
               cosp=dcos(p)
        elseif(wvno.eq.xka)then
               cosp = 1.0d+00
               w = dpth
               x = 0.0d+00
        elseif(wvno.gt.xka)then
               pex = p
               fac = 0.0d+00
               if(p.lt.16)fac = dexp(-2.0d+00*p)
               cosp = ( 1.0d+00 + fac) * 0.5d+00
               sinp = ( 1.0d+00 - fac) * 0.5d+00
               w=sinp/ra
               x=ra*sinp
        endif
c-----
c   examine S-wave eigenfunctions
c      checking whether c > vs, c = vs, c < vs
c-----
        if(wvno.lt.xkb)then
               sinq=dsin(q)
               y=sinq/rb
               z=-rb*sinq
               cosq=dcos(q)
        elseif(wvno.eq.xkb)then
               cosq=1.0d+00
               y=dpth
               z=0.0d+00
        elseif(wvno.gt.xkb)then
               sex = q
               fac = 0.0d+00
               if(q.lt.16)fac = dexp(-2.0d+0*q)
               cosq = ( 1.0d+00 + fac ) * 0.5d+00
               sinq = ( 1.0d+00 - fac ) * 0.5d+00
               y = sinq/rb
               z = rb*sinq
        endif
c-----
c   form eigenfunction products for use with compound matrices
c-----
        exa = pex + sex
        a0=0.0d+00
        if(exa.lt.60.0d+00) a0=dexp(-exa)
        cpcq=cosp*cosq
        cpy=cosp*y
        cpz=cosp*z
        cqw=cosq*w
        cqx=cosq*x
        xy=x*y
        xz=x*z
        wy=w*y
        wz=w*z
        qmp = sex - pex
        fac = 0.0d+00
        if(qmp.gt.-40.0d+00)fac = dexp(qmp)
        cosq = cosq*fac
        y=fac*y
        z=fac*z
        return
        end
c
c
c
        subroutine normc(ee,ex)
c   This routine is an important step to control over- or
c   underflow.
c   The Haskell or Dunkin vectors are normalized before
c   the layer matrix stacking.
c   Note that some precision will be lost during normalization.
c
        implicit double precision (a-h,o-z)
        dimension ee(5)
        ex = 0.0d+00
        t1 = 0.0d+00
        do 10 i = 1,5
          if(dabs(ee(i)).gt.t1) t1 = dabs(ee(i))
   10 continue
        if(t1.lt.1.d-40) t1=1.d+00
        do 20 i =1,5
          t2=ee(i)
          t2=t2/t1
          ee(i)=t2
   20 continue
c-----
c   store the normalization factor in exponential form.
c-----
        ex=dlog(t1)
        return
        end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
        subroutine dnka(ca,wvno2,gam,gammk,rho)
c    Dunkin's matrix.
c
        implicit double precision (a-h,o-z)
        dimension ca(5,5)
        common/ ovrflw / a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
        data one,two/1.d+00,2.d+00/
        gamm1 = gam-one
        twgm1=gam+gamm1
        gmgmk=gam*gammk
        gmgm1=gam*gamm1
        gm1sq=gamm1*gamm1
        rho2=rho*rho
        a0pq=a0-cpcq
        ca(1,1)=cpcq-two*gmgm1*a0pq-gmgmk*xz-wvno2*gm1sq*wy
        ca(1,2)=(wvno2*cpy-cqx)/rho
        ca(1,3)=-(twgm1*a0pq+gammk*xz+wvno2*gamm1*wy)/rho
        ca(1,4)=(cpz-wvno2*cqw)/rho
        ca(1,5)=-(two*wvno2*a0pq+xz+wvno2*wvno2*wy)/rho2
        ca(2,1)=(gmgmk*cpz-gm1sq*cqw)*rho
        ca(2,2)=cpcq
        ca(2,3)=gammk*cpz-gamm1*cqw
        ca(2,4)=-wz
        ca(2,5)=ca(1,4)
        ca(4,1)=(gm1sq*cpy-gmgmk*cqx)*rho
        ca(4,2)=-xy
        ca(4,3)=gamm1*cpy-gammk*cqx
        ca(4,4)=ca(2,2)
        ca(4,5)=ca(1,2)
        ca(5,1)=-(two*gmgmk*gm1sq*a0pq+gmgmk*gmgmk*xz+
     *          gm1sq*gm1sq*wy)*rho2
        ca(5,2)=ca(4,1)
        ca(5,3)=-(gammk*gamm1*twgm1*a0pq+gam*gammk*gammk*xz+
     *          gamm1*gm1sq*wy)*rho
        ca(5,4)=ca(2,1)
        ca(5,5)=ca(1,1)
        t=-two*wvno2
        ca(3,1)=t*ca(5,3)
        ca(3,2)=t*ca(4,3)
        ca(3,3)=a0+two*(cpcq-ca(1,1))
        ca(3,4)=t*ca(2,3)
        ca(3,5)=t*ca(1,3)
        return
        end


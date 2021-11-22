      program srfdrl96
        use srfdrl96_module
        implicit none
        integer ip,ig
        character*10 fname(2)
        data fname/'tmpsrfi.06','tmpsrfi.05'/

        integer mmax,nsph,mmax2,kw,ll
        real d(NL),a(NL),b(NL),rho(NL)
        real xmu(NL)
        real btp(NL),dtp(NL)
c        real dphw(NL),dphw0
        integer nper,igr
        real h
        integer itst,mode
        real t,t1,c,cn
        real t0,ugr
        real dcdb(NL2),dudb(NL2)


        integer i

        ip=1
        ig=2
        open(1,file=fname(1),form='unformatted',access='sequential')
        open(2,file=fname(2),form='unformatted',access='sequential')
        rewind 1
        rewind 2

c-----
c       obtain the earth model:
c-----
        call read_model_from_tmpsrfi_06(1,
     1      mmax,nsph,mmax2,
     2      btp,dtp,d,a,b,rho,
     3      xmu)
        
        call check_for_water_layer(b,d,
     1      kw,ll)       
c-----
c       nper    number of frequencies (periods)
c       igr 0 phase velocity only
c           1 group velocity only
c           2 phase and group velocity data
c       h   increment for period to get partial with respect to period
c-----
        read(1) nper,igr,h
        if(igr.ge.2) then
            open(4,file='tmpsrfi.09',form='unformatted',
     1          access='sequential')
            rewind 4
        endif

  400   continue
        read(1,end=700) itst,mode,t,t1,c,cn
        if(itst.eq.0)go to 400

        call srfdrl(
     1      mmax,nsph,mmax2,
     2      btp,dtp,d,b,rho,
     3      xmu,
     1      ll,
     2      igr,h,
     3      itst,mode,t,t1,c,cn,
     4      t0,ugr,dcdb,dudb)
        
c-----
c       output the derivatives.
c       the first mmax elements are partial with respect to velocity
c       the next mmax are the partial with respect to moving the
c       boundary, e.g., takes two to change layer thickness
c-----
c       m=mode
        call write_Love_derivates_to_tmpsrfi_05_09(2,4,
     1      igr,itst,ip,ig,mode,mmax,mmax2,
     2      t0,c,ugr,dcdb,dudb)

        go to 400
  700   continue

c-----
c       end of data read and processing
c       do final clean up, outputting group velocity stuff
c       after all the phase velocity stuff
c-----
        if(igr.ge.2) then
            rewind 4
            call copy_group_velocity_from_tmpsrfi_09_to_05(2,4,
     1          mmax,mmax2)
            close(4,status='delete')
        endif
c-----
c       output an indicator that this is the end of the Love Wave data
c-----
        do 950 i=1,2
            close(i,status='keep')
  950   continue


      end program srfdrl96

      subroutine srfdrl(
     1      mmax,nsph,mmax2,
     2      btp,dtp,d,b,rho,
     3      xmu,
     1      ll,
     2      igr,h,
     3      itst,mode,t,t1,c,cn,
     4      t0,ugr,dcdb,dudb)
        use srfdrl96_module
c       NL2 - number of columns in model (first NL2/2 are
c           - velocity parameters, second NL2/2 are Q values)
c-----
        real d(NL),b(NL),rho(NL)
        real xmu(NL)
        integer mmax,ll
        real ut(NL),tt(NL),dcdb(NL2),dcdr(NL),uu0(4),
     *                 dudb(NL2),btp(NL),dtp(NL)
        real flagr,ale,ugr
        real uu(NL,2)
        real*8 exl(NL)

c       main loop, to line 196
c       label 700 also end of the 400 loop
c       read itst(RorL),mode(1,2..),prd1,prd2,v1,v2
c       itst=0 is invalid, continue to the next entry
c-----
c 400   continue

c-----
c     read in the dispersion values.
c-----

c       itst    0 - higher mode or wave type does not exist 
c               but make dummy
c               entry
c       mode    surface wave mode 1=FUND 2=1st if 0 mode 
c               does not exist here
c       t   period
c       t1  slightly different period for partial
c       c   phase velocity at t
c       cn  phase velocity at tn
c-----
c       read(1,end=700) itst,mode,t,t1,c,cn
c       if(itst.eq.0)go to 400

        t0=t
        if(itst.gt.0) then
            if(igr.gt.0) t0=t*(1.+h)
c-----
c           main part.
c-----

c           phase velocity (igr=0)
c           and its derivatives
c           or pertubted phase velocity (igr>0)
            twopi=2.*3.141592654
            om=twopi/t0
            omega=twopi/t
            wvno=omega/c

            call assemble_uu(uu,ut,tt)
            call shfunc(omega,wvno,
     *          exl,
     *          d,b,xmu,
     *          mmax,ll,
     *          uu,uu0)
            call energy(omega,wvno,
     *          d,b,rho,xmu,
     *          mmax,ll,
     *          uu,dcdb,dcdr,
     *          xi0,xi1,xi2,flagr,ale,ugr)
            call distribute_uu(uu,ut,tt)

            if(igr.gt.0)then
c           second pertubted phase velocity (igr>0)
                cp=c
                omp=omega
                ugp=ugr
                omega=twopi/t1
                omn=omega
                c=cn
                wvno=omega/c
c-----
c           save previous results
c-----
                do 420 i=1,mmax2
                    dudb(i)=dcdb(i)
  420           continue

                call assemble_uu(uu,ut,tt)
                call shfunc(omega,wvno,
     *              exl,
     *              d,b,xmu,
     *              mmax,ll,
     *              uu,uu0)
                call energy(omega,wvno,
     *              d,b,rho,xmu,
     *              mmax,ll,
     *              uu,dcdb,dcdr,
     *              xi0,xi1,xi2,flagr,ale,ugr)
                call distribute_uu(uu,ut,tt)

c               compute group velocity
c               and its derivatives
                ugr=(ugr+ugp)/2.
                c=(cp+cn)/2.
                do 430 i=1,mmax2
                    dcdn=dcdb(i)
                    dcdb(i)=(dudb(i)+dcdn)/2.
                    uc1=ugr/c
                    ta=uc1*(2.-uc1)*dcdb(i)
                    tb=uc1*uc1*((dudb(i)-dcdn)/(2.*h))
                    dudb(i)=ta+tb
  430           continue
            endif
c           end if for group velocity

c           derivatives for phase and group velocity
c           computed
c-----
c           sphericity correction
c-----
            call sphericity_correction(nsph,igr,
     1          om,c,ugr,mmax2,
     1          dcdb,
     1          dudb,btp,dtp)
        else
c           itst .eq. 0 (no measurement)
            mode = 0
            t0 = 0
        endif

c       go to 400

c 700   continue
        end

        subroutine splove(om,c,u,mmax2,iflag,
     1      dcdb,
     1      dudb,btp,dtp)
c-----
c       Transform spherical earth to flat earth
c       and relate the corresponding flat earth dispersion to spherical
c
c       Schwab, F. A., and L. Knopoff (1972). Fast surface wave and free
c       mode computations, in  Methods in Computational Physics, 
c               Volume 11,
c       Seismology: Surface Waves and Earth Oscillations,  
c               B. A. Bolt (ed),
c       Academic Press, New York
c
c       Love Wave Equations  49, 52  pp 114
c
c       Partial with respect to parameter uses the relation
c       For phase velocity, for example,
c
c       dcs/dps = dcs/dpf * dpf/dps, c is phase velocity, p is
c       parameter, and f is flat model
c
c       om  R*4 angular frequency
c       c   R*4 phase velocity
c       u   R*4 group velocity
c       mmax2   I*4 number of layers* 2, first mmax/2 values are
c               partial with respect to velocity, second are
c               partial with respect to layer thickness
c       iflag   I*4 0 - phase velocity
c               1 - group velocity
c-----
        use srfdrl96_module, only: NL,NL2
c       parameter(NL=200,NL2=NL+NL)
        real dcdb(NL2),
     1      dudb(NL2),btp(NL),dtp(NL)
c       common/eigfun/ ut(NL),tt(NL),dcdb(NL2),dcdr(NL),uu0(4),
c    1      dudb(NL2),btp(NL),dtp(NL)
        a=6370.0
        mmax = mmax2/2
        rval = a
        tm=sqrt(1.+(3.*c/(2.*a*om))**2)
        if(iflag.eq.1) then
            do 10 i=1,mmax
                tmp=dudb(i)*tm+u*c*dcdb(i)*(3./(2.*a*om))**2/tm
                dudb(i)=btp(i)*tmp
                tmp=dudb(i+mmax)*tm+u*c*dcdb(i+mmax)
     1              *(3./(2.*a*om))**2/tm
                dudb(i+mmax) = (a/rval)*tmp
                        rval = rval - dtp(i)
   10       continue
            u=u*tm
        else
            do 20 i=1,mmax
                dcdb(i)=dcdb(i)*btp(i)/(tm**3)
                dcdb(i+mmax)=dcdb(i+mmax)*(a/rval)/(tm**3)
                        rval = rval - dtp(i)
   20       continue
            c=c/tm
        endif
        end

      subroutine shfunc(omega,wvno,
     *    exl,
     *    d,b,xmu,
     *    mmax,ll,
     *    uu,uu0)
c-----
c     This routine evaluates the eigenfunctions by calling sub
c       up.
c-----
      parameter(NL=200,NL2=NL+NL)
      real*8 exl(NL),ext,fact
c     common/model/  d(NL),a(NL),b(NL),rho(NL),qa1(NL),
c    *                 qb1(NL),xmu(NL),xlam(NL),mmax,ll
      real d(NL),b(NL)
      real xmu(NL)
      integer mmax,ll
      real uu(NL,2),uu0(4)
c     common/eigfun/ uu(NL,2),dcdb(NL2),dcdr(NL),uu0(4),
c    *                 dudb(NL2),btp(NL),dtp(NL)
c     common/save/   exl
      call up(omega,wvno,fl,exl,
     *    d,b,xmu,
     *    mmax,ll,
     *    uu)
c-----
c       uu0(2)=stress0 is actually the value of period equation.
c       uu0(3) is used to print out the period euation value before
c       the root is refined.
c-----
        uu0(1)=1.0
        uu0(2)=fl
        uu0(3)=0.0
        uu0(4)=0.0
        ext=0.0
        do 100 k=ll+1,mmax
            ext=ext+exl(k-1)
            fact=0.0
            if(ext.lt.85.0) fact=1./dexp(ext)
            uu(k,1)=uu(k,1)*fact/uu(ll,1)
            uu(k,2)=uu(k,2)*fact/uu(ll,1)
  100   continue
        uu(ll,1)=1.0
        uu(ll,2)=0.0
        return
        end
c
c  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
      subroutine up(omega,wvno,fl,exl,
     *    d,b,xmu,
     *    mmax,ll,
     *    uu)
c     This routine calculates the elements of Haskell matrix,
c     and finds the eigenfunctions by analytic solution.
c
      parameter(LER=0,LIN=5,LOT=6)
      parameter(NL=200,NL2=NL+NL)
      real*8 exl(NL),qq,rr,ss,exqm,exqp,sinq,cosq
c     common/model/  d(NL),a(NL),b(NL),rho(NL),qa1(NL),
c    *                 qb1(NL),xmu(NL),xlam(NL),mmax,ll
      real d(NL),b(NL)
      real xmu(NL)
      integer mmax,ll
      real uu(NL,2)
c     common/eigfun/ uu(NL,2),dcdb(NL2),dcdr(NL),uu0(4),
c    *                 dudb(NL2),btp(NL),dtp(NL)
c     common/save/   exl
      wvno2=wvno*wvno
      xkb=omega/b(mmax)
c-----
c     kludge for fluid core
c-----
      if(b(mmax).gt.0.01)then
             rb=sqrt(abs(wvno2-xkb*xkb))
             if(wvno.lt.xkb)then
                   write(LOT,*) ' imaginary nub derivl'
                   write(LOT,*)'omega,wvno,b(mmax)',omega,wvno,b(mmax)
               endif
             uu(mmax,1)=1.0
             uu(mmax,2)=-xmu(mmax)*rb
      else
             uu(mmax,1)=1.0
             uu(mmax,2)=0.0
      endif
      mmx1=mmax-1
      do 500 k=mmx1,ll,-1
      k1=k+1
      dpth=d(k)
      xkb=omega/b(k)
      rb=abs(wvno2-xkb*xkb)
      rr=dble(rb)
      rr=dsqrt(rr)
      ss=dble(dpth)
      qq=rr*ss
C      if(wvno-xkb) 100,200,300
        if( wvno .lt. xkb)then
            go to 100
        else if(wvno .eq. xkb)then
            go to 200
        else
            go to 300
        endif
100   sinq=dsin(qq)
      cosq=dcos(qq)
      y=sinq/rr
      z=-rr*sinq
      qq=0.0
      go  to 400
200   qq=0.0
      cosq=1.0d+0
      y=dpth
      z=0.0
      go to 400
300   if(qq.gt.40.0) go to 350
      exqp=1.
      exqm=1./dexp(qq+qq)
      sinq=(exqp-exqm)*0.5
      cosq=(exqp+exqm)*0.5
      y=sinq/rr
      z=rr*sinq
      go to 400
350   continue
      y=0.5/rr
      z=0.5*rr
      cosq=0.5
400   continue
      amp0=cosq*uu(k1,1)-y*uu(k1,2)/xmu(k)
      str0=cosq*uu(k1,2)-z*xmu(k)*uu(k1,1)
      rr=abs(amp0)
      ss=abs(str0)
      if(ss.gt.rr) rr=ss
      if(rr.lt.1.d-30) rr=1.d+00
      exl(k)=dlog(rr)+qq
      uu(k,1)=amp0/rr
      uu(k,2)=str0/rr
500   continue
      fl=uu(ll,2)
      return
      end
c
c  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
      subroutine energy(omega,wvno,
     *    d,b,rho,xmu,
     *    mmax,ll,
     *    uu,dcdb,dcdr,
     *    xi0,xi1,xi2,flagr,ale,ugr)
c     This routine calculates the values of integrals I0, I1,
c     and I2 using analytic solutions. It is found
c     that such a formulation is more efficient and practical.
c
      parameter(NL=200,NL2=NL+NL)
      real*8 wvno0,omega0,c,sumi0,sumi1,sumi2
      real*8 xkb,rb,dbb,drho,dpth,dmu,wvno2,omega2
      real*8 upup,dupdup,dcb,dcr
      complex*16 nub,xnub,exqq,top,bot,f1,f2,f3,zarg
c     common/model/  d(NL),a(NL),b(NL),rho(NL),qa1(NL),
c    *                 qb1(NL),xmu(NL),xlam(NL),mmax,ll
      real d(NL),b(NL),rho(NL)
      real xmu(NL)
      integer mmax,ll
      real uu(NL,2),dcdb(NL2),dcdr(NL)
c     common/eigfun/ uu(NL,2),dcdb(NL2),dcdr(NL),uu0(4),
c    *                 dudb(NL2),btp(NL),dtp(NL)
      real xi0,xi1,xi2,flagr,ale,ugr
c     common/sumi/   xi0,xi1,xi2,flagr,ale,ugr
        real*8 c2, fac, dvdz, dfac
      wvno0=dble(wvno)
      omega0=dble(omega)
      c=omega0/wvno0
      omega2=omega0*omega0
      wvno2=wvno0*wvno0
      sumi0=0.0d+00
      sumi1=0.0d+00
      sumi2=0.0d+00
      do 300 k=ll,mmax
            k1=k+1
            dbb=dble(b(k))
            drho=dble(rho(k))
            dpth=dble(d(k))
            dmu=dble(xmu(k))
            xkb=omega0/dbb
            rb=dsqrt(dabs(wvno2-xkb*xkb))
            if(k.eq.mmax) then
                  upup  =(0.5/rb)*uu(mmax,1)*uu(mmax,1)
                  dupdup=0.5*rb*uu(mmax,1)*uu(mmax,1)
            else
                  if(wvno0.lt.xkb)then
                        nub=dcmplx(0.0d+00,rb)
                  else
                        nub=dcmplx(rb,0.0d+00)
                  endif
                  xnub=dmu*nub
                  top=uu(k,1)-uu(k,2)/xnub
                  bot=uu(k1,1)+uu(k1,2)/xnub
                  f3=nub*dpth
                  if(dreal(f3).lt.40.0d+00) then
                        zarg = -2.0d+00*f3
                        exqq=dexp(dreal(zarg))*
     1                     dcmplx(dcos(dimag(zarg)),dsin(dimag(zarg)))
c                       exqq=cdexp(-2.*f3)
                  else
                        exqq=dcmplx(0.0d+00,0.0d+00)
                  endif
                  f1=(1.-exqq)/(2.*nub)
                  if(dreal(f3).lt.80.0d+00)then
                        zarg = -f3
                        exqq=dexp(dreal(zarg))*
     1                     dcmplx(dcos(dimag(zarg)),dsin(dimag(zarg)))
c                       exqq=cdexp(-f3)
                  else
                        exqq=dcmplx(0.0d+00,0.0d+00)
                  endif
                  f1=0.25*f1*(top*top+bot*bot)
                  f2=0.5 *dpth*exqq*top*bot
                  upup=dreal(f1+f2)
                  f3=xnub*xnub*(f1-f2)
                  dupdup=dreal(f3)/(dmu*dmu)
            endif
            sumi0=sumi0+drho*upup
            sumi1=sumi1+dmu*upup
            sumi2=sumi2+dmu*dupdup
            dcr=-0.5*c*c*c*upup
            dcb=0.5*c*(upup+dupdup/wvno2)
            dcdb(k)=2.*drho*dbb*dcb
            dcdr(k)=dcr+dbb*dbb*dcb
  300   continue
c-----
c       now that the energy integral I1 is defined get final partial
c-----
        do 400 k=ll,mmax
            dcdb(k)=dcdb(k)/sumi1
            dcdr(k)=dcdr(k)/sumi1
  400   continue
            
c-----
c       get lagrangian, group velocity, energy integral
c-----
            flagr=omega2*sumi0-wvno2*sumi1-sumi2
            ugr=sumi1/(c*sumi0)
            ale=0.5/sumi1
            xi0=sumi0
            xi1=sumi1
            xi2=sumi2
c-----
c       define partial with respect to layer thickness
c-----
c       fac = 0.5d+00*c**3/(omega2*sumi1)
        fac = ale*c/wvno2
        c2 = c*c
c-----
c       for initial layer
c-----
        if(ll.ne.1)then
            dcdb(1) = 0.0
            dcdb(mmax+1) = 0.0
        endif
        do 500 k=ll,mmax
            if(k.eq.ll)then
                drho = rho(k)
                dmu  = xmu(k)
            else
                drho = rho(k) - rho(k-1)
                dmu  = xmu(k) - xmu(k-1)
            endif
            if(k.eq.ll)then
                dvdz = 0.0
            else
                dvdz = uu(k,2)*uu(k,2)*(1.0/xmu(k) - 1.0/xmu(k-1))
            endif
            dfac = fac * ( uu(k,1)*uu(k,1)*
     1          (omega2*drho - wvno2*dmu) + dvdz)
            if(dabs(dfac).lt.1.0d-38)then
                dcdb(k+mmax) = 0.0
            else
                dcdb(k+mmax) = dble(dfac)
            endif
  500   continue
c-----
c       now convert from partials with respect to changes in layer
c       boundary to changes in layer thickness
c-----
c-----
c           up to this point the dcdh are changes to phase velocity if
c           if the layer boundary changes. Here we change this to mean
c           the dc/dh for a change in layer thickness
c
c           A layer becomes thicker if the base increases and the top
c           decreases its position. The dcdh to this point indicates 
c           the effect of moving a boundary down. Now we convert to
c           the effect of changing a layer thickness.
c-----
            do 505 i=1,mmax-1
                sum = 0.0
                do 506 j=i+1,mmax
                    sum = sum + dcdb(mmax+j)
  506           continue
                dcdb(mmax+i) = sum
  505       continue
            dcdb(mmax+mmax) = 0.0
        return
        end

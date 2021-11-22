module srfdis96_module
    implicit none
    integer, parameter :: LER=0,LIN=5,LOT=6,&
        NL=200,NLAY=200,NL2=NL+NL,NP=512
    contains

    subroutine getmod_from_file(mname,&
        title,iiso,iunit,iflsph,idimen,icnvel,&
        mmax,d,a,b,rho,qa,qb,etap,etas,&
        frefp,frefs,refdep)
    !call getmod(2,'tmpsrfi.17',mmax,title,iunit,iiso,iflsph,
    !      idimen,icnvel,ierr,.false.)
    implicit none
    character mname*(*), title*(*)
    integer*4 mmax, iunit, iiso, iflsph, idimen, icnvel
    real d(:),a(:),b(:),rho(:)
    real qa(:),qb(:),etap(:),etas(:)
    real frefp(:), frefs(:)
    real refdep

    character string*80
    integer*4 ierr
    logical ext
    character ftype*80
    integer lun, j, i, irefdp


!-----
!       test to see if the file exists
!-----
    ierr = 0
!    write(LOT,*) 'ierr=', ierr
    inquire(file=mname,exist=ext)
    if(.not.ext)then
        ierr = -1
        write(LER,*)'Model file does not exist'
        return
    endif
!-----
!           open the file
!-----
    lun = 2
    open(lun,file=mname,status='old',form='formatted',&
        access='sequential')
    rewind lun

!-----
!       verify the file type
!-----
!-----
!       LINE 01
!-----
    read(lun,'(a)')ftype
    if(ftype(1:5).ne.'model' .and. ftype(1:5).ne.'MODEL')then
        ierr = -2
        write(LER,*)'Model file is not in model format'
        close(lun)
        return
    endif
!-----
!       LINE 02
!-----
    read(lun,'(a)')title
!-----
!       LINE 03
!-----
    read(lun,'(a)')string
    if(string(1:3).eq.'ISO' .or. string(1:3).eq.'iso')then
        iiso = 0
    else if(string(1:3).eq.'TRA' .or. string(1:3).eq.'tra')then
        iiso = 1
    else if(string(1:3).eq.'ANI' .or. string(1:3).eq.'ani')then
        iiso = 2
    endif
!-----
!       LINE 04
!-----
    read(lun,'(a)')string
    if(string(1:3).eq.'KGS' .or. string(1:3).eq.'kgs')then
        iunit = 0
    endif
!-----
!       LINE 05
!-----
    read(lun,'(a)')string
    if(string(1:3).eq.'FLA' .or. string(1:3).eq.'fla')then
        iflsph = 0
    else if(string(1:3).eq.'SPH' .or. string(1:3).eq.'sph')then
        iflsph = 1
    endif
!-----
!       LINE 06
!-----
    read(lun,'(a)')string
    if(string(1:3).eq.'1-d' .or. string(1:3).eq.'1-D')then
        idimen = 1
    else if(string(1:3).eq.'2-d' .or. string(1:3).eq.'2-D')then
        idimen = 2
    else if(string(1:3).eq.'3-d' .or. string(1:3).eq.'3-D')then
        idimen = 3
    endif
!-----
!       LINE 07
!-----
    read(lun,'(a)')string
    if(string(1:3).eq.'CON' .or. string(1:3).eq.'con')then
        icnvel = 0
    else if(string(1:3).eq.'VAR' .or. string(1:3).eq.'var')then
        icnvel = 1
    endif
!-----
!       get lines 8 through 11
!-----
    do i=8,11
        read(lun,'(a)')string
    end do
!-----
!       get model specifically for 1-D flat isotropic
!-----
!-----
!       get comment line
!-----
    read(lun,'(a)')string
    mmax = 0
    refdep = 0.0
    irefdp = 0
    if(iiso.eq.0)then
    do
        j = mmax +1
        read(lun,*,err=9000,end=9000)d(j),a(j),b(j),&
            rho(j),qa(j),qb(j),etap(j),etas(j),&
            frefp(j),frefs(j)
        if(d(j).lt.0.0)then
            d(j) = -d(j)
            refdep = refdep + d(j)
            irefdp = j
        endif
        mmax = j
    end do
9000 continue
    endif
    return   
    end subroutine getmod_from_file

    subroutine sphere(n,ifunc,iflag, &
        d,a,b,rho,rtp,dtp,btp,mmax)
!-----
!     Transform spherical earth to flat earth
!
!     Schwab, F. A., and L. Knopoff (1972). Fast surface wave and free
!     mode computations, in  Methods in Computational Physics, 
!         Volume 11,
!     Seismology: Surface Waves and Earth Oscillations,  
!         B. A. Bolt (ed),
!     Academic Press, New York
!
!     Love Wave Equations  44, 45 , 41 pp 112-113
!     Rayleigh Wave Equations 102, 108, 109 pp 142, 144
!
!     Revised 28 DEC 2007 to use mid-point, assume linear variation in
!     slowness instead of using average velocity for the layer
!     Use the Biswas (1972:PAGEOPH 96, 61-74, 1972) density mapping
!
!     ifunc   I*4 1 - Love Wave
!                 2 - Rayleigh Wave
!     iflag   I*4 0 - Initialize
!                 1 - Make model  for Love or Rayleigh Wave
!-----
!       parameter(NL=200,NP=512)
!       real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
    implicit none
    integer n
    real*4 d(n),a(n),b(n),rho(n),rtp(n),dtp(n),btp(n)
!       common/modl/ d,a,b,rho,rtp,dtp,btp
!       common/para/ mmax,llw,twopi
    double precision z0,z1,r0,r1,dr,ar,tmp
    integer, save :: dhalf
    integer :: i, ifunc, iflag, mmax

! debug
!   print *, 'size(d,a,b,rho) =', size(d), size(a), size(b), size(rho)
!   print *, 'size(rtp,dtp,btp)=', size(rtp), size(dtp), &
!       size(btp)

    ar=6370.0d0
    dr=0.0d0
    r0=ar
    d(mmax)=1.0
    if(iflag.eq.0) then
        do i=1,mmax
            dtp(i)=d(i)
            rtp(i)=rho(i)
        end do
        do i=1,mmax
            dr=dr+dble(d(i))
            r1=ar-dr
            z0=ar*dlog(ar/r0)
            z1=ar*dlog(ar/r1)
            d(i)=z1-z0
    !-----
    !               use layer midpoint
    !-----
            TMP=(ar+ar)/(r0+r1)
            a(i)=a(i)*tmp
            b(i)=b(i)*tmp
            btp(i)=tmp
            r0=r1
        end do
        dhalf = d(mmax)
    else
        d(mmax) = dhalf
        do i=1,mmax
            if(ifunc.eq.1)then
                rho(i)=rtp(i)*btp(i)**(-5)
            else if(ifunc.eq.2)then
                rho(i)=rtp(i)*btp(i)**(-2.275)
            endif
        end do
    endif
    d(mmax)=0.0
    return
    end

    subroutine zero_cb_and_c(n,kmax,cb,c)
    implicit none
    ! set cb and c to 0
    integer n
    double precision cb(n), c(n)
    integer kmax, i
    ! write(LOT,*) 'kmax', kmax
    ! write(LOT,*) 'cb', (cb(i),i=1,kmax)
    ! write(LOT,*) 'c', (c(i),i=1,kmax)
    do i=1,kmax
        cb(i) = 0.0d0
        c(i) = 0.0d0
    end do
    ! write(LOT,*) kmax
    ! write(LOT,*) (cb(i),i=1,kmax)
    ! write(LOT,*) (c(i),i=1,kmax)
    end subroutine zero_cb_and_c

    subroutine print_warning_for_no_zero_found(&
        nlayer, nperiod,&
        ifunc, &
        iq,is,ie,cc,cm,c1,&
        mmax,d,a,b,rho,&
        k,t,c)
    implicit none
    integer nlayer, nperiod
    integer ifunc,iq,is,ie
    double precision cc,cm,c1
    integer mmax,k
    real(kind=4) t(nperiod),d(nlayer),a(nlayer),b(nlayer),rho(nlayer)
    double precision c(nperiod)

    integer i

    write(LOT,*)'improper initial value in disper - no zero found'
    write(LOT,*)'in fundamental mode '
    write(LOT,*)'This may be due to low velocity zone '
    write(LOT,*)'causing reverse phase velocity dispersion, '
    write(LOT,*)'and mode jumping.'
    write(LOT,*)'due to looking for Love waves in a halfspace'
    write(LOT,*)'which is OK if there are Rayleigh data.'
    write(LOT,*)'If reverse dispersion is the problem,'
    write(LOT,*)'Get present model using OPTION 28, edit sobs.d,'
    write(LOT,*)'Rerun with onel large than 2'
    write(LOT,*)'which is the default '
!-----
!   if we have higher mode data and the model does not find that
!   mode, just indicate (itst=0) that it has not been found, but
!   fill out file with dummy results to maintain format - note
!   eigenfunctions will not be found for these values. The subroutine
!   'amat' in 'surf' will worry about this in building up the
!   input file for 'surfinv'
!-----
    write(LOT,*)'ifunc = ',ifunc ,' (1=L, 2=R)'
    write(LOT,*)'mode  = ',iq-1
    write(LOT,*)'period= ',t(k), ' for k,is,ie=',k,is,ie
    write(LOT,*)'cc,cm = ',cc,cm
    write(LOT,*)'c1    = ',c1
    write(LOT,*)'d,a,b,rho (d(mmax)=control ignore)'
    write(LOT,'(4f15.5)')(d(i),a(i),b(i),rho(i),i=1,mmax)
    write(LOT,*)' c(i),i=1,k (NOTE may be part)'
    write(LOT,*)(c(i),i=1,k)
    end subroutine print_warning_for_no_zero_found

    subroutine write_to_file_at_end_of_each_mode(ifunc,k,ie,&
        itst,iq)
    ! for some failed cases
    implicit none
    integer ifunc,k,ie,itst,iq
    real*4 t(NP)

    integer i
    real*4 t1a
    do i=k,ie
        t1a=t(i)
        write(ifunc) itst,iq,t1a,t1a,t1a,t1a
    end do
    end subroutine

    subroutine get_initial_phase_velocity_estimate(nperiod,&
        c1,clow,&
        ifirst,k,is,iq,cc,c,dc,one,onea)
    implicit none
    integer nperiod
    double precision c1,clow,cc,cm,dc,one,onea,c(nperiod)
    integer ifirst,k,is,iq

    if(k.eq.is .and. iq.eq.1)then
        c1 = cc
        clow = cc
        ifirst = 1
    elseif(k.eq.is .and. iq.gt.1)then
        c1 = c(is) + one*dc
        clow = c1
        ifirst = 1
    elseif(k.gt.is .and. iq.gt.1)then
        ifirst = 0
        clow = c(k) + one*dc
        c1 = c(k-1)
        if(c1 .lt. clow)c1 = clow
    elseif(k.gt.is .and. iq.eq.1)then
        ifirst = 0
        c1 = c(k-1) - onea*dc
        clow = cm
    endif
    end subroutine get_initial_phase_velocity_estimate

    subroutine write_to_file_before_searching_velocity(n,ifunc,&
        mmax,nsph,&
        btp,dtp,&
        d,a,b,rho,&
        kmax,igr,h)
    integer n,ifunc,mmax,nsph,kmax,igr
    real*4 btp(n),dtp(n),d(n),a(n),b(n),rho(n),h

    integer i

    write(ifunc) mmax,nsph
    write(ifunc) (btp(i),i=1,mmax)
    write(ifunc) (dtp(i),i=1,mmax)
    do i=1,mmax
        write(ifunc) d(i),a(i),b(i),rho(i)
    end do
    write(ifunc) kmax,igr,h
    end subroutine write_to_file_before_searching_velocity

!   subroutine get_maximum_and_minimum_velocities(betmx,betmn,&
!       jmn,jsol,mmax,a,b)
!   real betmx, betmn
!   integer mmax,jmn,jsol
!   real*4 a(NL),b(NL)
!
!   integer i
!
!   jmn = 1
!   betmx=-1.e20
!   betmn=1.e20
!   write(LOT,*) 'before loop', jmn, betmx, betmn, mmax
!   do i=1,mmax
!       ! find min velocity
!       if(b(i).gt.0.01 .and. b(i).lt.betmn)then
!           betmn = b(i)
!           jmn = i
!           jsol = 1
!       elseif(b(i).le.0.01 .and. a(i).lt.betmn)then
!           betmn = a(i)
!           jmn = i
!           jsol = 0
!       endif
!       ! find max velocity
!       if(b(i).gt.betmx) betmx=b(i)
!       write(LOT,*) 'i=', i, jmn, jsol,betmx, betmn
!   end do
!   write(LOT,*) 'out loop'
!   end subroutine get_maximum_and_minimum_velocities

    subroutine get_start_value_for_phase_velocity(n, betmn,jsol,&
        jmn,a,b,ddc,cc,dc,c1,cm)
    real betmn,cc1,ddc
    integer n,jsol,jmn
    real*4 a(n),b(n)
    double precision cc,dc,c1,cm

    ! determine a starting value for phase velocity
    ! using the lowest velocity layer found in
    ! previous step

    ! get starting value for phase velocity, 
    ! which will correspond to the 
    ! VP/VS ratio
    if(jsol.eq.0)then
        ! water layer
        cc1 = betmn
    else
        ! solid layer
        ! solve halfspace period equation
        call gtsolh(a(jmn),b(jmn),cc1)
    endif
    ! back off a bit to get a starting value at 
    ! a lower phase velocity
    cc1=.95*cc1
    cc1=.90*cc1
    cc=dble(cc1)
    dc=dble(ddc)
    dc=dabs(dc)
    c1=cc
    cm=cc
    end subroutine get_start_value_for_phase_velocity
end module

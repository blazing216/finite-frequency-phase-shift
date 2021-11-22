module srfdrl96_module
implicit none
integer, parameter :: LER=0,LIN=5,LOT=6,NL=200,NL2=NL+NL
contains

subroutine read_model_from_tmpsrfi_06(lun,&
    mmax,nsph,mmax2, &
    btp,dtp,d,a,b,rho, &
    xmu)
implicit none
integer lun
integer mmax,nsph,mmax2
real btp(NL),dtp(NL),d(NL),a(NL),b(NL),rho(NL)
real xmu(NL)

integer i

read(lun) mmax,nsph
mmax2 = mmax + mmax
read(lun)(btp(i),i=1,mmax)
read(lun)(dtp(i),i=1,mmax)
do  i=1,mmax
    read(lun) d(i),a(i),b(i),rho(i)
    xmu(i)=sngl(dble(rho(i))*dble(b(i))*dble(b(i)))
    ! xlam(i)=rho(i)*(a(i)*a(i)-2.*b(i)*b(i))
end do
end subroutine read_model_from_tmpsrfi_06

subroutine check_for_water_layer(b,d,&
    kw,ll)
implicit none
real b(NL),d(NL)
real dphw(NL),dphw0
integer kw,ll
if(b(1).le.0.0)then
    kw = 1
    ll = kw + 1
    dphw(1) = 0.0
    dphw(2) = d(1)
    dphw0   = d(1)
else
    kw = 0
    ll = kw + 1
endif
end subroutine check_for_water_layer

subroutine write_Love_derivates_to_tmpsrfi_05_09(lun5,lun9,&
    igr,itst,ip,ig,mode,mmax,mmax2,&
    t0,c,ugr,dcdb,dudb)
integer lun5,lun9
integer igr,itst,ip,ig,mode,mmax,mmax2
real t0,c,ugr,dcdb(NL2),dudb(NL2)

integer i

if(igr.eq.0 .or. igr.eq.2) then
    write(lun5) itst,ip,mode,t0
    if(itst.ne.0) then
        write(lun5) c,(dcdb(i),i=1,mmax)
        write(lun5)   (dcdb(i),i=mmax+1,mmax2)
    endif
    if(igr.eq.2)then
        write(lun9) itst,ig,mode,t0
        if(itst.ne.0) then
            write(lun9) ugr,(dudb(i),i=1,mmax)
            write(lun9) (dudb(i),i=mmax+1,mmax2)
            write(lun9) c,(dcdb(i),i=1,mmax)
            write(lun9) (dcdb(i),i=mmax+1,mmax2)
        endif
    endif
else if(igr.eq.1) then
    write(lun5) itst,ig,mode,t0
    if(itst.ne.0) then
        write(lun5) ugr,(dudb(i),i=1,mmax)
        write(lun5)   (dudb(i),i=mmax+1,mmax2)
        write(lun5) c,(dcdb(i),i=1,mmax)
        write(lun5)   (dcdb(i),i=mmax+1,mmax2)
    endif
endif
    
end subroutine write_Love_derivates_to_tmpsrfi_05_09

subroutine sphericity_correction(nsph,igr,&
    om,c,ugr,mmax2,&
    dcdb,&
    dudb,btp,dtp)
integer nsph,igr
real om,c,ugr
integer mmax2
real dcdb(NL2)
real dudb(NL2),btp(NL),dtp(NL)

if(nsph.gt.0)then
    if(igr.eq.0)then
        call splove(om,c,ugr,mmax2,0,&
            dcdb,&
            dudb,btp,dtp)

    else if(igr.eq.1)then
        call splove(om,c,ugr,mmax2,1,&
            dcdb,&
            dudb,btp,dtp)

    else if(igr.eq.2)then
        call splove(om,c,ugr,mmax2,0,&
            dcdb,&
            dudb,btp,dtp)
        call splove(om,c,ugr,mmax2,1,&
            dcdb,&
            dudb,btp,dtp)
    endif
endif
    
end subroutine sphericity_correction


subroutine copy_group_velocity_from_tmpsrfi_09_to_05(lun5,lun9,&
    mmax,mmax2)
integer lun5,lun9
integer itst,ig,mode,mmax,mmax2
real t0,ugr,c,dudb(NL2),dcdb(NL2)

integer i,j
do i=1,9000
    read(lun9,end=900) itst,ig,mode,t0
    write(lun5) itst,ig,mode,t0
    if(itst.ne.0) then
        read(lun9) ugr,(dudb(j),j=1,mmax)
        read(lun9)(dudb(j),j=mmax+1,mmax2)
        read(lun9) c, (dcdb(j),j=1,mmax)
        read(lun9)(dcdb(j),j=mmax+1,mmax2)

        write(lun5) ugr,(dudb(j),j=1,mmax)
        write(lun5)(dudb(j),j=mmax+1,mmax2)
        write(lun5)c, (dcdb(j),j=1,mmax)
        write(lun5)(dcdb(j),j=mmax+1,mmax2)

    endif
end do
900 continue  
end subroutine copy_group_velocity_from_tmpsrfi_09_to_05

subroutine assemble_uu(uu,ut,tt)
real uu(NL,2), ut(NL), tt(NL)
uu(:,1) = ut
uu(:,2) = tt
end subroutine assemble_uu

subroutine distribute_uu(uu,ut,tt)
real uu(NL,2), ut(NL), tt(NL)
ut = uu(:,1)
tt = uu(:,2)
end subroutine distribute_uu

end module srfdrl96_module















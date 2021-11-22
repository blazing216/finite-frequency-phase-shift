module srfdrr96_module
implicit none
integer, parameter :: LER=0,LIN=5,LOT=6,NL=200,NL2=NL+NL
contains
subroutine read_model_from_tmpsrfi_07(lun,&
    mmax,nsph,mmax2, &
    btp,dtp,d,a,b,rho, &
    xmu,xlam)
implicit none
integer lun
integer mmax,nsph,mmax2
real*4 btp(NL),dtp(NL),d(NL),a(NL),b(NL),rho(NL)
double precision xmu(NL),xlam(NL)

integer i

read(lun) mmax,nsph
mmax2 = mmax + mmax
read(lun)(btp(i),i=1,mmax)
read(lun)(dtp(i),i=1,mmax)
do i=1,mmax
    read(lun) d(i),a(i),b(i),rho(i)
    xmu(i)=rho(i)*b(i)*b(i)
    xlam(i)=rho(i)*(a(i)*a(i)-2.*b(i)*b(i))
end do
end subroutine read_model_from_tmpsrfi_07

subroutine check_for_water_layer(b,d,&
    kw,ll,dphw,dphw0)
implicit none
real b(NL),d(NL)
integer kw,ll
double precision dphw(NL),dphw0
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


end module srfdrr96_module


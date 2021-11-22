#####
# LINUX gcc/gfortran -frecord-marker=4 Compilers
#####
# -std=legacy to avoid compliation warnings for obselete features
# -fPIC for future compilation to a dynamic library to be called
#    from c and python
#FCMP=gfortran -O -fPIC -Wunused-dummy-argument -Wunused -Wno-unused-variable -Wall -fcheck=all
#FCMP=gfortran -O -fPIC -Wall
FCMP=gfortran -O -fPIC -std=legacy
#FCMP=gfortran -pg
CCMP=gcc 
MCHDEP=mchdep.o
# MCHCMD=mnmarg.o mgtarg.o ssytem.o
# CALPLTF=../../lib/libcalpltf.a
RM= rm -f
MV= mv

#DEST=../bin
#LIBDEST=/home/xuyh/Autocorrelation/Programs/lib
LIBDEST=../../lib

%.o: %.f
	$(FCMP) $(FFLAG) -c $<

%.o: %.f90
	$(FCMP) $(FFLAG) -c $<	

all: libsrfdis96.so  \
	libsrfdrr96.so \
	cython
	#libsrfdrl96.so \
#all: srfdis96 libsrfdis96.so  \
#	srfdrr96 libsrfdrr96.so \
#	srfdrl96 libsrfdrl96.so \
#	cython

cython: srfdrr96_cython.pyx libsrfdis96.so libsrfdrr96.so
	python setup.py build_ext --inplace
	mv `pwd`/surf96py.cpython*.so $(LIBDEST)/

libsrfdis96.so: srfdis96_mod.o srfdis96.o igetmod.o 
	$(FCMP) $(FFLAG) -shared -o libsrfdis96.so srfdis96.o igetmod.o srfdis96_mod.o
	rm -f $(LIBDEST)/libsrfdis96.so
	#ln -s `pwd`/libsrfdis96.so $(LIBDEST)/libsrfdis96.so

#srfdis96: srfdis96_mod.o srfdis96.o igetmod.o 
#	$(FCMP) $(FFLAG) -o srfdis96 srfdis96.o igetmod.o srfdis96_mod.o
#	$(MV) srfdis96 $(DEST)

#libsrfdrl96.so: srfdrl96_mod.o srfdrl96.o 
#	$(FCMP) $(FFLAG) -shared -o libsrfdrl96.so srfdrl96.o srfdrl96_mod.o
#	rm -f $(LIBDEST)/libsrfdrl96.so
#	ln -s `pwd`/libsrfdrl96.so $(LIBDEST)/libsrfdrl96.so

#srfdrl96: srfdrl96_mod.o srfdrl96.o 
#	$(FCMP) $(FFLAG) -o srfdrl96 srfdrl96.o srfdrl96_mod.o
#	$(MV) srfdrl96 $(DEST)

libsrfdrr96.so: srfdrr96_mod.o srfdrr96.o 
	$(FCMP) $(FFLAG) -shared -o libsrfdrr96.so srfdrr96.o srfdrr96_mod.o
	rm -f $(LIBDEST)/libsrfdrr96.so
	#ln -s `pwd`/libsrfdrr96.so $(LIBDEST)/libsrfdrr96.so

#srfdrr96: srfdrr96_mod.o srfdrr96.o 
#	$(FCMP) $(FFLAG) -o srfdrr96 srfdrr96.o srfdrr96_mod.o 
#	$(MV) srfdrr96 $(DEST)

clean:
	$(RM) *.o *.mod *.so


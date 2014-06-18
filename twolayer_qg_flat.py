from pylab import *

#import numpy
#np.use_fastnumpy = True
import mkl

import time

mkl.set_num_threads(1)

start_time = time.time()

#=======================================
#   Function Definitions
#=======================================

# Invert PV for streamfunction
def invph(zh1,zh2,a11,a12,a21,a22):
  ph1=a11*zh1+a12*zh2
  ph2=a21*zh1+a22*zh2
  return ph1, ph2

# compute u and v from streamfunction
def caluv(ph,k,l):
  u=-real(ifft2(1j*l*ph));
  v= real(ifft2(1j*k*ph));
  return u, v

# compute advection in grid spave (returns qdot in fourier space)
def advect(q,u,v,k,l):
  qdot=1j*k*fft2(u*q)+1j*l*fft2(v*q);
  return qdot

# ===================================================
# Here starts the actual model
#====================================================

dt= 16.0*200.; 
           
tplot=100000. #interval for plots (in timesteps)
tcfl=1000. #interval for cfl writeout (in timesteps)
tmax=15.*365*86400; # total time of integration
tavestart=5.*365*86400; # start time for averaging
taveint=100; # time interval use for averaging in timesteps (for performance purposes the averaging does not use every time-step)
tpickup=365*86400; # time interval to write out pickup fields ("experimental")

nx=64;ny=nx; # grid resolution (some parts of the model (or diagnostics) might actually assume nx=ny - check before making different choice)

L=1e6; W=L; # Domain size
[x,y]=meshgrid(linspace(1/(2*nx),1,nx)*L,linspace(1/(2*ny),1,ny)*W);
# initial conditions: (PV anomalies)
q1=1e-6*dot(ones((ny,1)),rand(1,nx))+1e-7*rand(ny,nx);
q2=0*x;


beta=1.6e-11;
rek=1./25./86400.; # Linear drag in lower layer


Rd=12000.0 # Deformation radius
delta=1.0  # Layer thickness ratio (H1/H2) (currently diagnostics assume delta=1!)

# Background zonal flow (m/s):
U1=0.025; 
U2=0.0;
U=U1-U2;


k0x=2*pi/L;
k0y=2*pi/W;
kk=fftfreq(nx,L/nx)*2*pi;
ll=fftfreq(ny,W/ny)*2*pi;

[k,l]=meshgrid(kk,ll);


dx=L/nx;
dy=W/ny;

F1=1./Rd**2/(1.+delta);
F2=delta*F1;

beta1=beta+F1*(U1-U2);
beta2=beta-F2*(U1-U2);


# determine inversion matrix: psi=Aq (i.e. A=B**(-1) where q=B*psi)
# this gives a warning because wavenumber 0 entries cause div by 0
# they are set to zero later though (could probably write this more elegantly)
wv2=(k*k+l*l);
det=wv2*(wv2+F1+F2);
a11=-(wv2+F2)/det;
a12=-F1/det;
a21=-F2/det;
a22=-(wv2+F1)/det;

a11[0,0]=0.;
a12[0,0]=0.;
a21[0,0]=0.;
a22[0,0]=0.;


wv2i=1./wv2;
wv2i[0,0]=0.;


# this defines the spectral filter (following Arbic and Flierl, 2003)
cphi=0.65*pi; 
wvx=sqrt((k*dx)**2.+(l*dy)**2.);
filtr=exp(-18.4*(wvx-cphi)**4.)*(wvx>cphi)+(wvx<=cphi);
filtr[isnan(filtr)]=1.0;

t=0.;
tc=0;

ts=[];

qh1=fft2(q1);
qh2=fft2(q2);

# Set time-stepping parameters for very first timestep (Euler-forward stepping).
# Adams Bashford used thereafter and is set up at the end of the first time-step (see below)
dqh1dt_p=0;
dqh2dt_p=0;
dt0=dt;dt1=0;

close('all')

figure(num=1, figsize=(12,5))

# Initialization for diagnotics
count=0.;
q1sum=0.;
q2sum=0.;
KE1sum=0.;
KE2sum=0.;
dissum=0.;
gensum=0.;
enstsum=0.;
KE1specsum=0.;
KE2specsum=0.;
APEfluxsum=0.;
KEfluxsum=0.;
APEgenspecsum=0.;


# pickup goes here(remove cfl init for pickup) 
# (this is older and should be checked!)
# load pickup_negla_Leith_256.mat
cfl=array([]);

#------------------------------------------------
# Beginning of time-stepping loop
#------------------------------------------------

while t<=tmax+dt/2:
    
    q1=real(ifft2(qh1));
    q2=real(ifft2(qh2));
    [ph1,ph2]=invph(qh1,qh2,a11,a12,a21,a22);
    [u1,v1]=caluv(ph1,k,l);
    [u2,v2]=caluv(ph2,k,l);
 
    if ((tc%(tcfl)==0) and (tc>0)):
        cfl=append(cfl,(r_[abs(u1),abs(v1),abs(u2),abs(v2)]).max()*dt/dx)
        print cfl[-1]
        
          
  #Diagnostics (Note that these assume delta=1 ! - should be generalized):       
    if ((t>tavestart) & (tc%taveint==0)):  
        p1=real(ifft2(ph1));
        p2=real(ifft2(ph2));
        xi1=real(ifft2(-wv2*ph1));
        xi2=real(ifft2(-wv2*ph2));
        Jptpc=-advect(0.5*(p1-p2),0.5*(u1+u2),0.5*(v1+v2),k,l); 
        Jp1xi1=advect(xi1,u1,v1,k,l);
        Jp2xi2=advect(xi2,u2,v2,k,l);
        enstsum=enstsum + abs((qh1+qh2)/2.)**2.;   # barotropic Enstr. spec
        APEfluxsum= APEfluxsum +Rd**(-2.)*real((ph1-ph2)*conj(Jptpc)); 
        KEfluxsum=KEfluxsum + real(ph1*conj(Jp1xi1))+real(ph2*conj(Jp2xi2));
        KE1specsum=KE1specsum + 0.5*wv2*abs(ph1)**2;
        KE2specsum=KE2specsum + 0.5*wv2*abs(ph2)**2;  
        q2sum=q2sum+double(q2);
        q1sum=q1sum+double(q1);
        KE1sum=KE1sum+0.5*double(mean(mean(v1**2+u1**2.)));
        KE2sum=KE2sum+0.5*double(mean(mean(v2**2+u2**2.)));
        dissum= dissum + sum(sum(rek*wv2*abs(ph2)**2.));           
        gensum= gensum + (0.5*U*Rd**(-2.)*
                sum(sum(real(0.5*real(1j*k*(ph1+ph2)*conj(ph1-ph2)))))) # I tested that this gives the same result as the computation in real space
        APEgenspecsum=APEgenspecsum+U*Rd**(-2.)*real(0.25*real(1j*k*(ph1+ph2)*conj(ph1-ph2)));
        count=count+1.;
 
    # Runtime Plots:
    if (tc%tplot==0):
        t
        ts=[ts,t];    
        figure(1);
        clf()
        subplot(1,2,1)
        contourf(x,y,q1 +beta1*y,20)
        colorbar();
        sub=8;
        axis('equal')
        title('t = %g' % t);
        axis([0,L,0,W])
        subplot(1,2,2)
        contourf(x,y,q2+ beta2*y,20)
        colorbar()
        axis('equal')
        title('t = %g' % t);
        axis([0,L,0,W])
        show()
        pause(1e-6)
  
      
   # Compute tendencies:  
    dqh1dt=(-advect(q1,u1+U1,v1,k,l)-beta1*1j*k*ph1);
    dqh2dt=(-advect(q2,u2+U2,v2,k,l)-beta2*1j*k*ph2 + rek*wv2*ph2);  
       
    
    # Add time tendencies (using Adams-Bashfort):
     
    qh1=filtr*(qh1+dt0*dqh1dt+dt1*dqh1dt_p);
    qh2=filtr*(qh2+dt0*dqh2dt+dt1*dqh2dt_p);  
   
    dqh1dt_p=dqh1dt;
    dqh2dt_p=dqh2dt;
       
    # The actual Adams-Bashfort stepping can only be used starting at the
    # second time-step and is thus set here:   
    if tc==0:
        dt0=1.5*dt; dt1=-0.5*dt;
 
    # write out pickup file (gets overwritten - mostly for crashes, if desired at all)
    if (tc%tpickup==0):
        savez('./pickup.npz',t=t, tc=tc, qh1=qh1, qh2=qh2, cfl=cfl,
        dqh1dt_p=dqh1dt_p, dqh2dt_p=dqh2dt_p) 
   
    # advance time step      
    tc=tc+1;
    t=t + dt;

#------------------------------------------------
# End of time-stepping loop
#------------------------------------------------

# Compute average diagnostics
KE1=KE1sum/count;
KE2=KE2sum/count;
EKEgen=gensum/count/nx**2.
EKEdiss=dissum/count/nx**2.
Enstspec=enstsum/count;
APEfluxspec=APEfluxsum/count;
KEfluxspec=KEfluxsum/count;
APEgenspec=APEgenspecsum/count;
KE1spec=KE1specsum/count;
KE2spec=KE2specsum/count;
q1mean=q1sum/count;
q2mean=q2sum/count;
qh1=fft2(q1mean);
qh2=fft2(q2mean);
[ph1,ph2]=invph(qh1,qh2,a11,a12,a21,a22);
p1=real(ifft2(ph1));
p2=real(ifft2(ph2));
[u1,v1]=caluv(ph1,k,l);
[u2,v2]=caluv(ph2,k,l);


EKE1=KE1-0.5*double(mean(mean(v1**2+u1**2)));
EKE2=KE2-0.5*double(mean(mean(v2**2+u2**2)));

# Plot mean flow (useless without topography)
figure(2)
clf()
subplot(2,2,1)
pcolor(x,y,double(q1mean))
title('q_1');
colorbar;
sub=8
axis('equal')
axis([0,L,0,W])
subplot(2,2,2)
pcolor(x,y,double(q2mean))
colorbar;
axis('equal')
title('q_2');
axis([0,L,0,W])
subplot(2,2,3)
contourf(x,y,double(p1-U1*y)-mean(mean(double(p1-U1*y))),25)
axis('equal');title('\psi_1');
axis([0,L,0,W])
colorbar;
subplot(2,2,4)
contourf(x,y,double(p2)-mean(mean(double(p2))),25)
axis('equal');
colorbar;
axis([0,L,0,W])
title('\psi_2');



Etotspec=(KE1spec+KE2spec);    
    
Especx=sum(wv2i*Enstspec,0);    
Especx_1=sum(KE1spec,0);    
Especx_2=sum(KE2spec,0);    
Especx_nomean=sum(wv2i*(Enstspec-abs((qh1+qh2)/2)**2),0);    

# Compute spectral diagnostics as a function of total wavenumber
# i.e. average along circles in spectral space
# Notice that the second half of these all stays zero (it was just practical 
# to have the same length vectors as the wavenumber)
dk=k[0,1];
Especr=zeros((size(k,1)));
Especr_2=zeros((size(k,1)));
APEflux_specr=zeros((size(k,1)));   
KEflux_specr=zeros((size(k,1)));   
APEgen_specr=zeros((size(k,1)));    
for ii in range(1,size(k,1)/2):
    kk=k[0,ii]
    ivec=logical_and(wv2>=(kk-dk/2.)**2.,wv2<(kk+dk/2.)**2.)
    Especr[ii]=sum(wv2i[ivec]*Enstspec[ivec])/nx**4.
    Especr[ii]=Especr[ii]*(2.0*pi*abs(kk))/sum(sum(ivec))/dk
    Especr_2[ii]= sum(KE2spec[ivec])/nx**4.
    Especr_2[ii]=Especr_2[ii]*(2.0*pi*abs(kk))/sum(sum(ivec))/dk
    APEflux_specr[ii]=sum(APEfluxspec[ivec])/nx**4.
    APEflux_specr[ii]=APEflux_specr[ii]*(2.0*pi*abs(kk))/sum(sum(ivec))/dk
    KEflux_specr[ii]=sum(KEfluxspec[ivec])/nx**4.
    KEflux_specr[ii]=KEflux_specr[ii]*(2.0*pi*abs(kk))/sum(sum(ivec))/dk
    APEgen_specr[ii]=sum(APEgenspec[ivec])/nx**4.
    APEgen_specr[ii]=APEgen_specr[ii]*(2.0*pi*abs(kk))/sum(sum(ivec))/dk


# Plot barotropic EKE spectrum                
figure(3) 
clf()
loglog(k[0,1:ceil((size(k,1))/2)]*L/2/pi,Especr[1:ceil((size(k,1))/2)],'r');
loglog(k[0,2:ceil((size(k,1))/2)]*L/2/pi,5e-18*k[0,2:ceil((size(k,1))/2)]**(-3));
loglog(k[0,1:ceil((size(k,1))/6)]*L/2/pi,5e-29*k[0,1:ceil((size(k,1))/6)]**(-5));
xlim(1,(nx/2))
ylim(1e-5,1e-2)
show()


# Plot spectral energy budget
figure(4) 
clf()
semilogx(k[0,1:ceil(size(k,1)/2)]*L/2/pi,-k[0,1:ceil(size(k,1)/2)]*rek*2*Especr_2[1:ceil(size(k,1)/2)],'b')
semilogx(k[0,1:ceil(size(k,1)/2)]*L/2/pi,k[0,1:ceil(size(k,1)/2)]*KEflux_specr[1:ceil(size(k,1)/2)],'--b')
semilogx(k[0,1:ceil(size(k,1)/2)]*L/2/pi,k[0,1:ceil(size(k,1)/2)]*APEflux_specr[1:ceil(size(k,1)/2)],'--r')
semilogx(k[0,1:ceil(size(k,1)/2)]*L/2/pi,k[0,1:ceil(size(k,1)/2)]*APEgen_specr[1:ceil(size(k,1)/2)],'r')
semilogx(k[0,1:ceil(size(k,1)/2)]*L/2/pi,-k[0,1:ceil(size(k,1)/2)]*(-rek*2*Especr_2[1:ceil(size(k,1)/2)]
                            +KEflux_specr[1:ceil(size(k,1)/2)] +APEflux_specr[1:ceil(size(k,1)/2)]
                            +APEgen_specr[1:ceil(size(k,1)/2)]),':k')
xlim(1,(nx/2))
show()


           
run_time=time.time() - start_time

print 'elapsed time (in s): ' + repr(run_time)


savez('./out_flat_tauf25d_U025_beta16_64.npz', U1=U1, U2=U2, Rd=Rd, beta=beta, rek=rek, dt=dt,
               q1=q1, q2=q2, q1mean=q1mean, q2mean=q2mean, EKE1=EKE1, EKE2=EKE2,
               KE1=KE1, KE2=KE2, EKEgen=EKEgen, EKEdiss=EKEdiss,
               Enstspec=Enstspec, k=k, l=l, Especr=Especr,
               Especr_2=Especr_2, APEflux_specr=APEflux_specr, KEflux_specr=KEflux_specr,
               APEgen_specr=APEgen_specr, cfl=cfl,tmax=tmax)
               
               
                  
               
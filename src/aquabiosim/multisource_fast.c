/* Vector RK4 integrator for the v0.18 development branch. Seven energy channels. */
#include <math.h>
#include <stdlib.h>
/* Input columns: w0,T,days,GE,feeding_mode,offer_frac,update_days,FCE,relative,control_index. */
static int rates(int n,const double *in,const double *y,const double *cap,const double *eta0,
                 const double *adc,const double *urine,const double *rho,const double *anchor,
                 const double *params,int np,int day,double *out){
    double p=np>8?params[8]:1., gamma=0.;
    (void)gamma;
    for(int i=0;i<n;i++){
        const double *a=in+10*i;double w=y[7*i];
        if(!(w>0.)||!isfinite(w)||w>1e7)return 1;
        for(int j=0;j<7;j++)out[7*i+j]=0.;
        if(day>=(int)a[2])continue;
        double capacity=cap[i]*pow(w/10.,p)*10.;
        double offer=capacity;int mode=(int)a[4],control=(int)a[9];
        if(mode==1){double wc=y[7*control];offer=a[8]*cap[control]*pow(wc/10.,p)*10.;}
        if(mode==2||mode==3)offer=a[5]*anchor[i];
        if(mode==4)offer=a[8]*anchor[control];
        double intake=fmin(capacity,offer),eta=1./(1.+exp(-(eta0[i]+params[6]*log(w/10.))));
        double c=a[3]*intake,f=(1.-adc[i])*c,u=urine[i]*c,assimilated=c-f-u;
        double loss=(1.-eta)*assimilated,maint=6.14421235332821e-6*pow(2.,(a[1]-27.)/10.)*pow(w,2./3.);
        out[7*i]=(assimilated-loss-maint)/rho[i];out[7*i+1]=intake;
        out[7*i+2]=c;out[7*i+3]=f;out[7*i+4]=u;out[7*i+5]=loss;out[7*i+6]=maint;
    }return 0;
}
int aquasim(int n,const double *in,const double *cap,const double *eta,const double *adc,
            const double *urine,const double *rho,const double *params,int np,double step,double *out){
    int count=7*n,sub=(int)round(1./step),end=0;
    double *mem=calloc((size_t)count*6+(size_t)n,sizeof(double));if(!mem)return 2;
    double *y=mem,*a=y+count,*b=a+count,*c=b+count,*d=c+count,*tmp=d+count,*anchor=tmp+count;
    for(int i=0;i<n;i++){y[7*i]=in[10*i];anchor[i]=in[10*i];if(in[10*i+2]>end)end=(int)in[10*i+2];}
    for(int day=0;day<end;day++){
        for(int i=0;i<n;i++){
            const double *v=in+10*i;int mode=(int)v[4],up=(int)v[6];
            if(day%up==0){
                if(mode==2)anchor[i]=y[7*i];
                if(mode==3&&day>0)anchor[i]*=1.+v[7]*v[5]*up;
                if(mode==4)anchor[i]=cap[i]*pow(y[7*i]/10.,np>8?params[8]:1.)*10.;
            }
        }
        for(int k=0;k<sub;k++){
            if(rates(n,in,y,cap,eta,adc,urine,rho,anchor,params,np,day,a))goto fail;
            for(int j=0;j<count;j++)tmp[j]=y[j]+step*a[j]/2.;
            if(rates(n,in,tmp,cap,eta,adc,urine,rho,anchor,params,np,day,b))goto fail;
            for(int j=0;j<count;j++)tmp[j]=y[j]+step*b[j]/2.;
            if(rates(n,in,tmp,cap,eta,adc,urine,rho,anchor,params,np,day,c))goto fail;
            for(int j=0;j<count;j++)tmp[j]=y[j]+step*c[j];
            if(rates(n,in,tmp,cap,eta,adc,urine,rho,anchor,params,np,day,d))goto fail;
            for(int j=0;j<count;j++)y[j]+=step*(a[j]+2*b[j]+2*c[j]+d[j])/6.;
        }
    }
    for(int j=0;j<count;j++){if(!isfinite(y[j]))goto fail;out[j]=y[j];}
    free(mem);return 0;
    fail:free(mem);return 1;
}

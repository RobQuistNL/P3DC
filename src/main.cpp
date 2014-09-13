/*

Title: Precise 3D Collisions Dll, Version 6.0 (P3DC.Dll V6.0 or simply P3DC V6.0)
Author: Brett Binnersley
Additional Credit: Samuel Hanson, Thomas Miller
Date (Creation): August 8, 2009
Date (Last Edit): March 16, 2012



**********************************
**********************************

Check out my most recent project:
GameDev Studio: http://gamedevstudio.yolasite.com/


License:
P3DC is free to use. If re-releasing source
code, you must include all authors names. No
need in a binary form (although it is appreciated)




Notes:


    P3DC is based off of Samuel Hanson's Modmod Collisions.
    Source Code: http://gmc.yoyogames.com/index.php?showtopic=329495

    Modmod Collisions uses some code by Thomas Miller. Please
    See article "A Fast Triangle-Triangle Intersection Test",
    Journal of Graphics Tools, 2(2), 1997
    Source Code: http://jgt.akpeters.com/papers/Moller97/tritri.html

    The source code for this (p3dc) is very sloppy. It was my first project other
    than a hello world application in C++. It is not optimized, and the coding
    does not meet any standards. Many parts of it simply do not make sense, and
    there are much better ways to do what was done. The code resembles C more than
    C++ due to the lack of use of classes. It was created for my own personal use,
    and was a learning experience. I decided to make it available for others. Do
    what ever you please with it.

    ~Have Fun, and enjoy!
*/
#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <cmath>
#include <vector>
#include <windows.h>
#include <stdio.h>
using std::vector;

//GLOBAL VARIABLES
vector< vector<double> > G_modellist;
vector<double> temp_vector;
bool hassplit=0;
int d3dterrain=0;
int d3dmodel[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int p3dcxlist[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int p3dcylist[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int p3dczlist[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int p3dctexture[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int p3dctextureid[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int p3dc_splitmodels[50][50];
int splitregionx,splitregiony;
double tridata[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0};
unsigned int ow_tri=0,ow_mod=0;
double vecrotx=0,vecroty=0,vecrotz=0;

double G_replacenum=0,normalx=0,normaly=0,normalz=0,addx1,addx2,addx3,addy1,addy2,addy3,addz1,addz2,addz3,
xpoint,ypoint,zpoint,Cx,Cy,Cz,Sx,Sy,Sz,Cx2,Cy2,Cz2,Sx2,Sy2,Sz2,addingtriangleid,trianglehit;

char bytes[128];

//Pre-Processors
#define export extern "C" __declspec (dllexport)
#define FABS(x) ((double)fabs(x))
#define EPSILON 0.000001
#define CROSS(dest,v1,v2)                      \
              dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
              dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
              dest[2]=v1[0]*v2[1]-v1[1]*v2[0];


#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) dest[0]=v1[0]-v2[0]; dest[1]=v1[1]-v2[1]; dest[2]=v1[2]-v2[2];
#define ADD(dest,v1,v2) dest[0]=v1[0]+v2[0]; dest[1]=v1[1]+v2[1]; dest[2]=v1[2]+v2[2];
#define MULT(dest,v,factor) dest[0]=factor*v[0]; dest[1]=factor*v[1]; dest[2]=factor*v[2];
#define SORT(a,b)       \
    if(a>b){    \
    double c=a; \
    a=b;     \
    b=c;     \
}
#define ISECT(VV0,VV1,VV2,D0,D1,D2,isect0,isect1) \
              isect0=VV0+(VV1-VV0)*D0/(D0-D1);    \
              isect1=VV0+(VV2-VV0)*D0/(D0-D2);
#define NEWCOMPUTE_INTERVALS(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,A,B,C,X0,X1){ \
    if(D0D1>0.0f){                                                 \
        A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
    }else if(D0D2>0.0f){                                           \
        A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
    }else if(D1*D2>0.0f || D0!=0.0f){                              \
        A=VV0; B=(VV1-VV0)*D0; C=(VV2-VV0)*D0; X0=D0-D1; X1=D0-D2; \
    }else if(D1!=0.0f){                                            \
        A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
    }else if(D2!=0.0f){                                            \
        A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
    }else{return 0;}                                                \
}
/*
Triangle/triangle intersection test routine,
by Tomas Moller, 1997.
See article "A Fast Triangle-Triangle Intersection Test",
Journal of Graphics Tools, 2(2), 1997
updated: 2001-06-20 (added line of intersection)
*/
inline bool n_t_i(double V0[3],double V1[3],double V2[3],double U0[3],double U1[3],double U2[3]){
  double E1[3],E2[3],N1[3],d1,du0,du1,du2,du0du1,du0du2;

  SUB(E1,V1,V0);
  SUB(E2,V2,V0);
  CROSS(N1,E1,E2);
  d1=-DOT(N1,V0);
  du0=DOT(N1,U0)+d1;
  du1=DOT(N1,U1)+d1;
  du2=DOT(N1,U2)+d1;
  du0du1=du0*du1;
  du0du2=du0*du2;

  if(du0du1>0.0f && du0du2>0.0f) return 0;
  double N2[3];
  SUB(E1,U1,U0);
  SUB(E2,U2,U0);
  CROSS(N2,E1,E2);
  double d2=-DOT(N2,U0),
  dv0=DOT(N2,V0)+d2,
  dv1=DOT(N2,V1)+d2,
  dv2=DOT(N2,V2)+d2,
  dv0dv1=dv0*dv1,
  dv0dv2=dv0*dv2;

  if(dv0dv1>0.0f && dv0dv2>0.0f)return 0;
  double D[3];
  CROSS(D,N1,N2);
  double max=FABS(D[0]),
  bb=FABS(D[1]),
  cc=FABS(D[2]);
  short index=0;
  if(bb>max) max=bb,index=1;else{
  if(cc>max) max=cc,index=2;}

  double vp0=V0[index],
  vp1=V1[index],
  vp2=V2[index],
  up0=U0[index],
  up1=U1[index],
  up2=U2[index],
  a,b,c,x0,x1,d,e,f,y0,y1;

  NEWCOMPUTE_INTERVALS(vp0,vp1,vp2,dv0,dv1,dv2,dv0dv1,dv0dv2,a,b,c,x0,x1);
  NEWCOMPUTE_INTERVALS(up0,up1,up2,du0,du1,du2,du0du1,du0du2,d,e,f,y0,y1);

  double xx=x0*x1,isect1[2],isect2[2],
  yy=y0*y1,
  xxyy=xx*yy,
  tmp=a*xxyy;
  isect1[0]=tmp+b*x1*yy;
  isect1[1]=tmp+c*x0*yy;
  tmp=d*xxyy;
  isect2[0]=tmp+e*xx*y1;
  isect2[1]=tmp+f*xx*y0;
  SORT(isect1[0],isect1[1]);
  SORT(isect2[0],isect2[1]);

  if(isect1[1]<isect2[0] || isect2[1]<isect1[0]) return 0;
  return 1;
}

//Ray-Tri code by Samuel Hanson.
//Edited By Brett Binnersley
inline bool i_r_t(double orig[3], double dir[3],double vert0[3], double vert1[3], double vert2[3],double *t){
   double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3],det,u,v;

   SUB(edge1, vert1, vert0);
   SUB(edge2, vert2, vert0);
   CROSS(pvec, dir, edge2);
   det = DOT(edge1, pvec);
   SUB(tvec, orig, vert0);
   u = DOT(tvec, pvec);

    if (det > 0.0001){
          if (u < 0.0 || u > det)return 0;
          CROSS(qvec, tvec, edge1);
          v = DOT(dir, qvec);
          if (v < 0.0 || u + v > det)return 0;
    }else if(det < -0.0001){
        if (u > 0.0 || u < det)return 0;
        CROSS(qvec, tvec, edge1);
        v = DOT(dir, qvec);
        if (v > 0.0 || u + v < det)return 0;
    }else return 0;
    *t = DOT(edge2, qvec) * (1.0 / det);
    return 1;
}
inline double degtorad(double degrees) {
	return degrees*3.14159/180;
}
inline double radtodeg(double rad) {
	return rad*180/3.14159;
}
inline double lengthdir_x(double length, double angle){
	return cos(degtorad(angle))*length;
}
inline double lengthdir_y(double length, double angle){
	return -sin(degtorad(angle))*length;
}
inline double point_distance(double x1,double y1,double x2,double y2){
	return sqrt(((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2)));
}
inline double point_direction(double x1,double y1,double x2,double y2){
	return radtodeg((atan2(( y1 - y2),-(x1 - x2))));
}
inline void rotatepoint(double *x, double *y, double *z, double xorg, double yorg, double zorg){
//Rotate DOUBLE[*x,*y,*z] around DOUBLE[xorg,yorg,zorg] by DOUBLE[rotx,roty,rotz]
*x-=xorg;
*y-=yorg;
*z-=zorg;
double tt = Cx*(*y) - Sx*(*z);
*z = Sx*(*y) + Cx*(*z);*y = tt;
tt = Cy*(*z) - Sy*(*x);
*x = Sy*(*z) + Cy*(*x);*z = tt+zorg;
tt = Cz*(*x) - Sz*(*y);
*y = Sz*(*x) + Cz*(*y)+yorg;
*x = tt+xorg;
}
inline void rotatepoint_faster(double *x, double *y, double *z){
//Rotate DOUBLE[*x,*y,*z] around [0,0,0] by DOUBLE[rotx,roty,rotz]
double tt = Cx2*(*y) - Sx2*(*z);
*z = Sx2*(*y) + Cx2*(*z);*y = tt;
tt = Cy2*(*z) - Sy2*(*x);
*x = Sy2*(*z) + Cy2*(*x);*z = tt;
tt = Cz2*(*x) - Sz2*(*y);
*y = Sz2*(*x) + Cz2*(*y);*x = tt;
}
inline void rotatepoint_float(float *x, float *y, float *z){
//Rotate FLOAT[*x,*y,*z] around [0,0,0] by FLOAT[rotx,roty,rotz]
float tt = Cx2*(*y) - Sx2*(*z);
*z = Sx2*(*y) + Cx2*(*z);*y = tt;
tt = Cy2*(*z) - Sy2*(*x);
*x = Sy2*(*z) + Cy2*(*x);*z = tt;
tt = Cz2*(*x) - Sz2*(*y);
*y = Sz2*(*x) + Cz2*(*y);*x = tt;
}
inline double mmax(double v1, double v2, double v3){
    if(v1<v2) v1=v2;
    if(v1<v3) v1=v3;
    return v1;
}
inline double mmin(double v1, double v2, double v3){
    if(v1>v2) v1=v2;
    if(v1>v3) v1=v3;
    return v1;
}
inline int force_below(double value1, int value2){
    if(value1>=value2-1)return value2-1;
    if(value1<0)return 0;
    return (int)value1;
}
//Calculate the normals of the last triangle Added
inline void calc_normals(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3){
double ax,ay,az,bx,by,bz,m;
ax = x2-x1;
ay = y2-y1;
az = z2-z1;
bx = x3-x1;
by = y3-y1;
bz = z3-z1;
normalx = ay*bz-by*az;
normaly = az*bx-bz*ax;
normalz = ax*by-bx*ay;
m = sqrt(normalx*normalx+normaly*normaly+normalz*normalz);
if(m == 0.0)m = 0.000000001;
normalx /= m;
normaly /= m;
normalz /= m;
}
//Add a Triangle (model add triangle
void mat(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3){
	//[X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3]
	calc_normals(x1,y1,z1,x2,y2,z2,x3,y3,z3);
    temp_vector.push_back(x1);
    temp_vector.push_back(y1);
    temp_vector.push_back(z1);
    temp_vector.push_back(x2);
    temp_vector.push_back(y2);
    temp_vector.push_back(z2);
    temp_vector.push_back(x3);
    temp_vector.push_back(y3);
    temp_vector.push_back(z3);
    temp_vector.push_back(normalx);
    temp_vector.push_back(normaly);
    temp_vector.push_back(normalz);
	temp_vector.push_back(addingtriangleid);
}

export double gmn(){//Get # of Model models (Get Model Number)
	return G_modellist.size();
}
export double gmt(double arg0){//Model Get Triangles
	return G_modellist[(int) arg0].size()/13;
}
export double gms(double arg0, double arg1){//Get split model ID
	return (double)p3dc_splitmodels[(int)arg0/splitregionx][(int)arg1/splitregiony];
}
export double gtr(double arg0){//Get Triangle Data
	return tridata[(int) arg0];
}
export double gtm(){//The triangle hit
    return trianglehit;
}
//************************************************************************
//Create the collisions / Define the models                              *
//************************************************************************
//Internal
void int_apw(double x1, double y1, double z1, double x2, double y2, double z2){
mat(x1,y1,z1,x2,y2,z1,x2,y2,z2);
mat(x2,y2,z2,x1,y1,z1,x1,y1,z2);
}
void int_apf(double x1, double y1, double z1, double x2, double y2, double z2){
mat(x1,y1,z1, x2,y1,z2, x2,y2,z2);
mat(x2,y2,z2, x1,y1,z1, x1,y2,z1);
}
void int_apb(double x1, double y1, double z1, double x2, double y2, double z2){
mat(x1,y2,z1,x1,y2,z2,x2,y2,z1);
mat(x2,y2,z2,x2,y2,z1,x1,y2,z2);
mat(x2,y1,z1,x1,y1,z2,x1,y1,z1);
mat(x2,y1,z2,x1,y1,z2,x2,y1,z1);
mat(x1,y1,z1,x1,y1,z2,x1,y2,z1);
mat(x1,y2,z1,x1,y1,z2,x1,y2,z2);
mat(x2,y1,z2,x2,y1,z1,x2,y2,z1);
mat(x2,y1,z2,x2,y2,z1,x2,y2,z2);
mat(x1,y1,z2,x2,y1,z2,x1,y2,z2);
mat(x1,y2,z2,x2,y1,z2,x2,y2,z2);
mat(x2,y1,z1,x1,y1,z1,x1,y2,z1);
mat(x2,y1,z1,x1,y2,z1,x2,y2,z1);
}
void int_apc(double x1, double y1, double z1, double x2, double y2, double z2, double closed, double steps){
double a,b,c,d,e,f,g,h,r,s;
a=x1+(x2-x1)/2;
b=y1+(y2-y1)/2;
r=FABS((double)(x2-x1))/2;
s=FABS((double)(y2-y1))/2;
if(closed==1){
   for(int i=2;i<steps;i+=1){
   c=lengthdir_x(r,0);
   d=lengthdir_y(s,0);
   e=lengthdir_x(r,i*360/steps);
   f=lengthdir_x(r,(i-1)*360/steps);
   g=lengthdir_y(s,i*360/steps);
   h=lengthdir_y(s,(i-1)*360/steps);
   mat(a+c,b+d,z1,  a+e,b+g,z1, a+f,b+h,z1 );
   mat(a+c,b+d,z2,  a+e,b+g,z2, a+f,b+h,z2 );
   }
}

for(int i=0;i<steps;i+=1){
    c=lengthdir_x(r,(i+1)*360/steps);
    d=lengthdir_y(s,(i+1)*360/steps);
    e=lengthdir_x(r,i*360/steps);
    f=lengthdir_y(s,i*360/steps);
    mat(a+e,b+f,z1, a+e,b+f,z2, a+c,b+d,z1 );
    mat(a+e,b+f,z2, a+c,b+d,z1, a+c,b+d,z2 );
}
}
void int_apo(double x1, double y1, double z1, double x2, double y2, double z2, double closed, double steps){
double a,b,c,d,e,f,g,h,r,s;
a=x1+(x2-x1)/2;
b=y1+(y2-y1)/2;
r=FABS(x2-x1)/2;
s=FABS(y2-y1)/2;
if(closed==1){
   for(int i=2;i<steps;i+=1){
   c=lengthdir_x(r,0);
   d=lengthdir_y(s,0);
   e=lengthdir_x(r,i*360/steps);
   f=lengthdir_x(r,(i-1)*360/steps);
   g=lengthdir_y(s,i*360/steps);
   h=lengthdir_y(s,(i-1)*360/steps);
   mat(a+c,b+d,z1,  a+e,b+g,z1, a+f,b+h,z1 );
   }
}

for(int i=0;i<steps;i+=1){
    c=lengthdir_x(r,(i+1)*360/steps);
    d=lengthdir_y(s,(i+1)*360/steps);
    e=lengthdir_x(r,i*360/steps);
    f=lengthdir_y(s,i*360/steps);
    mat(a+e,b+f,z1, a,b,z2, a+c,b+d,z1 );
}
}

export double bdm(){//Begin Define Model
temp_vector.clear();
return G_modellist.size();
}
export double brm(double arg0){//Begin Replace Model
temp_vector.clear();
G_replacenum=arg0;
return G_replacenum;
}
export double edm(){//End Define Model
       G_modellist.push_back(temp_vector);
       double size = (double)(temp_vector.size()/13);
       temp_vector.clear();
	   return size;
}
export double erm(){//End Replace Model
       temp_vector.clear();
	   return 1;
}
export double bs3(double arg0, double arg1,double arg2, double arg3,double arg4, double arg5){//SPLIT THE MODEL (MUCH FASTER COLLISIONS)
double width,height,xreg,yreg,space,x1,x2,y1,y2,tx1,tx2,tx3,ty1,ty2,ty3,tz1,tz2,tz3,xmin,xmax,ymin,ymax,nx,ny,nz,tid;
unsigned int modnum=(unsigned int)arg0;
width=arg1;
height=arg2;
xreg=arg3;
yreg=arg4;
space=arg5;
splitregionx=(int)(arg1/arg3);
splitregiony=(int)(arg2/arg4);


for(int xx=0;xx<(int)xreg;xx++){
     for(int yy=0;yy<(int)yreg;yy++){

		 if(hassplit==0){
             temp_vector.clear();
             p3dc_splitmodels[xx][yy]=(int)G_modellist.size();
		 }else{
             temp_vector.clear();
             G_replacenum=p3dc_splitmodels[xx][yy];
		 }

		for(unsigned int i=0;i<G_modellist[modnum].size();i+=13){

		tx1=G_modellist[modnum][i];
        ty1=G_modellist[modnum][i+1];
		tz1=G_modellist[modnum][i+2];
        tx2=G_modellist[modnum][i+3];
        ty2=G_modellist[modnum][i+4];
		tz2=G_modellist[modnum][i+5];
        tx3=G_modellist[modnum][i+6];
        ty3=G_modellist[modnum][i+7];
		tz3=G_modellist[modnum][i+8];
		nx=G_modellist[modnum][i+9];
		ny=G_modellist[modnum][i+10];
		nz=G_modellist[modnum][i+11];
		tid=G_modellist[modnum][i+12];

		x1=mmin(tx1,tx2,tx3);
		y1=mmin(ty1,ty2,ty3);
		x2=mmax(tx1,tx2,tx3);
		y2=mmax(ty1,ty2,ty3);
		xmin=xx*width/xreg;
		xmax=(xx+1)*width/xreg;
		ymin=yy*height/yreg;
		ymax=(yy+1)*height/yreg;

			if(!(x1>xmax+space || x2<xmin-space || y1>ymax+space || y2<ymin-space)){
                temp_vector.push_back(tx1);
                temp_vector.push_back(ty1);
                temp_vector.push_back(tz1);
                temp_vector.push_back(tx2);
                temp_vector.push_back(ty2);
                temp_vector.push_back(tz2);
                temp_vector.push_back(tx3);
                temp_vector.push_back(ty3);
                temp_vector.push_back(tz3);
                temp_vector.push_back(nx);
                temp_vector.push_back(ny);
                temp_vector.push_back(nz);
                temp_vector.push_back(tid);
			}
		}
		if(hassplit==0){
            G_modellist.push_back(temp_vector);
            temp_vector.clear();
		}else{
            G_modellist[(int) G_replacenum]=temp_vector;
            temp_vector.clear();
		}
     }
}
hassplit=1;
return 1;
}
export double stm(double arg0){//Set the triangles to be added id
addingtriangleid=arg0;
return 1;
}

//************************************************************************
//Add to the collision model                                             *
//************************************************************************

export double mat_exported(double arg0, double arg1,double arg2, double arg3,double arg4,
                           double arg5, double arg6, double arg7,double arg8){//Add a Triangle
	//[X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3]
	double x1=arg0,y1=arg1,z1=arg2,x2=arg3,y2=arg4,z2=arg5,x3=arg6,y3=arg7,z3=arg8;
	calc_normals(x1,y1,z1,x2,y2,z2,x3,y3,z3);
    temp_vector.push_back(x1);
    temp_vector.push_back(y1);
    temp_vector.push_back(z1);
    temp_vector.push_back(x2);
    temp_vector.push_back(y2);
    temp_vector.push_back(z2);
    temp_vector.push_back(x3);
    temp_vector.push_back(y3);
    temp_vector.push_back(z3);
    temp_vector.push_back(normalx);
    temp_vector.push_back(normaly);
    temp_vector.push_back(normalz);
	temp_vector.push_back(addingtriangleid);
	return temp_vector.size()-13;
}
export double apw(double arg0, double arg1,double arg2, double arg3,double arg4, double arg5){//Add a Wall
//double x1, double y1, double z1, double x2, double y2, double z2
double x1=arg0,y1=arg1,z1=arg2,x2=arg3,y2=arg4,z2=arg5;
mat(x1,y1,z1,x2,y2,z1,x2,y2,z2);
mat(x2,y2,z2,x1,y1,z1,x1,y1,z2);
return temp_vector.size()-26;//2*13
}
export double apf(double arg0, double arg1,double arg2, double arg3,double arg4, double arg5){//Add a Floor
double x1=arg0,y1=arg1,z1=arg2,x2=arg3,y2=arg4,z2=arg5;
mat(x1,y1,z1, x2,y1,z2, x2,y2,z2);
mat(x2,y2,z2, x1,y1,z1, x1,y2,z1);
return temp_vector.size()-26;//2*13
}
export double apb(double arg0, double arg1,double arg2, double arg3,double arg4, double arg5){//Add a Block
double x1=arg0,y1=arg1,z1=arg2,x2=arg3,y2=arg4,z2=arg5;
mat(x1,y2,z1,x1,y2,z2,x2,y2,z1);
mat(x2,y2,z2,x2,y2,z1,x1,y2,z2);
mat(x2,y1,z1,x1,y1,z2,x1,y1,z1);
mat(x2,y1,z2,x1,y1,z2,x2,y1,z1);
mat(x1,y1,z1,x1,y1,z2,x1,y2,z1);
mat(x1,y2,z1,x1,y1,z2,x1,y2,z2);
mat(x2,y1,z2,x2,y1,z1,x2,y2,z1);
mat(x2,y1,z2,x2,y2,z1,x2,y2,z2);
mat(x1,y1,z2,x2,y1,z2,x1,y2,z2);
mat(x1,y2,z2,x2,y1,z2,x2,y2,z2);
mat(x2,y1,z1,x1,y1,z1,x1,y2,z1);
mat(x2,y1,z1,x1,y2,z1,x2,y2,z1);
return temp_vector.size()-156;//12*13
}
export double apc(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6, double arg7){//Add a Cylinder
//double x1, double y1, double z1, double x2, double y2, double z2, double closed, double steps
double a,b,c,d,e,f,g,h,r,s,x1,y1,z1,x2,y2,z2,closed,steps;
unsigned int tlid=temp_vector.size();
x1=arg0;
y1=arg1;
z1=arg2;
x2=arg3;
y2=arg4;
z2=arg5;
closed=arg6;
steps=arg7;
a=x1+(x2-x1)/2;
b=y1+(y2-y1)/2;
r=FABS(x2-x1)/2;
s=FABS(y2-y1)/2;
if(closed==1){
   for(int i=2;i<steps;i+=1){
   c=lengthdir_x(r,0);
   d=lengthdir_y(s,0);
   e=lengthdir_x(r,i*360/steps);
   f=lengthdir_x(r,(i-1)*360/steps);
   g=lengthdir_y(s,i*360/steps);
   h=lengthdir_y(s,(i-1)*360/steps);
   mat(a+c,b+d,z1,  a+e,b+g,z1, a+f,b+h,z1 );
   mat(a+c,b+d,z2,  a+e,b+g,z2, a+f,b+h,z2 );
   }
}

for(int i=0;i<steps;i+=1){
    c=lengthdir_x(r,(i+1)*360/steps);
    d=lengthdir_y(s,(i+1)*360/steps);
    e=lengthdir_x(r,i*360/steps);
    f=lengthdir_y(s,i*360/steps);
    mat(a+e,b+f,z1, a+e,b+f,z2, a+c,b+d,z1 );
    mat(a+e,b+f,z2, a+c,b+d,z1, a+c,b+d,z2 );
}

return (double)tlid;
}
export double apo(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6, double arg7){//Add a Cone
//double x1, double y1, double z1, double x2, double y2, double z2, double closed, double steps
double a,b,c,d,e,f,g,h,r,s,x1,y1,z1,x2,y2,z2,closed,steps;
unsigned int tlid=temp_vector.size();
x1=arg0;
y1=arg1;
z1=arg2;
x2=arg3;
y2=arg4;
z2=arg5;
closed=arg6;
steps=arg7;
a=x1+(x2-x1)/2;
b=y1+(y2-y1)/2;
r=FABS(x2-x1)/2;
s=FABS(y2-y1)/2;
if(closed==1){
   for(int i=2;i<steps;i+=1){
   c=lengthdir_x(r,0);
   d=lengthdir_y(s,0);
   e=lengthdir_x(r,i*360/steps);
   f=lengthdir_x(r,(i-1)*360/steps);
   g=lengthdir_y(s,i*360/steps);
   h=lengthdir_y(s,(i-1)*360/steps);
   mat(a+c,b+d,z1,  a+e,b+g,z1, a+f,b+h,z1 );
   }
}

for(int i=0;i<steps;i+=1){
    c=lengthdir_x(r,(i+1)*360/steps);
    d=lengthdir_y(s,(i+1)*360/steps);
    e=lengthdir_x(r,i*360/steps);
    f=lengthdir_y(s,i*360/steps);
    mat(a+e,b+f,z1, a,b,z2, a+c,b+d,z1 );
}
return (double)tlid;
}
export double apm(char* arg0, double arg1,double arg2, double arg3){//Add An External Model (.d3d)
int countvertex=0;
unsigned int tlid=temp_vector.size();
//char *fname, double xv, double yv, double zv, Xrot,Yrot,Zrot (in RADIANS)
//ERRORS: 1=failed opening file, 2=wrong version, 3=not long enough file, 4=NULL FILE OPENED, 5=wrong data type
double xv=arg1;
double yv=arg2;
double zv=arg3;
double rotx=vecrotx;
double roty=vecroty;
double rotz=vecrotz;
Cx2 = cos(rotx);
Sx2 = -sin(rotx);
Cy2 = cos(roty);
Sy2 = -sin(roty);
Cz2 = cos(rotz);
Sz2 = -sin(rotz);
float r1=0,r2=0,r3=0,r4=0,r5=0,r6=0,r7=0,r8=0,r9=0,r10=0,x1=0,x2=0,x3=0,x4=0,x5=0,x6=0,x7=0,x8=0,x9=0,ax1=0.f,ax2=0.f,ay1=0.f,ay2=0.f,az1=0.f,az2=0.f;
long int r0=0,i=0,lines=0,type=0;

//Open the text file
FILE *f;
f = fopen(arg0,"r");
if(!f){
    return 1;
}
if(f==NULL){
    fclose(f);
    return 4;
}
//Read the first few numbers, get the total lines of data
fscanf( f, "%ld\r\n", &i );
if(i!=100){
    fclose(f);
    return 2;
}
//Not enough data in the file
fscanf( f, "%ld\r\n", &lines );
if(lines<1){
    fclose(f);
    return 3;
}


//loop through the whole model file, receive data about it
for(int count=0;count<lines;count++){
    //Read the next line of data
    fscanf( f, "%ld %f %f %f %f %f %f %f %f %f %f\r\n",&r0,&r1,&r2,&r3,&r4,&r5,&r6,&r7,&r8,&r9,&r10);

    //Primitive Begin
    if(r0==0){

            //Error, Unsupported (points+lines)
            if(r1==1 || r1==2 || r1==3){
			return 5;
            }

            //Triangle List
            if(r1==4){
                type=0;
				countvertex=0;
            }

            //Triangle Strip
            if(r1==5){
                type=1;
                fscanf( f, "%ld %f %f %f %f %f %f %f %f %f %f\r\n",&r0,&x1,&x2,&x3,&r4,&r5,&r6,&r7,&r8,&r9,&r10);
                fscanf( f, "%ld %f %f %f %f %f %f %f %f %f %f\r\n",&r0,&x4,&x5,&x6,&r4,&r5,&r6,&r7,&r8,&r9,&r10);
                fscanf( f, "%ld %f %f %f %f %f %f %f %f %f %f\r\n",&r0,&x7,&x8,&x9,&r4,&r5,&r6,&r7,&r8,&r9,&r10);
				rotatepoint_float(&x1,&x2,&x3);
				rotatepoint_float(&x4,&x5,&x6);
				rotatepoint_float(&x7,&x8,&x9);
                mat(x1+xv,x2+yv,x3+zv,x4+xv,x5+yv,x6+zv,x7+xv,x8+yv,x9+zv);
                count+=3;
            }

            //Triangle Fan
            if(r1==6){
                type=2;
                fscanf( f, "%ld %f %f %f %f %f %f %f %f %f %f\r\n",&r0,&x1,&x2,&x3,&r4,&r5,&r6,&r7,&r8,&r9,&r10);
                fscanf( f, "%ld %f %f %f %f %f %f %f %f %f %f\r\n",&r0,&x4,&x5,&x6,&r4,&r5,&r6,&r7,&r8,&r9,&r10);
                fscanf( f, "%ld %f %f %f %f %f %f %f %f %f %f\r\n",&r0,&x7,&x8,&x9,&r4,&r5,&r6,&r7,&r8,&r9,&r10);
				rotatepoint_float(&x1,&x2,&x3);
				rotatepoint_float(&x4,&x5,&x6);
				rotatepoint_float(&x7,&x8,&x9);
                mat(x1+xv,x2+yv,x3+zv,x4+xv,x5+yv,x6+zv,x7+xv,x8+yv,x9+zv);
                count+=3;
            }
    }
    //Primitive End
    if(r0==1){
    //Do nothing, everything is done in primitive begin
    }

    //Add a vertex
    if(r0==9 || r0==8 || r0==7 || r0==6 || r0==5 || r0==4 || r0==3 || r0==2){

        //Triangle List
        if(type==0){
			switch(countvertex){
				case(0):
				ax1=r1;
				ay1=r2;
				az1=r3;
				break;
				case(1):
				ax2=r1;
				ay2=r2;
				az2=r3;
				break;
				case(2):
				countvertex=-1;
				rotatepoint_float(&ax1,&ay1,&az1);
				rotatepoint_float(&ax2,&ay2,&az2);
				rotatepoint_float(&r1,&r2,&r3);
				mat(ax1+xv,ay1+yv,az1+zv,ax2+xv,ay2+yv,az2+zv,r1+xv,r2+yv,r3+zv);
				break;
			}
		countvertex++;
        }

        //Triangle Strip
        if(type==1){
        x1=x4;
        x2=x5;
        x3=x6;
        x4=x7;
        x5=x8;
        x6=x9;
        x7=r1;
        x8=r2;
        x9=r3;
		rotatepoint_float(&x7,&x8,&x9);
        mat(x1+xv,x2+yv,x3+zv,x4+xv,x5+yv,x6+zv,x7+xv,x8+yv,x9+zv);
        }

        //Triangle Fan
        if(type==2){
        x4=x7;
        x5=x8;
        x6=x9;
        x7=r1;
        x8=r2;
        x9=r3;
		rotatepoint_float(&x7,&x8,&x9);
        mat(x1+xv,x2+yv,x3+zv,x4+xv,x5+yv,x6+zv,x7+xv,x8+yv,x9+zv);
        }

    }

    //Add a block
    if(r0==10){
	int_apb(r1+xv,r2+yv,r3+zv,r4+xv,r5+yv,r6+zv);
    }

    //Add a cylinder
    if(r0==11){
    int_apc(r1+xv,r2+yv,r3+zv,r4+xv,r5+yv,r6+zv,r9,r10);
    }

    //Add a cone
    if(r0==12){
    int_apo(r1+xv,r2+yv,r3+zv,r4+xv,r5+yv,r6+zv,r9,r10);
    }

    //Add a wall
    if(r0==14){
    int_apw(r1+xv,r2+yv,r3+zv,r4+xv,r5+yv,r6+zv);
    }

    //Add a floor
    if(r0==15){
    int_apf(r1+xv,r2+yv,r3+zv,r4+xv,r5+yv,r6+zv);
    }
}


return (double)tlid;
}


//************************************************************************
//Overwriting Functions                                                  *
//************************************************************************
export double obd(double arg0, double arg1){//Begin Overwriting
ow_mod=(unsigned int)arg0;
ow_tri=(unsigned int)arg1;
return 1;
}
export double oed(){//End Overwriting
ow_mod=0;
ow_tri=0;
return 1;
}
export double opt(double arg0, double arg1,double arg2, double arg3,double arg4,
                           double arg5, double arg6, double arg7,double arg8){//Overwrite a single triangle
G_modellist[ow_mod][ow_tri] = arg0;
G_modellist[ow_mod][ow_tri+1] = arg1;
G_modellist[ow_mod][ow_tri+2] = arg2;
G_modellist[ow_mod][ow_tri+3] = arg3;
G_modellist[ow_mod][ow_tri+4] = arg4;
G_modellist[ow_mod][ow_tri+5] = arg5;
G_modellist[ow_mod][ow_tri+6] = arg6;
G_modellist[ow_mod][ow_tri+7] = arg7;
G_modellist[ow_mod][ow_tri+8] = arg8;
//correct the normals
calc_normals(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8);
G_modellist[ow_mod][ow_tri+9] = normalx;
G_modellist[ow_mod][ow_tri+10] = normaly;
G_modellist[ow_mod][ow_tri+11] = normalz;
G_modellist[ow_mod][ow_tri+12] = addingtriangleid;//triangleid
ow_tri+=13;
return ow_tri;
}
export double opb(double arg0, double arg1,double arg2, double arg3,double arg4, double arg5){//Overwrite a Block
double x1=arg0,y1=arg1,z1=arg2,x2=arg3,y2=arg4,z2=arg5;
opt(x1,y2,z1,x1,y2,z2,x2,y2,z1);
opt(x2,y2,z2,x2,y2,z1,x1,y2,z2);
opt(x2,y1,z1,x1,y1,z2,x1,y1,z1);
opt(x2,y1,z2,x1,y1,z2,x2,y1,z1);
opt(x1,y1,z1,x1,y1,z2,x1,y2,z1);
opt(x1,y2,z1,x1,y1,z2,x1,y2,z2);
opt(x2,y1,z2,x2,y1,z1,x2,y2,z1);
opt(x2,y1,z2,x2,y2,z1,x2,y2,z2);
opt(x1,y1,z2,x2,y1,z2,x1,y2,z2);
opt(x1,y2,z2,x2,y1,z2,x2,y2,z2);
opt(x2,y1,z1,x1,y1,z1,x1,y2,z1);
opt(x2,y1,z1,x1,y2,z1,x2,y2,z1);
return (double)ow_tri;
}
export double opw(double arg0, double arg1,double arg2, double arg3,double arg4, double arg5){//Add a Wall
//double x1, double y1, double z1, double x2, double y2, double z2
double x1=arg0,y1=arg1,z1=arg2,x2=arg3,y2=arg4,z2=arg5;
opt(x1,y1,z1,x2,y2,z1,x2,y2,z2);
opt(x2,y2,z2,x1,y1,z1,x1,y1,z2);
return (double)ow_tri;
}
export double opf(double arg0, double arg1,double arg2, double arg3,double arg4, double arg5){//Add a Floor
double x1=arg0,y1=arg1,z1=arg2,x2=arg3,y2=arg4,z2=arg5;
opt(x1,y1,z1, x2,y1,z2, x2,y2,z2);
opt(x2,y2,z2, x1,y1,z1, x1,y2,z1);
return (double)ow_tri;
}
export double opc(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6, double arg7){//Add a Cylinder
//double x1, double y1, double z1, double x2, double y2, double z2, double closed, double steps
double a,b,c,d,e,f,g,h,r,s,x1,y1,z1,x2,y2,z2,closed,steps;
unsigned int tlid=temp_vector.size();
x1=arg0;
y1=arg1;
z1=arg2;
x2=arg3;
y2=arg4;
z2=arg5;
closed=arg6;
steps=arg7;
a=x1+(x2-x1)/2;
b=y1+(y2-y1)/2;
r=FABS(x2-x1)/2;
s=FABS(y2-y1)/2;
if(closed==1){
   for(int i=2;i<steps;i+=1){
   c=lengthdir_x(r,0);
   d=lengthdir_y(s,0);
   e=lengthdir_x(r,i*360/steps);
   f=lengthdir_x(r,(i-1)*360/steps);
   g=lengthdir_y(s,i*360/steps);
   h=lengthdir_y(s,(i-1)*360/steps);
   opt(a+c,b+d,z1,  a+e,b+g,z1, a+f,b+h,z1 );
   opt(a+c,b+d,z2,  a+e,b+g,z2, a+f,b+h,z2 );
   }
}

for(int i=0;i<steps;i+=1){
    c=lengthdir_x(r,(i+1)*360/steps);
    d=lengthdir_y(s,(i+1)*360/steps);
    e=lengthdir_x(r,i*360/steps);
    f=lengthdir_y(s,i*360/steps);
    opt(a+e,b+f,z1, a+e,b+f,z2, a+c,b+d,z1 );
    opt(a+e,b+f,z2, a+c,b+d,z1, a+c,b+d,z2 );
}

return (double)tlid;
}
export double opo(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6, double arg7){//Add a Cone
//double x1, double y1, double z1, double x2, double y2, double z2, double closed, double steps
double a,b,c,d,e,f,g,h,r,s,x1,y1,z1,x2,y2,z2,closed,steps;
unsigned int tlid=temp_vector.size();
x1=arg0;
y1=arg1;
z1=arg2;
x2=arg3;
y2=arg4;
z2=arg5;
closed=arg6;
steps=arg7;
a=x1+(x2-x1)/2;
b=y1+(y2-y1)/2;
r=FABS(x2-x1)/2;
s=FABS(y2-y1)/2;
if(closed==1){
   for(int i=2;i<steps;i+=1){
   c=lengthdir_x(r,0);
   d=lengthdir_y(s,0);
   e=lengthdir_x(r,i*360/steps);
   f=lengthdir_x(r,(i-1)*360/steps);
   g=lengthdir_y(s,i*360/steps);
   h=lengthdir_y(s,(i-1)*360/steps);
   opt(a+c,b+d,z1,  a+e,b+g,z1, a+f,b+h,z1 );
   }
}

for(int i=0;i<steps;i+=1){
    c=lengthdir_x(r,(i+1)*360/steps);
    d=lengthdir_y(s,(i+1)*360/steps);
    e=lengthdir_x(r,i*360/steps);
    f=lengthdir_y(s,i*360/steps);
    opt(a+e,b+f,z1, a,b,z2, a+c,b+d,z1 );
}
return (double)tlid;
}
//************************************************************************
//Collision Checking (MC=collision, MR=raycast)                          *
//************************************************************************
export double mcs(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6, double arg7){//Model Check
int mod1=(int)arg0,mod2=(int)arg4;
double mod1_xpos=arg1-arg5,mod1_ypos=arg2-arg6,mod1_zpos=arg3-arg7,
t1p1[3],t1p2[3],t1p3[3],t2p1[3],t2p2[3],t2p3[3];
        for(unsigned int ML1=0;  ML1<G_modellist[mod1].size();  ML1+=13){
        t1p1[0]=mod1_xpos+G_modellist[mod1][ML1];
        t1p1[1]=mod1_ypos+G_modellist[mod1][ML1+1];
        t1p1[2]=mod1_zpos+G_modellist[mod1][ML1+2];
        t1p2[0]=mod1_xpos+G_modellist[mod1][ML1+3];
        t1p2[1]=mod1_ypos+G_modellist[mod1][ML1+4];
        t1p2[2]=mod1_zpos+G_modellist[mod1][ML1+5];
        t1p3[0]=mod1_xpos+G_modellist[mod1][ML1+6];
        t1p3[1]=mod1_ypos+G_modellist[mod1][ML1+7];
        t1p3[2]=mod1_zpos+G_modellist[mod1][ML1+8];
            for(unsigned int ML2=0;  ML2<G_modellist[mod2].size();  ML2+=13){
            t2p1[0]=G_modellist[mod2][ML2];
            t2p1[1]=G_modellist[mod2][ML2+1];
            t2p1[2]=G_modellist[mod2][ML2+2];
            t2p2[0]=G_modellist[mod2][ML2+3];
            t2p2[1]=G_modellist[mod2][ML2+4];
            t2p2[2]=G_modellist[mod2][ML2+5];
            t2p3[0]=G_modellist[mod2][ML2+6];
            t2p3[1]=G_modellist[mod2][ML2+7];
            t2p3[2]=G_modellist[mod2][ML2+8];
            if(n_t_i(t1p1,t1p2,t1p3,t2p1,t2p2,t2p3)){
                return 1;
            }
        }
    }
return 0;
}
//Model Check Rotation (Modify model1 pos by difference in movement) - faster than actually moving both
export double mcr(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6, double arg7,
         double arg8, double arg9,double arg10, double arg11, double arg12, double arg13){
int mod1=(int)arg0,mod2=(int)arg4;
double mod1_xpos=arg1-arg5, mod1_ypos=arg2-arg6, mod1_zpos=arg3-arg7, rotx1=arg8, roty1=arg9,
rotz1=arg10, rotx2=vecrotx, roty2=vecroty, rotz2=vecrotz,t1p1[3],t1p2[3],t1p3[3],t2p1[3],t2p2[3],t2p3[3];

Cx = cos(rotx1);
Sx = -sin(rotx1);
Cy = cos(roty1);
Sy = -sin(roty1);
Cz = cos(rotz1);
Sz = -sin(rotz1);
Cx2 = cos(rotx2);
Sx2 = -sin(rotx2);
Cy2 = cos(roty2);
Sy2 = -sin(roty2);
Cz2 = cos(rotz2);
Sz2 = -sin(rotz2);
    for(unsigned int ML1=0;  ML1<G_modellist[mod1].size();  ML1+=13){
    t1p1[0]=mod1_xpos+G_modellist[mod1][ML1];
    t1p1[1]=mod1_ypos+G_modellist[mod1][ML1+1];
    t1p1[2]=mod1_zpos+G_modellist[mod1][ML1+2];
    rotatepoint(&t1p1[0],&t1p1[1],&t1p1[2],mod1_xpos,mod1_ypos,mod1_zpos);
    t1p2[0]=mod1_xpos+G_modellist[mod1][ML1+3];
    t1p2[1]=mod1_ypos+G_modellist[mod1][ML1+4];
    t1p2[2]=mod1_zpos+G_modellist[mod1][ML1+5];
    rotatepoint(&t1p2[0],&t1p2[1],&t1p2[2],mod1_xpos,mod1_ypos,mod1_zpos);
    t1p3[0]=mod1_xpos+G_modellist[mod1][ML1+6];
    t1p3[1]=mod1_ypos+G_modellist[mod1][ML1+7];
    t1p3[2]=mod1_zpos+G_modellist[mod1][ML1+8];
    rotatepoint(&t1p3[0],&t1p3[1],&t1p3[2],mod1_xpos,mod1_ypos,mod1_zpos);
        for(unsigned int ML2=0;  ML2<G_modellist[mod2].size();  ML2+=13){
        t2p1[0]=G_modellist[mod2][ML2];
        t2p1[1]=G_modellist[mod2][ML2+1];
        t2p1[2]=G_modellist[mod2][ML2+2];
        rotatepoint_faster(&t2p1[0],&t2p1[1],&t2p1[2]);
        t2p2[0]=G_modellist[mod2][ML2+3];
        t2p2[1]=G_modellist[mod2][ML2+4];
        t2p2[2]=G_modellist[mod2][ML2+5];
        rotatepoint_faster(&t2p2[0],&t2p2[1],&t2p2[2]);
        t2p3[0]=G_modellist[mod2][ML2+6];
        t2p3[1]=G_modellist[mod2][ML2+7];
        t2p3[2]=G_modellist[mod2][ML2+8];
        rotatepoint_faster(&t2p3[0],&t2p3[1],&t2p3[2]);
            if(n_t_i(t1p1,t1p2,t1p3,t2p1,t2p2,t2p3)){
            return 1;
            }
        }
    }
return 0;
}
//Model Check Split
export double mc3(double arg0, double arg1,double arg2, double arg3){
//double xregion, double yregion, double mod1, double mod1_xpos, double mod1_ypos, double mod1_zpos
int xr=(int)(arg1/splitregionx),yr=(int)(arg2/splitregionx),mod1=(int)arg0,mod2=p3dc_splitmodels[xr][yr];
double mod1_xpos=arg1,mod1_ypos=arg2,mod1_zpos=arg3,t1p1[3],t1p2[3],t1p3[3],t2p1[3],t2p2[3],t2p3[3];
    for(unsigned int ML1=0;  ML1<G_modellist[mod1].size();  ML1+=13){
    t1p1[0]=mod1_xpos+G_modellist[mod1][ML1];
    t1p1[1]=mod1_ypos+G_modellist[mod1][ML1+1];
    t1p1[2]=mod1_zpos+G_modellist[mod1][ML1+2];
    t1p2[0]=mod1_xpos+G_modellist[mod1][ML1+3];
    t1p2[1]=mod1_ypos+G_modellist[mod1][ML1+4];
    t1p2[2]=mod1_zpos+G_modellist[mod1][ML1+5];
    t1p3[0]=mod1_xpos+G_modellist[mod1][ML1+6];
    t1p3[1]=mod1_ypos+G_modellist[mod1][ML1+7];
    t1p3[2]=mod1_zpos+G_modellist[mod1][ML1+8];
        for(unsigned int ML2=0;  ML2<G_modellist[mod2].size();  ML2+=13){
        t2p1[0]=G_modellist[mod2][ML2];
        t2p1[1]=G_modellist[mod2][ML2+1];
        t2p1[2]=G_modellist[mod2][ML2+2];
        t2p2[0]=G_modellist[mod2][ML2+3];
        t2p2[1]=G_modellist[mod2][ML2+4];
        t2p2[2]=G_modellist[mod2][ML2+5];
        t2p3[0]=G_modellist[mod2][ML2+6];
        t2p3[1]=G_modellist[mod2][ML2+7];
        t2p3[2]=G_modellist[mod2][ML2+8];
            if(n_t_i(t1p1,t1p2,t1p3,t2p1,t2p2,t2p3)){
                return 1;
            }
        }
    }
return 0;
}
//Model Ray
export double mrs(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6, double arg7, double arg8, double arg9){

int mod1=(int)arg0;
double xorig=arg4-arg1,yorig=arg5-arg2,zorig=arg6-arg3,xdir=arg7,ydir=arg8,zdir=arg9,
t_1_p_1[3],t_1_p_2[3],t_1_p_3[3],t=0,dist=10000000,orig[3]={xorig,yorig,zorig},dir[3]={xdir,ydir,zdir};
tridata[12]=0;
    for(unsigned int ML1=0;  ML1<G_modellist[mod1].size();  ML1+=13){
        t_1_p_1[0]=G_modellist[mod1][ML1];
        t_1_p_1[1]=G_modellist[mod1][ML1+1];
        t_1_p_1[2]=G_modellist[mod1][ML1+2];
        t_1_p_2[0]=G_modellist[mod1][ML1+3];
        t_1_p_2[1]=G_modellist[mod1][ML1+4];
        t_1_p_2[2]=G_modellist[mod1][ML1+5];
        t_1_p_3[0]=G_modellist[mod1][ML1+6];
        t_1_p_3[1]=G_modellist[mod1][ML1+7];
        t_1_p_3[2]=G_modellist[mod1][ML1+8];
        if(i_r_t(orig,dir,t_1_p_1,t_1_p_2,t_1_p_3,&t)){
            if(t>0 && t<dist){
            dist=t;
            tridata[12]=ML1;
            }
        }
    }
unsigned int i = (int)tridata[12];
tridata[0]=G_modellist[mod1][i];
tridata[1]=G_modellist[mod1][i+1];
tridata[2]=G_modellist[mod1][i+2];
tridata[3]=G_modellist[mod1][i+3];
tridata[4]=G_modellist[mod1][i+4];
tridata[5]=G_modellist[mod1][i+5];
tridata[6]=G_modellist[mod1][i+6];
tridata[7]=G_modellist[mod1][i+7];
tridata[8]=G_modellist[mod1][i+8];
tridata[9]=G_modellist[mod1][i+9];
tridata[10]=G_modellist[mod1][i+10];
tridata[11]=G_modellist[mod1][i+11];
trianglehit=G_modellist[mod1][i+12];
return dist;
}
//Model Ray First (No moving, much faster, first hit is returned)
export double mrf(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6){
int mod1=(int)arg0;
double xorig=arg1,yorig=arg2,zorig=arg3,xdir=arg4,ydir=arg5,zdir=arg6, t_1_p_1[3],t_1_p_2[3],t_1_p_3[3],t=0,orig[3]={xorig,yorig,zorig},dir[3]={xdir,ydir,zdir};
    for(unsigned int ML1=0;  ML1<G_modellist[mod1].size();  ML1+=13){
        t_1_p_1[0]=G_modellist[mod1][ML1];
        t_1_p_1[1]=G_modellist[mod1][ML1+1];
        t_1_p_1[2]=G_modellist[mod1][ML1+2];
        t_1_p_2[0]=G_modellist[mod1][ML1+3];
        t_1_p_2[1]=G_modellist[mod1][ML1+4];
        t_1_p_2[2]=G_modellist[mod1][ML1+5];
        t_1_p_3[0]=G_modellist[mod1][ML1+6];
        t_1_p_3[1]=G_modellist[mod1][ML1+7];
        t_1_p_3[2]=G_modellist[mod1][ML1+8];
        if(i_r_t(orig,dir,t_1_p_1,t_1_p_2,t_1_p_3,&t)){
            if(t>0){
            tridata[0]=G_modellist[mod1][ML1];
            tridata[1]=G_modellist[mod1][ML1+1];
            tridata[2]=G_modellist[mod1][ML1+2];
            tridata[3]=G_modellist[mod1][ML1+3];
            tridata[4]=G_modellist[mod1][ML1+4];
            tridata[5]=G_modellist[mod1][ML1+5];
            tridata[6]=G_modellist[mod1][ML1+6];
            tridata[7]=G_modellist[mod1][ML1+7];
            tridata[8]=G_modellist[mod1][ML1+8];
            tridata[9]=G_modellist[mod1][ML1+9];
            tridata[10]=G_modellist[mod1][ML1+10];
            tridata[11]=G_modellist[mod1][ML1+11];
            trianglehit=G_modellist[mod1][ML1+12];
            return t;
            }
        }
    }
return 10000000;
}
//Model Ray Split
export double mr3(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5){
double xr=arg0/splitregionx,yr=arg1/splitregionx,xorig=arg0,yorig=arg1,zorig=arg2,xdir=arg3,ydir=arg4,zdir=arg5,
t_1_p_1[3],t_1_p_2[3],t_1_p_3[3],t=0,orig[3]={xorig,yorig,zorig},dir[3]={xdir,ydir,zdir},dist=10000000;
int mod1=p3dc_splitmodels[(int)xr][(int)yr];
tridata[12]=0;
        for(unsigned int ML1=0;  ML1<G_modellist[mod1].size();  ML1+=13){
            t_1_p_1[0]=G_modellist[mod1][ML1];
            t_1_p_1[1]=G_modellist[mod1][ML1+1];
            t_1_p_1[2]=G_modellist[mod1][ML1+2];
            t_1_p_2[0]=G_modellist[mod1][ML1+3];
            t_1_p_2[1]=G_modellist[mod1][ML1+4];
            t_1_p_2[2]=G_modellist[mod1][ML1+5];
            t_1_p_3[0]=G_modellist[mod1][ML1+6];
            t_1_p_3[1]=G_modellist[mod1][ML1+7];
            t_1_p_3[2]=G_modellist[mod1][ML1+8];
            if(i_r_t(orig,dir,t_1_p_1,t_1_p_2,t_1_p_3,&t)){
				if(t>0 && t<dist){
				dist=t;
				tridata[12]=ML1;
				}
            }
        }

unsigned int i = (int)tridata[12];
tridata[0]=G_modellist[mod1][i];
tridata[1]=G_modellist[mod1][i+1];
tridata[2]=G_modellist[mod1][i+2];
tridata[3]=G_modellist[mod1][i+3];
tridata[4]=G_modellist[mod1][i+4];
tridata[5]=G_modellist[mod1][i+5];
tridata[6]=G_modellist[mod1][i+6];
tridata[7]=G_modellist[mod1][i+7];
tridata[8]=G_modellist[mod1][i+8];
tridata[9]=G_modellist[mod1][i+9];
tridata[10]=G_modellist[mod1][i+10];
tridata[11]=G_modellist[mod1][i+11];
trianglehit=G_modellist[mod1][i+12];
return dist;
}
//Model Ray Rotation (Modify ray position if model moves),rotx, roty, rotz are model rotations, origins are ray origins
export double mrr(double arg0, double arg1,double arg2, double arg3, double arg4, double arg5, double arg6, double arg7, double arg8, double arg9){
	//double mod1, double xorig, double yorig, double zorig, double xdir, double ydir, double zdir, double rotx, double roty, double rotz
int mod1=(int)arg0;
double xorig=arg4-arg1, yorig=arg5-arg2, zorig=arg6-arg3, xdir=arg7, ydir=arg8, zdir=arg9, rotx=vecrotx, roty=vecroty, rotz=vecrotz,
t_1_p_1[3],t_1_p_2[3],t_1_p_3[3],t=0,dist=10000000,orig[3]={xorig,yorig,zorig},dir[3]={xdir,ydir,zdir};
Cx2 = cos(rotx);
Sx2 = -sin(rotx);
Cy2 = cos(roty);
Sy2 = -sin(roty);
Cz2 = cos(rotz);
Sz2 = -sin(rotz);
tridata[12]=0;
    for(unsigned int ML1=0;  ML1<G_modellist[mod1].size(); ML1+=13){
        t_1_p_1[0]=G_modellist[mod1][ML1];
        t_1_p_1[1]=G_modellist[mod1][ML1+1];
        t_1_p_1[2]=G_modellist[mod1][ML1+2];
        t_1_p_2[0]=G_modellist[mod1][ML1+3];
        t_1_p_2[1]=G_modellist[mod1][ML1+4];
        t_1_p_2[2]=G_modellist[mod1][ML1+5];
        t_1_p_3[0]=G_modellist[mod1][ML1+6];
        t_1_p_3[1]=G_modellist[mod1][ML1+7];
        t_1_p_3[2]=G_modellist[mod1][ML1+8];
        rotatepoint_faster(&t_1_p_1[0],&t_1_p_1[1],&t_1_p_1[2]);
        rotatepoint_faster(&t_1_p_2[0],&t_1_p_2[1],&t_1_p_2[2]);
        rotatepoint_faster(&t_1_p_3[0],&t_1_p_3[1],&t_1_p_3[2]);
        if(i_r_t(orig,dir,t_1_p_1,t_1_p_2,t_1_p_3,&t)){
            if(t<dist && t>0){
            dist=t;
            tridata[12]=ML1;
            }
        }
    }
int td12=(int)tridata[12];
tridata[0]=G_modellist[mod1][td12];
tridata[1]=G_modellist[mod1][td12+1];
tridata[2]=G_modellist[mod1][td12+2];
tridata[3]=G_modellist[mod1][td12+3];
tridata[4]=G_modellist[mod1][td12+4];
tridata[5]=G_modellist[mod1][td12+5];
tridata[6]=G_modellist[mod1][td12+6];
tridata[7]=G_modellist[mod1][td12+7];
tridata[8]=G_modellist[mod1][td12+8];
rotatepoint_faster(&tridata[0],&tridata[1],&tridata[2]);
rotatepoint_faster(&tridata[3],&tridata[4],&tridata[5]);
rotatepoint_faster(&tridata[6],&tridata[7],&tridata[8]);

calc_normals(tridata[0],tridata[1],tridata[2],tridata[3],tridata[4],tridata[5],tridata[6],tridata[7],tridata[8]);

tridata[9]=normalx;
tridata[10]=normaly;
tridata[11]=normalz;
trianglehit=G_modellist[mod1][td12+12];
return dist;
}
//Set model rotation
export double smr(double arg0, double arg1, double arg2){
    vecrotx=arg0;
    vecroty=arg1;
    vecrotz=arg2;
    return 1;
}

BOOL APIENTRY DllMain(HINSTANCE aInstanceHandle, int aReason, int aReserved ) {
	switch(aReason){
        case DLL_PROCESS_ATTACH:
        addingtriangleid=-1;
        trianglehit=-1;
        break;
        case DLL_PROCESS_DETACH:
        G_modellist.clear();
        temp_vector.clear();
        break;
    }
	return TRUE;
}

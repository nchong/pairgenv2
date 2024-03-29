/*
 * Pairwise interaction of particles i and j
 *
 * Read-only per-particle:
 * 	xi, xj		(* position *)
 * 	vi, vj
 * 	omegai, omegaj
 * 	radiusi, radiusj
 * 	massi, massj
 * 	typei, typej
 * Read-write per-particle:
 * 	--
 * Sum-into per-particle:
 * 	forcei_delta
 * 	torquei_delta
 *
 * Read-only per-neighbor:
 * 	--
 * Read-write per-neighbor:
 * 	shear
 * 	touch
 *
 * Constants:
 * 	D_DT		(* timestep *)
 * 	D_NKTV2P
 * 	D_YEFF
 * 	D_GEFF
 * 	D_BETAEFF
 * 	D_COEFFFRICT
 */

#define sqrtFiveOverSix 0.91287092917527685576161630466800355658790782499663875

#ifdef TRACE
#if defined(__CUDACC__)
  #include "cuPrintf.cu"
  #define PRINT cuPrintf
#else
  #include <cstdio>
  #define PRINT printf
#endif

#define QUOTE(str) #str
#define ITRACE(x) PRINT("%s\t%d\n", QUOTE(x), x)
#define DTRACE(x) PRINT("%s\t%.16f\n", QUOTE(x), x)
#endif

#if defined(__CUDACC__)
  __device__
#else
  inline
#endif
void hertz_pair_kernel(
#ifdef TRACE
    int i, int j,
#endif
    double xi[3], double xj[3], 
    double vi[3], double vj[3], 
    double omegai[3], double omegaj[3], 
    double radiusi, double radiusj, 
    double massi, double massj, 
    int typei, int typej,
    double forcei_delta[3], 
    double torquei_delta[3], 
    double shear[3], 
    int *touch) {
  // del is the vector from j to i
  double delx = xi[0] - xj[0];
  double dely = xi[1] - xj[1];
  double delz = xi[2] - xj[2];

  double rsq = delx*delx + dely*dely + delz*delz;
  double radsum = radiusi + radiusj;
  if (rsq >= radsum*radsum) {
    //unset non-touching atoms
    *touch = 0;
    shear[0] = 0.0;
    shear[1] = 0.0;
    shear[2] = 0.0;
  } else {
    //distance between centres of atoms i and j
    //or, magnitude of del vector
    double r = sqrt(rsq);
    double rinv = 1.0/r;
    double rsqinv = 1.0/rsq;

    // relative translational velocity
    double vr1 = vi[0] - vj[0];
    double vr2 = vi[1] - vj[1];
    double vr3 = vi[2] - vj[2];

    // normal component
    double vnnr = vr1*delx + vr2*dely + vr3*delz;
    double vn1 = delx*vnnr * rsqinv;
    double vn2 = dely*vnnr * rsqinv;
    double vn3 = delz*vnnr * rsqinv;

    // tangential component
    double vt1 = vr1 - vn1;
    double vt2 = vr2 - vn2;
    double vt3 = vr3 - vn3;

    // relative rotational velocity
    double deltan = radsum-r;
    double cri = radiusi-0.5*deltan;
    double crj = radiusj-0.5*deltan;
    double wr1 = (cri*omegai[0] + crj*omegaj[0]) * rinv;
    double wr2 = (cri*omegai[1] + crj*omegaj[1]) * rinv;
    double wr3 = (cri*omegai[2] + crj*omegaj[2]) * rinv;

    // normal forces = Hookian contact + normal velocity damping
    double meff = massi*massj/(massi+massj);
    //not-implemented: freeze_group_bit

    //derive contact model parameters (inlined)
    //Yeff, Geff, betaeff, coeffFrict are lookup tables
    double reff = radiusi * radiusj / (radiusi + radiusj);
    double sqrtval = sqrt(reff * deltan);
    double Sn = 2.    * D_YEFF * sqrtval;
    double St = 8.    * D_GEFF * sqrtval;
    double kn = 4./3. * D_YEFF * sqrtval;
    double kt = St;
    double gamman=-2.*sqrtFiveOverSix*D_BETAEFF*sqrt(Sn*meff);
    double gammat=-2.*sqrtFiveOverSix*D_BETAEFF*sqrt(St*meff);
    double xmu=D_COEFFFRICT;
    //not-implemented if (dampflag == 0) gammat = 0;
    kn /= D_NKTV2P;
    kt /= D_NKTV2P;

    double damp = gamman*vnnr*rsqinv;
	  double ccel = kn*(radsum-r)*rinv - damp;

    //not-implemented cohesionflag

    // relative velocities
    double vtr1 = vt1 - (delz*wr2-dely*wr3);
    double vtr2 = vt2 - (delx*wr3-delz*wr1);
    double vtr3 = vt3 - (dely*wr1-delx*wr2);

    // shear history effects
    shear[0] += vtr1 * D_DT;
    shear[1] += vtr2 * D_DT;
    shear[2] += vtr3 * D_DT;

    // rotate shear displacements
    double rsht = shear[0]*delx + shear[1]*dely + shear[2]*delz;
    rsht *= rsqinv;

    shear[0] -= rsht*delx;
    shear[1] -= rsht*dely;
    shear[2] -= rsht*delz;

    // tangential forces = shear + tangential velocity damping
    double fs1 = - (kt*shear[0]);
    double fs2 = - (kt*shear[1]);
    double fs3 = - (kt*shear[2]);

    // rescale frictional displacements and forces if needed
    double fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
    double fn = xmu * fabs((double)ccel*r);
    double shrmag = sqrt(shear[0]*shear[0] +
                         shear[1]*shear[1] +
                         shear[2]*shear[2]);
    if (fs > fn) {
      if (shrmag != 0.0) {
        fs1 *= fn/fs;
        fs2 *= fn/fs;
        fs3 *= fn/fs;
        shear[0] = -fs1/kt;
        shear[1] = -fs2/kt;
        shear[2] = -fs3/kt;
      } else {
        fs1 = 0.0;
        fs2 = 0.0;
        fs3 = 0.0;
      }
    } else {
      fs1 -= (gammat*vtr1);
      fs2 -= (gammat*vtr2);
      fs3 -= (gammat*vtr3);
    }

    double fx = delx*ccel + fs1;
    double fy = dely*ccel + fs2;
    double fz = delz*ccel + fs3;

    double tor1 = rinv * (dely*fs3 - delz*fs2);
    double tor2 = rinv * (delz*fs1 - delx*fs3);
    double tor3 = rinv * (delx*fs2 - dely*fs1);

    // this is what we've been working up to!
    forcei_delta[0] += fx;
    forcei_delta[1] += fy;
    forcei_delta[2] += fz;

    torquei_delta[0] -= cri*tor1;
    torquei_delta[1] -= cri*tor2;
    torquei_delta[2] -= cri*tor3;

#ifdef TRACE
    if (i == TRACE) {
      ITRACE(i); ITRACE(j);
      DTRACE(fx); DTRACE(fy); DTRACE(fz);
    }
#endif
  }
}

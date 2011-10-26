#include "framework.h"
#include "hertz_pair_kernel.h"

using namespace std;

void run(struct params *input, int num_iter) {
  NeighListLike *nl = new NeighListLike(input);

  //setup constants
  D_DT = input->dt;
  D_NKTV2P = input->nktv2p;
  D_YEFF = input->yeff[3];
  D_GEFF = input->geff[3];
  D_BETAEFF = input->betaeff[3];
  D_COEFFFRICT = input->coeffFrict[3];

  //internal copies of outputs
  double *force = new double[input->nnode*3];
  double *torque = new double[input->nnode*3];
  double **firstdouble = NULL;
  double **dpages = NULL;
  int    **firsttouch = NULL;
  int    **tpages = NULL;

  per_iter.push_back(SimpleTimer("run"));
  for (int run=0; run<num_iter; run++) {
    //make copies
    copy(input->force,  input->force  + input->nnode*3, force);
    copy(input->torque, input->torque + input->nnode*3, torque);
    nl->copy_into(firstdouble, dpages, firsttouch, tpages);

    per_iter[0].start();
    for (int ii=0; ii<nl->inum; ii++) {
      int i = nl->ilist[ii];
      for (int jj=0; jj<nl->numneigh[i]; jj++) {
        int j = nl->firstneigh[i][jj];
        double *shear = &(firstdouble[i][3*jj]);
        int *touch = &(firsttouch[i][jj]);
      
        hertz_pair_kernel(
          &input->x[(i*3)],     &input->x[(j*3)],
          &input->v[(i*3)],     &input->v[(j*3)],
          &input->omega[(i*3)], &input->omega[(j*3)],
           input->radius[i],     input->radius[j],
           input->mass[i],       input->mass[j],
           input->type[i],       input->type[j],
          &force[(i*3)],
          &torque[(i*3)],
           shear, touch
        );
      }
    }
    per_iter[0].stop_and_add_to_total();

    if (run == 0) {
      // check results
    }
  }

  delete[] force;
  delete[] torque;
  delete[] firstdouble;
  delete[] dpages;
  delete[] firsttouch;
  delete[] tpages;
}

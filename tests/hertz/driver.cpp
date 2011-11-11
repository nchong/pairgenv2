#include "framework.h"
#include "hertz_wrapper.h"

#ifndef BLOCK_SIZE
#error "You need to #define BLOCK_SIZE"
#endif
#ifndef KERNEL
#error "You need to #define KERNEL TPA|BPA"
#endif

using namespace std;

void run(struct params *input, int num_iter) {
  //cout << clinfo();
  CLWrapper clw(/*platform=*/0,/*device=*/0,/*profiling=*/false);
  NeighListLike *nl = new NeighListLike(input);

  one_time.push_back(SimpleTimer("initialization"));
  one_time.back().start();
  size_t wx = BLOCK_SIZE;
  HertzWrapper *hw = new HertzWrapper(
    clw, wx,
    input->nnode, nl->maxpage, nl->pgsize,
    //constants
    input->dt,
    input->nktv2p,
    input->yeff[3],
    input->geff[3],
    input->betaeff[3],
    input->coeffFrict[3],
    //non-reload parameters
    input->radius, input->mass, input->type
  );
  one_time.back().stop_and_add_to_total();

  one_time.push_back(SimpleTimer("neighlist refill"));
  one_time.back().start();
  hw->refill_neighlist(
    nl->numneigh,
    nl->firstneigh,
    nl->pages,
    nl->maxpage,
    nl->firstdouble,
    nl->firsttouch);
  one_time.back().stop_and_add_to_total();

  //internal copies of outputs
  double *force = new double[input->nnode*3];
  double *torque = new double[input->nnode*3];
  double **firstdouble = NULL;
  double **dpages = NULL;
  int    **firsttouch = NULL;
  int    **tpages = NULL;
  for (int run=0; run<num_iter; run++) {
    //make copies
    copy(input->force,  input->force  + input->nnode*3, force);
    copy(input->torque, input->torque + input->nnode*3, torque);
    nl->copy_into(firstdouble, dpages, firsttouch, tpages);

    hw->run(
      HertzWrapper::KERNEL,
      input->x,
      input->v,
      input->omega,
      force, torque,
      NULL, NULL);

    //only check results the first time around
    if (run == 0) {
      hw->d_nl->unload_shear(dpages);
      check_result(input, nl, force, torque, firstdouble, 0.5, false, false);
    }
  }
  delete[] force;
  delete[] torque;

  one_time.push_back(SimpleTimer("cleanup"));
  one_time.back().start();
  delete hw;
  one_time.back().stop_and_add_to_total();
}

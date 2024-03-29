#include "framework.h"
#include "hertz_cudawrapper.h"

#ifndef BLOCK_SIZE
#error "You need to #define BLOCK_SIZE"
#endif
#ifndef KERNEL
#error "You need to #define KERNEL TPA|BPA"
#endif

#define XQUOTE(s) QUOTE(s)
#define QUOTE(str) #str

using namespace std;

void run(struct params *input, int num_iter) {
  if (input->verbose) {
    cout << "# kernel: " << XQUOTE(KERNEL) << endl;
    cout << "# flags: "  << XQUOTE(KERNELFLAGS) << endl;
  }

  NeighListLike *nl = new NeighListLike(input);

  int block_size = BLOCK_SIZE;
  one_time.push_back(SimpleTimer("initialization"));
  one_time.back().start();
  HertzCudaWrapper *hw = new HertzCudaWrapper(
    block_size,
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
    nl->dpages,
    nl->tpages);
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
      HertzCudaWrapper::KERNEL,
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

  per_iter.push_back(SimpleTimer("memcpy_to_dev"));
  per_iter.push_back(SimpleTimer("kernel"));
  per_iter.push_back(SimpleTimer("memcpy_from_dev"));
  per_iter[0].set_total_time(get_cuda_m0());
  per_iter[1].set_total_time(get_cuda_k0());
  per_iter[2].set_total_time(get_cuda_m1());
  per_iter_timings.push_back(get_cuda_m0_raw());
  per_iter_timings.push_back(get_cuda_k0_raw());
  per_iter_timings.push_back(get_cuda_m1_raw());

  one_time.push_back(SimpleTimer("cleanup"));
  one_time.back().start();
  delete hw;
  one_time.back().stop_and_add_to_total();
}

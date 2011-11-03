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

  //per_iter.push_back(SimpleTimer("run"));
  for (int run=0; run<num_iter; run++) {
    //per_iter[0].start();
    hw->run(
      HertzWrapper::KERNEL,
      input->x,
      input->v,
      input->omega,
      input->force,
      input->torque, NULL, NULL);
    //per_iter[0].stop_and_add_to_total();

    if (run == 0) {
      // check results
    }
  }
  per_iter.push_back(SimpleTimer("memcpy_to_dev"));
  per_iter.push_back(SimpleTimer("kernel"));
  per_iter.push_back(SimpleTimer("memcpy_from_dev"));
  per_iter[0].set_total_time(get_m0());
  per_iter[1].set_total_time(get_k0());
  per_iter[2].set_total_time(get_m1());

  one_time.push_back(SimpleTimer("cleanup"));
  one_time.back().start();
  delete hw;
  one_time.back().stop_and_add_to_total();
}

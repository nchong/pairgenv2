#include "framework.h"
#include "hertz_wrapper.h"

using namespace std;

void run(struct params *input, int num_iter) {
  CLWrapper clw(/*platform=*/0,/*device=*/1,/*profiling=*/true);
  NeighListLike *nl = new NeighListLike(input);

  one_time.push_back(SimpleTimer("initialization"));
  one_time.back().start();
  size_t wx = 1;
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

  per_iter.push_back(SimpleTimer("run"));
  for (int run=0; run<num_iter; run++) {
    per_iter[0].start();
    hw->run(
      input->x,
      input->v,
      input->omega,
      input->force,
      input->torque, NULL, NULL);
    per_iter[0].stop_and_add_to_total();

    if (run == 0) {
      // check results
    }
  }

  one_time.push_back(SimpleTimer("cleanup"));
  one_time.back().start();
  delete hw;
  one_time.back().stop_and_add_to_total();
}

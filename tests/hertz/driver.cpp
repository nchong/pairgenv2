#include "framework.h"
#include "hertz_wrapper.h"

using namespace std;

void run(struct params *input, int num_iter) {
  HertzWrapper::kernel_decomposition k =
    (input->cl_kernel == 0) ? HertzWrapper::TPA : HertzWrapper::BPA;

  if (input->verbose) {
    cout << clinfo();
    cout << "# platform " << input->cl_platform
         << " device "    << input->cl_device
         << endl;
    cout << "# kernel: " << (k == 0 ? "TPA" : "BPA") << endl;
    cout << "# block_size: " << input->cl_blocksize << endl;
    if (input->cl_flags) {
      cout << "# flags: " << input->cl_flags << endl;
    }
  }
  CLWrapper clw(/*platform=*/input->cl_platform,/*device=*/input->cl_device,/*profiling=*/false);
  NeighListLike *nl = new NeighListLike(input);

  one_time.push_back(SimpleTimer("initialization"));
  one_time.back().start();
  size_t wx = input->cl_blocksize;
  HertzWrapper *hw = new HertzWrapper(
    clw, wx, input->cl_flags,
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
      k,
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
  per_iter[0].set_total_time(get_m0());
  per_iter[1].set_total_time(get_k0());
  per_iter[2].set_total_time(get_m1());
  per_iter_timings.push_back(get_m0_raw());
  per_iter_timings.push_back(get_k0_raw());
  per_iter_timings.push_back(get_m1_raw());

  one_time.push_back(SimpleTimer("cleanup"));
  one_time.back().start();
  delete hw;
  one_time.back().stop_and_add_to_total();
}

#include "framework.h"
#include "clneighlist.h"

void run(struct params *input, int /* num_iter unused*/) {
  cout << clinfo();
  CLWrapper clw(/*platform=*/0,/*device=*/1,/*profiling=*/true);
  NeighListLike *nl = new NeighListLike(input);
  CLNeighList *gnl = new CLNeighList(clw, /*wx=*/1, nl->inum, nl->maxpage, nl->pgsize);
  gnl->reload(nl->numneigh, nl->firstneigh, nl->pages, nl->maxpage);
}

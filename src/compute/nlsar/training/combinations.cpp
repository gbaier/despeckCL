#include "combinations.h"

#include <vector>
#include <algorithm>

#include "patches.h"

std::vector<float> nlsar::training::get_all_dissim_combs(std::vector<data> patches, std::vector<float> acc)
{
    if (patches.size() == 1) {
        return acc;
    } else {
        data head = patches.back();
        patches.pop_back();
        std::vector<float> dissims {};
        std::transform(patches.begin(),
                       patches.end(),
                       std::back_inserter(dissims),
                       [&head] (data& x) { return head.dissimilarity(x); });
        acc.insert(acc.end(), dissims.begin(), dissims.end());
        return get_all_dissim_combs(patches, acc);
    }
}

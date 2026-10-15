#include "strata/kernels/exl3_route_plan.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

using strata::kernels::Exl3RoutePlan;

static void require(bool condition) {
    if (!condition) throw std::runtime_error("route plan differs from independent grouping");
}

static void check(Exl3RoutePlan& plan, const std::vector<int32_t>& ids, int tokens, int k, int experts) {
    std::map<int, std::vector<int>> reference;
    for (size_t r = 0; r < ids.size(); ++r) reference[ids[r]].push_back((int) r);
    plan.build(ids.data(), tokens, k, experts);
    require(plan.experts.size() == reference.size());
    require(plan.destinations.size() == ids.size());
    require(plan.offsets.size() == reference.size() + 1);
    size_t group = 0;
    for (const auto& item : reference) {
        require(plan.experts[group] == item.first);
        const int begin = plan.offsets[group], end = plan.offsets[group + 1];
        require(end - begin == (int) item.second.size());
        require(std::equal(item.second.begin(), item.second.end(), plan.destinations.begin() + begin));
        for (int r = begin; r < end; ++r) {
            require(plan.token_at((size_t) r) == item.second[(size_t) (r - begin)] / k);
            require(ids[(size_t) plan.destinations[(size_t) r]] == item.first);
        }
        ++group;
    }
    auto rows = plan.destinations;
    std::sort(rows.begin(), rows.end());
    for (size_t r = 0; r < rows.size(); ++r) require(rows[r] == (int) r);
    std::vector<uint8_t> resident(plan.experts.size());
    for (size_t g = 0; g < resident.size(); ++g) resident[g] = (plan.experts[g] % 3 == 0);
    plan.prioritize_resident(resident);
    bool saw_miss = false;
    for (size_t g = 0; g < plan.experts.size(); ++g) {
        const int expert = plan.experts[g];
        if (expert % 3 != 0) saw_miss = true;
        else require(!saw_miss);
        const auto& expected = reference.at(expert);
        require(plan.offsets[g + 1] - plan.offsets[g] == (int) expected.size());
        require(std::equal(expected.begin(), expected.end(), plan.destinations.begin() + plan.offsets[g]));
    }
}

int main() {
    try {
        Exl3RoutePlan plan;
        check(plan, {7, 1, 7, 1, 2, 7}, 2, 3, 8); // shared experts, including a duplicate within a row
        check(plan, {}, 0, 10, 512);
        std::mt19937 random(7319);
        for (int i = 0; i < 1000; ++i) {
            const int tokens = 1 + (int) (random() % 8), k = 1 + (int) (random() % 10);
            std::vector<int32_t> ids((size_t) tokens * k);
            for (auto& id : ids) id = (int32_t) (random() % 512);
            check(plan, ids, tokens, k, 512);
        }
        std::vector<int32_t> large(32768 * 10);
        for (auto& id : large) id = (int32_t) (random() % 512);
        check(plan, large, 32768, 10, 512);
        // Repeated references represent work, not multiple uploads of one expert.
        require(plan.experts.size() == 512 && plan.destinations.size() == large.size());
        for (int invalid : {-1, 512}) {
            bool rejected = false;
            try { plan.build(&invalid, 1, 1, 512); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && plan.experts.empty() && plan.destinations.empty());
        }
        bool overflow = false;
        try { plan.build(nullptr, std::numeric_limits<int>::max(), 10, 512); }
        catch (const std::invalid_argument&) { overflow = true; }
        require(overflow);
        std::cout << "EXL3 routes: stable grouping, 1000 random windows, 32K prompt and invalid inputs PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

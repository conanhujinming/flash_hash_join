#pragma once

#include <cstdint>

namespace flash_join {

struct BuildTuple {
    std::uint64_t key;
    std::uint64_t value;
};

}  // namespace flash_join

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(__SSE4_2__) || defined(_M_X64)
#include <nmmintrin.h>
#elif defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace flash_join {

namespace detail {

inline std::uint32_t crc32c_u64(std::uint32_t seed, std::uint64_t value) noexcept {
#if defined(__SSE4_2__) || defined(_M_X64)
    return static_cast<std::uint32_t>(_mm_crc32_u64(seed, value));
#elif defined(__ARM_FEATURE_CRC32)
    return __crc32cd(seed, value);
#else
    // Portable CRC32C fallback. Release builds enable the hardware path on
    // supported x86-64 and AArch64 CPUs.
    std::uint32_t crc = seed;
    for (unsigned byte = 0; byte < sizeof(value); ++byte) {
        crc ^= static_cast<std::uint8_t>(value);
        value >>= 8;
        for (unsigned bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0U - (crc & 1U);
            crc = (crc >> 1U) ^ (0x82F63B78U & mask);
        }
    }
    return crc;
#endif
}

constexpr std::size_t kBloomTagTableSize = 1U << 11U;
constexpr std::size_t kDistinctBloomTags = 1820;

// CedarDB uses all C(16, 4) four-bit tags, padded to 2^11 entries with a
// deterministic uniform sample. The power-of-two table makes tag selection a
// shift plus a single 4 KiB lookup.
constexpr std::array<std::uint16_t, kBloomTagTableSize> make_bloom_tags() {
    std::array<std::uint16_t, kBloomTagTableSize> tags{};
    std::size_t index = 0;
    for (unsigned a = 0; a < 16; ++a) {
        for (unsigned b = a + 1; b < 16; ++b) {
            for (unsigned c = b + 1; c < 16; ++c) {
                for (unsigned d = c + 1; d < 16; ++d) {
                    tags[index++] = static_cast<std::uint16_t>(
                        (1U << a) | (1U << b) | (1U << c) | (1U << d));
                }
            }
        }
    }

    std::uint32_t state = 0x9E3779B9U;
    while (index < tags.size()) {
        state = state * 1664525U + 1013904223U;
        tags[index++] = tags[(state >> 8U) % kDistinctBloomTags];
    }
    return tags;
}

inline constexpr auto kBloomTags = make_bloom_tags();

}  // namespace detail

// The 64-bit integer hash from Figure 8 of the CedarDB paper: two hardware
// CRC32C instructions produce independent halves, followed by one multiply.
inline std::uint64_t hash64(
    std::uint64_t key,
    std::uint32_t seed1 = 0x243F6A88U,
    std::uint32_t seed2 = 0x85A308D3U) noexcept {
    constexpr std::uint64_t kMix = 0x2545F4914F6CDD1DULL;
    const auto lower = static_cast<std::uint64_t>(detail::crc32c_u64(seed1, key));
    const auto upper = static_cast<std::uint64_t>(detail::crc32c_u64(seed2, key)) << 32U;
    return (lower | upper) * kMix;
}

inline std::uint16_t bloom_tag(std::uint64_t hash) noexcept {
    constexpr unsigned kIndexShift = 32U - 11U;
    const auto index = static_cast<std::uint32_t>(hash) >> kIndexShift;
    return detail::kBloomTags[index];
}

}  // namespace flash_join

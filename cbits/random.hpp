// Copyright (c) 2019, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "config.hpp"
#include <cstdint>

TCM_NAMESPACE_BEGIN

/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is xoshiro256** 1.0, one of our all-purpose, rock-solid
   generators. It has excellent (sub-ns) speed, a state (256 bits) that is
   large enough for any parallel application, and it passes all tests we
   are aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

// Code below is the original implementation of xoshiro256** by D. Blackman and
// S.Vigna. I only changed function signatures and names and put them in a class.

class xoshiro256starstar {
  private:
    uint64_t s[4];

    /// Rotate-left operation. It is assumed that the compiler replaces it with a
    /// single instruction.
    static constexpr auto rotl(uint64_t const x, int const k) noexcept
        -> uint64_t
    {
        return (x << k) | (x >> (64 - k));
    }

    /// SplitMix64 generator. It is the suggested way to initialise the
    /// xoshiro256** generator, since it can't directly be initialised with 0.
    /* This is a fixed-increment version of Java 8's SplittableRandom generator
       See http://dx.doi.org/10.1145/2714064.2660195 and
       http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

       It is a very fast generator passing BigCrush, and it can be useful if
       for some reason you absolutely want 64 bits of state. */
    class splitmix64 {
      private:
        uint64_t x; /* The state can be seeded with any value. */

      public:
        constexpr splitmix64(uint64_t const seed) noexcept : x{seed} {}
        constexpr splitmix64(splitmix64 const&) noexcept = default;
        constexpr splitmix64(splitmix64&&) noexcept      = default;
        constexpr auto operator=(splitmix64 const&) noexcept
            -> splitmix64&     = default;
        constexpr auto operator=(splitmix64&&) noexcept
            -> splitmix64&     = default;

        constexpr auto operator()() noexcept -> uint64_t
        {
            uint64_t z = (x += 0x9e3779b97f4a7c15);
            z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z          = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        }
    };

  public:
    using result_type = uint64_t;

    constexpr xoshiro256starstar(uint64_t const _seed) noexcept : s{0, 0, 0, 0}
    {
        seed(_seed);
    }

    constexpr xoshiro256starstar(xoshiro256starstar const&) noexcept = default;
    constexpr xoshiro256starstar(xoshiro256starstar&&) noexcept      = default;
    constexpr auto operator    =(xoshiro256starstar const&) noexcept
        -> xoshiro256starstar& = default;
    constexpr auto operator    =(xoshiro256starstar&&) noexcept
        -> xoshiro256starstar& = default;

    constexpr auto operator()() noexcept -> uint64_t
    {
        uint64_t const result = rotl(s[1] * 5, 7) * 9;
        uint64_t const t      = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);
        return result;
    }

    static constexpr auto min() noexcept -> uint64_t { return uint64_t{0}; }
    static constexpr auto max() noexcept -> uint64_t { return ~uint64_t{0}; }

    /* This is the jump function for the generator. It is equivalent
       to 2^128 calls to next(); it can be used to generate 2^128
       non-overlapping subsequences for parallel computations. */
    constexpr auto jump() noexcept -> void
    {
        constexpr uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                     0xa9582618e03fc9aa, 0x39abdc4529b1661c};
        static_assert(std::size(JUMP) == sizeof JUMP / sizeof *JUMP);

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        uint64_t s2 = 0;
        uint64_t s3 = 0;
        for (auto i = 0u; i < ::std::size(JUMP); ++i) {
            for (auto b = 0u; b < 64u; ++b) {
                if (JUMP[i] & uint64_t{1} << b) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                (*this)();
            }
        }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }

    /* This is the long-jump function for the generator. It is equivalent to
       2^192 calls to next(); it can be used to generate 2^64 starting points,
       from each of which jump() will generate 2^64 non-overlapping
       subsequences for parallel distributed computations. */
    constexpr auto long_jump() noexcept -> void
    {
        constexpr uint64_t LONG_JUMP[] = {
            0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
            0x39109bb02acbe635};

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        uint64_t s2 = 0;
        uint64_t s3 = 0;
        for (auto i = 0u; i < ::std::size(LONG_JUMP); ++i) {
            for (auto b = 0u; b < 64u; ++b) {
                if (LONG_JUMP[i] & uint64_t{1} << b) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                (*this)();
            }
        }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }

    constexpr auto seed(uint64_t const _seed) noexcept -> void
    {
        splitmix64 g{_seed};
        s[0] = g();
        s[1] = g();
        s[2] = g();
        s[3] = g();
    }
};

using RandomGenerator = xoshiro256starstar;

auto global_random_generator() -> RandomGenerator&;

constexpr auto random_bounded(uint32_t const range, RandomGenerator& g) noexcept
    -> uint32_t
{
    static_assert(noexcept(std::declval<RandomGenerator&>()()),
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    auto const random32 = [&g]() -> uint64_t {
        static_assert(std::is_same_v<RandomGenerator::result_type, uint64_t>);
        return g() & uint64_t{0xFFFFFFFF};
    };
    auto multiresult = random32() * range;
    auto leftover    = static_cast<uint32_t>(multiresult);
    if (leftover < range) {
        auto const threshold = -range % range;
        while (leftover < threshold) {
            multiresult = random32() * range;
            leftover    = static_cast<uint32_t>(multiresult);
        }
    }
    return multiresult >> 32;
}

TCM_NAMESPACE_END

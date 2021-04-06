#ifndef testingResources_asserts_hpp
#define testingResources_asserts_hpp
#include <gtest/gtest.h>

namespace testingResources {

template <class T>
inline void ASSERT_ABOUT(T expected, T computed, T difference) {
    ASSERT_LT(abs((expected - computed) / (0.5*(expected + computed + 1E-30))), difference) << "expected difference between " << expected << " and " << computed << " to be less than " << difference;
}

}  // namespace testingResources
#endif
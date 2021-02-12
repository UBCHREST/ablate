#include <gtest/gtest.h>

#include "MpiTestEventListener.h"
#include "MpiTestFixture.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Store the input parameters
    const bool inMpiTestRun = MpiTestFixture::InitializeTestingEnvironment(&argc, &argv);
    if (inMpiTestRun) {
        testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());

        listeners.Append(new MpiTestEventListener());
    }

    return RUN_ALL_TESTS();
}
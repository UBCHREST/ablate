#include <gtest/gtest.h>
#include "mpiTestEventListener.hpp"
#include "mpiTestFixture.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Store the input parameters
    const bool inMpiTestRun = testingResources::MpiTestFixture::InitializeTestingEnvironment(&argc, &argv);
    if (inMpiTestRun) {
        testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());

        listeners.Append(new testingResources::MpiTestEventListener());
    }

    return RUN_ALL_TESTS();
}
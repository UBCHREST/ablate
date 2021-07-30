#include <memory>
#include "gtest/gtest.h"
#include "monitors/logs/streamLog.hpp"

using namespace ablate;

TEST(StreamLog, ShouldPrintToStream) {
    // arrange create the stream
    std::stringstream outputStream;

    // create the log
    {
        monitors::logs::StreamLog log(outputStream);

        // act
        log.Print("StreamLog\n");
        log.Printf("Line %d\n", 12);
        log.Printf("Line %d, Line %d\n", 14, 16);
    }

    // assert
    ASSERT_EQ("StreamLog\nLine 12\nLine 14, Line 16\n", outputStream.str());
}

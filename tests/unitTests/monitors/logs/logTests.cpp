#include <memory>
#include <monitors/logs/streamLog.hpp>
#include "gtest/gtest.h"

using namespace ablate;

struct LogTestVectorParams {
    std::string testName;
    std::vector<double> values;
    std::string name;
    std::string format;
    std::string expectedOutput;
};

class LogTestVectorFixture : public ::testing::TestWithParam<LogTestVectorParams> {};

TEST_P(LogTestVectorFixture, ShouldPrintVectors) {
    // arrange
    const auto& params = GetParam();

    // create the stream
    std::stringstream outputStream;

    // Create the test log.  Stream log is being used for convenience, but we could have used a mock
    std::shared_ptr<monitors::logs::Log> log = std::make_shared<monitors::logs::StreamLog>(outputStream);

    // act
    if (params.format.empty()) {
        log->Print(params.name.c_str(), params.values);
    } else {
        log->Print(params.name.c_str(), params.values, params.format.c_str());
    }

    // assert
    ASSERT_EQ(params.expectedOutput, outputStream.str());
}

INSTANTIATE_TEST_SUITE_P(
    LogTests, LogTestVectorFixture,
    testing::Values((LogTestVectorParams){.testName = "default_vector", .values = {3.3, 4.4, 5.5}, .name = " output", .format = {}, .expectedOutput = " output: [3.3, 4.4, 5.5]"},
                    (LogTestVectorParams){.testName = "empty_vector", .values = {}, .name = " output", .format = {}, .expectedOutput = " output: []"},
                    (LogTestVectorParams){
                        .testName = "custom_format_vector", .values = {3.3432, 4.34534, 6.4565}, .name = " output", .format = "%2.3g", .expectedOutput = " output: [3.34, 4.35, 6.46]"}),
    [](const testing::TestParamInfo<LogTestVectorParams>& info) { return info.param.testName + std::to_string(info.index); });

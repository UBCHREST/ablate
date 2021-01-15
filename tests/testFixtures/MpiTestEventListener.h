#ifndef ABLATE_MPITESTEVENTLISTENER_H
#define ABLATE_MPITESTEVENTLISTENER_H
#include <gtest/gtest.h>

class MpiTestEventListener : public ::testing::EmptyTestEventListener
{
   public:
    virtual void OnTestPartResult(const ::testing::TestPartResult& test_part_result)
    {
        printf("%s in %s:%d\n%s\n",
               test_part_result.failed() ? "*** Failure" : "Success",
               test_part_result.file_name(),
               test_part_result.line_number(),
               test_part_result.summary());
    }
};

#endif  // ABLATE_MPITESTEVENTLISTENER_H
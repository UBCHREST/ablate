#ifndef TESTING_RESOURCE_ASSERT
#define TESTING_RESOURCE_ASSERT

namespace testingResources::asserts {

/**
 * This is an empty interface that other asserts should utilize
 */
class Assert {
   public:
    virtual ~Assert() = default;
};
}

#endif  // TESTING_RESOURCE_ASSERT
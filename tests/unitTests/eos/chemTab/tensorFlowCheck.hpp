#ifndef WITH_TENSORFLOW
#define ONLY_WITH_TENSORFLOW_CHECK                                       \
    SUCCEED() << ("Test is only applicable when built with TensorFlow"); \
    return;
#define ONLY_WITHOUT_TENSORFLOW_CHECK \
    {}
#else
#define ONLY_WITH_TENSORFLOW_CHECK \
    {}
#define ONLY_WITHOUT_TENSORFLOW_CHECK                                     \
    SUCCEED() << "Test is only applicable when built without TensorFlow"; \
    return;
#endif
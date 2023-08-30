#ifndef ABLATELIBRARY_NONCOPYABLE_HPP
#define ABLATELIBRARY_NONCOPYABLE_HPP

namespace ablate::utilities {

/**
 * A class can implement this interface to prevent an instance of it from accidentally being copied
 */
class NonCopyable {
   protected:
    // Do not allow the creation outside a derived class
    constexpr NonCopyable() = default;
    ~NonCopyable() = default;

   public:
    // Prevent a class from being copied
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
};
};      // namespace ablate::utilities
#endif  // ABLATELIBRARY_NONCOPYABLE_HPP

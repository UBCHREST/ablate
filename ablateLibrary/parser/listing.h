#ifndef ABLATELIBRARY_LISTING_H
#define ABLATELIBRARY_LISTING_H

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ablate::parser {

class Listing {
   public:
    struct ArgumentEntry {
        const std::string name;
        const std::string interface;
        const std::string description;
        bool operator==(const ArgumentEntry& other) const;
    };
    struct ClassEntry {
        const std::string interface;
        const std::string className;
        const std::string description;
        const std::vector<ArgumentEntry> arguments;
        const bool defaultConstructor;
        bool operator==(const ClassEntry& other) const;
    };

    // provide access to the listing
    virtual void RecordListing(ClassEntry entry);

    // get the singleton instance
    static Listing& Get();

    Listing(Listing& other) = delete;
    void operator=(const Listing&) = delete;

    // define operators to write to a stream
    friend std::ostream& operator<<(std::ostream& os, const Listing& listing);
    friend std::ostream& operator<<(std::ostream& os, const ArgumentEntry& argumentEntry);
    friend std::ostream& operator<<(std::ostream& os, const ClassEntry& classEntry);

    // Provide a way to replace the listing
    static void ReplaceListing(std::shared_ptr<Listing>);

   protected:
    Listing() = default;

   private:
    std::map<std::string, std::vector<ClassEntry>> entries;

    inline static std::shared_ptr<Listing> listing;
};
}  // namespace ablate::parser
#endif  // ABLATELIBRARY_LISTING_H

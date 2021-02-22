#include "listing.h"
#include "utilities/demangler.hpp"

void ablate::parser::Listing::RecordListing(ablate::parser::Listing::ClassEntry entry) { entries[entry.interface].push_back(entry); }
ablate::parser::Listing& ablate::parser::Listing::Get() {
    if (listing == nullptr) {
        listing = std::shared_ptr<Listing>(new Listing());
    }

    return *listing;
}
std::ostream& ablate::parser::operator<<(std::ostream& os, const ablate::parser::Listing& listing) {
    for (auto interface : listing.entries) {
        os << "Interface: " << utilities::Demangler::Demangle(interface.first) << std::endl;
        for (auto classEntry : interface.second) {
            os << '\t' << classEntry;
        }
    }

    return os;
}

std::ostream& ablate::parser::operator<<(std::ostream& os, const ablate::parser::Listing::ArgumentEntry& argumentEntry) {
    os << argumentEntry.name << "(" << utilities::Demangler::Demangle(argumentEntry.interface) << ") - " << argumentEntry.description;

    return os;
}

std::ostream& ablate::parser::operator<<(std::ostream& os, const ablate::parser::Listing::ClassEntry& classEntry) {
    os << classEntry.className  << (classEntry.defaultConstructor ? "*": "") << " - " << classEntry.description << std::endl;
    for (auto argumentEntry : classEntry.arguments) {
        os << '\t' << '\t' << argumentEntry << std::endl;
    }
    return os;
}

void ablate::parser::Listing::ReplaceListing(std::shared_ptr<Listing> replacementListing) { listing = replacementListing; }
bool ablate::parser::Listing::ClassEntry::operator==(const ablate::parser::Listing::ClassEntry& other) const {
    return className == other.className && interface == other.interface && description == other.description && arguments == other.arguments;
}
bool ablate::parser::Listing::ArgumentEntry::operator==(const ablate::parser::Listing::ArgumentEntry& other) const {
    return name == other.name && interface == other.interface && description == other.description;
}

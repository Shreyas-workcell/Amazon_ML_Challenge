// High-Performance C++ Core Engine for Business Entity Resolution
// Implements direct TSV data ingestion, multi-threaded country-partitioned inverted indexing,
// Unicode diacritic stripping, corporate legal suffix parser, RapidFuzz string metrics,
// and ML-based rule scoring for the Amazon ML Challenge.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <thread>
#include <mutex>
#include <future>
#include <chrono>
#include <numeric>
#include <atomic>

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

namespace {

// Purely grammatical noise stopwords (never strip business terms like group, holdings, solutions, mines, foods, or cities)
static const std::unordered_set<std::string> STOPWORDS = {
    "and", "the", "for", "with", "of", "in", "at", "by", "from", "de", "la", "le", "des", "du", "et", "d"
};

// Address noise stopwords that appear universally in addresses
static const std::unordered_set<std::string> ADDRESS_STOPWORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln",
    "way", "blvd", "boulevard", "floor", "fl", "unit", "apt", "apartment",
    "suite", "ste", "building", "bldg", "near", "opp", "opposite", "behind",
    "beside", "dist", "district", "post", "po", "box", "nagar", "colony", "marg",
    "sector", "sec", "phase", "rue", "place", "allee", "zone",
    "block", "blk", "plot", "no", "shop", "flat"
};

// Fast 32-bit FNV-1a hash
inline uint32_t fnv1a_32(std::string_view s) {
    uint32_t hash = 2166136261u;
    for (char c : s) {
        hash ^= (uint8_t)c;
        hash *= 16777619u;
    }
    return hash;
}

// Extract locality & street name hashes from address
std::vector<uint32_t> extract_address_token_hashes_cpp_internal(std::string_view clean_addr) {
    std::vector<uint32_t> hashes;
    hashes.reserve(8);
    size_t i = 0;
    while (i < clean_addr.size()) {
        if (std::isalpha((unsigned char)clean_addr[i])) {
            size_t start = i;
            while (i < clean_addr.size() && std::isalpha((unsigned char)clean_addr[i])) {
                i++;
            }
            std::string w(clean_addr.substr(start, i - start));
            if (w.length() >= 3 && ADDRESS_STOPWORDS.find(w) == ADDRESS_STOPWORDS.end()) {
                uint32_t h = fnv1a_32(w);
                bool exists = false;
                for (uint32_t eh : hashes) {
                    if (eh == h) { exists = true; break; }
                }
                if (!exists) {
                    hashes.push_back(h);
                    if (hashes.size() >= 8) break;
                }
            }
        } else {
            i++;
        }
    }
    return hashes;
}

struct LegalPattern {
    std::string pattern;
    std::string canonical;
};

static const std::vector<LegalPattern> LEGAL_PATTERNS = {
    {"private limited", "pvt_ltd"}, {"pvt ltd", "pvt_ltd"}, {"p. ltd", "pvt_ltd"},
    {"pvt limited", "pvt_ltd"}, {"prvt ltd", "pvt_ltd"},
    {"limited", "ltd"}, {"ltd", "ltd"}, {"l.t.d", "ltd"},
    {"limited liability company", "llc"}, {"llc", "llc"}, {"l.l.c", "llc"},
    {"incorporated", "inc"}, {"inc", "inc"}, {"i.n.c", "inc"},
    {"corporation", "corp"}, {"corp", "corp"}, {"c.o.r.p", "corp"},
    {"limited liability partnership", "llp"}, {"llp", "llp"}, {"l.l.p", "llp"},
    {"company", "co"}, {"co", "co"}, {"c.o", "co"},
    {"societe a responsabilite limitee", "sarl"}, {"sarl", "sarl"}, {"s.a.r.l", "sarl"},
    {"societe par actions simplifiee", "sas"}, {"sas", "sas"}, {"s.a.s", "sas"},
    {"sasu", "sasu"}, {"s.a.s.u", "sasu"},
    {"societe anonyme", "sa"}, {"sa", "sa"}, {"s.a", "sa"},
    {"eurl", "eurl"}, {"e.u.r.l", "eurl"},
    {"snc", "snc"}, {"s.n.c", "snc"},
    {"sci", "sci"}, {"s.c.i", "sci"},
    {"gmbh", "gmbh"}, {"g.m.b.h", "gmbh"},
    {"ag", "ag"}, {"a.g", "ag"},
    {"bv", "bv"}, {"b.v", "bv"},
    {"nv", "nv"}, {"n.v", "nv"},
    {"spa", "spa"}, {"s.p.a", "spa"}
};

// Fast Unicode diacritic & accent folding to ASCII
std::string normalize_text_cpp_internal(std::string_view input) {
    std::string result;
    result.reserve(input.size());

    size_t i = 0;
    while (i < input.size()) {
        unsigned char c = (unsigned char)input[i];
        
        if (c == 0xC3 && i + 1 < input.size()) { // UTF-8 Latin-1 Supplement
            i++;
            unsigned char next_c = (unsigned char)input[i];

            if (next_c >= 0x80 && next_c <= 0x85) result += 'a';      // à, á, â, ã, ä, å
            else if (next_c == 0x86) result += "ae";                 // æ
            else if (next_c == 0x87) result += 'c';                  // ç
            else if (next_c >= 0x88 && next_c <= 0x8B) result += 'e';  // è, é, ê, ë
            else if (next_c >= 0x8C && next_c <= 0x8F) result += 'i';  // ì, í, î, ï
            else if (next_c == 0x91) result += 'n';                  // ñ
            else if (next_c >= 0x92 && next_c <= 0x96) result += 'o';  // ò, ó, ô, õ, ö
            else if (next_c >= 0x99 && next_c <= 0x9C) result += 'u';  // ù, ú, û, ü
            else if (next_c == 0x9D) result += 'y';                  // ý
            else if (next_c >= 0xA0 && next_c <= 0xA5) result += 'a';  // À, Á, Â, Ã, Ä, Å
            else if (next_c == 0xA7) result += 'c';                  // Ç
            else if (next_c >= 0xA8 && next_c <= 0xAB) result += 'e';  // È, É, Ê, Ë
            else if (next_c >= 0xAC && next_c <= 0xAF) result += 'i';  // Ì, Í, Î, Ï
            else if (next_c == 0xB1) result += 'n';                  // Ñ
            else if (next_c >= 0xB2 && next_c <= 0xB6) result += 'o';  // Ò, Ó, Ô, Õ, Ö
            else if (next_c >= 0xB9 && next_c <= 0xBC) result += 'u';  // Ù, Ú, Û, Ü
            else result += ' ';
        } else if (std::isalnum(c)) {
            result += (char)std::tolower(c);
        } else if (c == ' ' || c == '-' || c == '.' || c == ',' || c == '/' || c == '&' || c == '\'' || c == '_') {
            if (!result.empty() && result.back() != ' ') {
                result += ' ';
            }
        }
        i++;
    }

    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

// Extract root company name and legal entity type
std::pair<std::string, std::string> parse_legal_suffix_cpp_internal(const std::string& clean_name) {
    std::string root = clean_name;
    std::string legal_type = "none";

    std::string lower_root = root;
    for (char& c : lower_root) c = (char)std::tolower((unsigned char)c);

    for (const auto& lp : LEGAL_PATTERNS) {
        if (lower_root.length() < lp.pattern.length()) continue;
        size_t pos = lower_root.rfind(lp.pattern);
        if (pos != std::string::npos &&
            (pos == 0 || lower_root[pos - 1] == ' ') &&
            (pos + lp.pattern.length() == lower_root.length() || lower_root[pos + lp.pattern.length()] == ' ')) {
            legal_type = lp.canonical;
            root.erase(pos, lp.pattern.length());
            break;
        }
    }

    size_t first = root.find_first_not_of(" .-,");
    if (first != std::string::npos) {
        size_t last = root.find_last_not_of(" .-,");
        root = root.substr(first, (last - first + 1));
    } else {
        root = clean_name;
    }

    return {root, legal_type};
}

// Open-set country normalization
std::string normalize_country_cpp_internal(std::string_view raw) {
    std::string s;
    for (char c : raw) {
        if (std::isalnum((unsigned char)c) || c == ' ') {
            s += (char)std::tolower((unsigned char)c);
        }
    }
    size_t f = s.find_first_not_of(" ");
    if (f == std::string::npos) return "UNKNOWN";
    size_t l = s.find_last_not_of(" ");
    s = s.substr(f, l - f + 1);

    if (s == "us" || s == "usa" || s == "united states" || s == "u s a" || s == "u s") return "US";
    if (s == "in" || s == "ind" || s == "india" || s == "bharat") return "IN";
    if (s == "fr" || s == "fra" || s == "france" || s == "republique francaise") return "FR";
    if (s == "gb" || s == "uk" || s == "united kingdom" || s == "great britain") return "GB";
    if (s == "de" || s == "germany" || s == "deutschland") return "DE";
    if (s == "ca" || s == "canada") return "CA";
    if (s == "au" || s == "australia") return "AU";

    std::string upper;
    for (char c : s) upper += (char)std::toupper((unsigned char)c);
    return upper;
}

// Extract search tokens (length >= 3 and not stopwords)
std::vector<std::string> extract_tokens_cpp_internal(const std::string& text) {
    std::vector<std::string> tokens;
    std::string word;
    std::stringstream ss(text);
    std::unordered_set<std::string> seen;

    while (ss >> word) {
        if (word.length() >= 3 && STOPWORDS.find(word) == STOPWORDS.end()) {
            if (seen.insert(word).second) {
                tokens.push_back(word);
            }
        }
    }
    return tokens;
}

// Extract street numbers from address
std::vector<std::string> extract_street_numbers_cpp_internal(std::string_view addr) {
    std::vector<std::string> numbers;
    std::unordered_set<std::string> seen;
    size_t i = 0;

    while (i < addr.size()) {
        if (std::isdigit((unsigned char)addr[i])) {
            size_t start = i;
            while (i < addr.size() && (std::isdigit((unsigned char)addr[i]) || std::isalpha((unsigned char)addr[i]))) {
                i++;
            }
            std::string num(addr.substr(start, i - start));
            for (char& c : num) c = (char)std::tolower((unsigned char)c);
            if (seen.insert(num).second) {
                numbers.push_back(num);
            }
        } else {
            i++;
        }
    }
    return numbers;
}

// Extract postal code from address
std::string extract_postal_code_cpp_internal(std::string_view addr) {
    size_t i = 0;
    while (i < addr.size()) {
        if (std::isdigit((unsigned char)addr[i])) {
            size_t start = i;
            while (i < addr.size() && (std::isdigit((unsigned char)addr[i]) || addr[i] == '-')) {
                i++;
            }
            std::string code(addr.substr(start, i - start));
            size_t dash = code.find('-');
            if (dash != std::string::npos && dash == 5) {
                return code.substr(0, 5); // US 5-digit from 5+4
            }
            if (code.length() >= 4 && code.length() <= 6) {
                bool all_digit = true;
                for (char c : code) {
                    if (!std::isdigit((unsigned char)c)) { all_digit = false; break; }
                }
                if (all_digit) return code;
            }
        } else {
            i++;
        }
    }
    return "";
}

// Extract acronym from multi-word business names (e.g. "tata consultancy services" -> "tcs")
std::string extract_acronym_cpp_internal(const std::string& text) {
    std::string acronym;
    std::stringstream ss(text);
    std::string word;
    int word_count = 0;
    while (ss >> word) {
        if (!word.empty() && std::isalnum((unsigned char)word[0])) {
            acronym += (char)std::tolower((unsigned char)word[0]);
            word_count++;
        }
    }
    if (word_count >= 2 && acronym.length() >= 2 && acronym.length() <= 6) {
        return acronym;
    }
    return "";
}

// Helper to fast split TSV lines
inline void split_tsv_fields(std::string_view line, std::vector<std::string_view>& fields) {
    fields.clear();
    size_t start = 0;
    size_t pos = 0;
    while ((pos = line.find('\t', start)) != std::string_view::npos) {
        fields.emplace_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    fields.emplace_back(line.substr(start));
}

// Structure for Target entity (S2 / S3)
struct TargetRecord {
    std::string id;
    std::string root_name;
    std::string acronym;
    std::vector<std::string> tokens;
    std::vector<std::string> street_nums;
    std::string postal_code;
    std::vector<uint32_t> addr_hashes;
    uint8_t source_type; // 2 for Source 2, 3 for Source 3
};

// Structure for Reference entity (S1)
struct S1Record {
    std::string id;
    std::string root_name;
    std::string acronym;
    std::string country;
    std::vector<std::string> tokens;
    std::vector<std::string> street_nums;
    std::string postal_code;
    std::vector<uint32_t> addr_hashes;
};

// Per-country target index with dedicated dual-source inverted maps (prevents S2 from starving S3)
struct CountryTargetIndex {
    std::vector<TargetRecord> records;
    std::unordered_map<std::string, std::vector<uint32_t>> exact_name_map_s2;
    std::unordered_map<std::string, std::vector<uint32_t>> exact_name_map_s3;
    std::unordered_map<std::string, std::vector<uint32_t>> token_inverted_index_s2;
    std::unordered_map<std::string, std::vector<uint32_t>> token_inverted_index_s3;
    // Frequency tables are built in a first pass.  We only retain rare name
    // tokens in the inverted index; capping a posting list by input order can
    // silently discard the true match on the full challenge data.
    std::unordered_map<std::string, size_t> token_frequency_s2;
    std::unordered_map<std::string, size_t> token_frequency_s3;
    std::unordered_set<std::string> query_root_names;
    std::unordered_set<std::string> query_tokens;
    std::unordered_set<std::string> query_postal_codes;
    std::unordered_set<uint32_t> query_address_hashes;
    std::unordered_map<std::string, std::vector<uint32_t>> postal_index_s2;
    std::unordered_map<std::string, std::vector<uint32_t>> postal_index_s3;
    std::unordered_map<uint32_t, std::vector<uint32_t>> address_hash_index_s2;
    std::unordered_map<uint32_t, std::vector<uint32_t>> address_hash_index_s3;
};

// Resolution rules configuration
struct ResolutionRules {
    double match_threshold = 0.55;
    double min_candidate_score = 0.35;
    double exact_name_score = 1.0;
    double street_num_bonus = 0.15;
    double street_num_penalty = -0.35;
    double postal_bonus = 0.15;
    double postal_penalty = -0.35;
    double addr_bonus = 0.15;
    double addr_penalty = -0.35;
    double weight_name = 0.65;
    double weight_street = 0.20;
    double weight_postal = 0.15;
    size_t max_candidates_per_entity = 10; // Compact candidate set for final ranking boost
    size_t max_matches_per_entity = 12; // Ground truth contains up to 11 links/entity
    size_t max_posting_list_size = 1000; // Rare-token frequency cutoff, not input-order cap
    size_t max_exact_posting_size = 50;
    int num_threads = 8;
};
} // namespace

extern "C" {

// Fast diacritic & accent folding
EXPORT_API void normalize_text_cpp(const char* input, char* output, int max_len) {
    if (!input || !output || max_len <= 0) return;
    std::string res = normalize_text_cpp_internal(input);
    std::strncpy(output, res.c_str(), max_len - 1);
    output[max_len - 1] = '\0';
}

// Strip legal corporate suffixes
EXPORT_API void parse_legal_suffix_cpp(const char* input, char* root_output, char* legal_type_output, int max_len) {
    if (!input || !root_output || !legal_type_output || max_len <= 0) return;
    auto [root, legal] = parse_legal_suffix_cpp_internal(input);
    std::strncpy(root_output, root.c_str(), max_len - 1);
    root_output[max_len - 1] = '\0';
    int legal_max = std::min(max_len, 256);
    std::strncpy(legal_type_output, legal.c_str(), legal_max - 1);
    legal_type_output[legal_max - 1] = '\0';
}

// Fast Levenshtein Distance
EXPORT_API int fast_levenshtein_distance(const char* s1, const char* s2) {
    if (!s1 || !s2) return 0;
    int len1 = std::strlen(s1);
    int len2 = std::strlen(s2);

    if (len1 == 0) return len2;
    if (len2 == 0) return len1;

    std::vector<int> col(len2 + 1);
    std::vector<int> prevCol(len2 + 1);

    for (int i = 0; i <= len2; ++i) prevCol[i] = i;

    for (int i = 0; i < len1; ++i) {
        col[0] = i + 1;
        for (int j = 0; j < len2; ++j) {
            int cost = (s1[i] == s2[j]) ? 0 : 1;
            col[j + 1] = std::min({ col[j] + 1, prevCol[j + 1] + 1, prevCol[j] + cost });
        }
        prevCol = col;
    }

    return col[len2];
}

// Fast Jaro-Winkler Similarity Metric (Returns 0.0 to 1.0)
EXPORT_API double fast_jaro_winkler(const char* s1, const char* s2) {
    if (!s1 || !s2) return 0.0;
    int len1 = std::strlen(s1);
    int len2 = std::strlen(s2);

    if (len1 == 0 && len2 == 0) return 1.0;
    if (len1 == 0 || len2 == 0) return 0.0;

    int match_distance = (std::max(len1, len2) / 2) - 1;
    if (match_distance < 0) match_distance = 0;

    std::vector<bool> s1_matches(len1, false);
    std::vector<bool> s2_matches(len2, false);

    int matches = 0;
    for (int i = 0; i < len1; i++) {
        int start = std::max(0, i - match_distance);
        int end = std::min(i + match_distance + 1, len2);

        for (int j = start; j < end; j++) {
            if (s2_matches[j]) continue;
            if (s1[i] != s2[j]) continue;
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches++;
            break;
        }
    }

    if (matches == 0) return 0.0;

    double transpositions = 0;
    int k = 0;
    for (int i = 0; i < len1; i++) {
        if (!s1_matches[i]) continue;
        while (!s2_matches[k]) k++;
        if (s1[i] != s2[k]) transpositions += 0.5;
        k++;
    }

    double jaro = ((double)matches / len1 + (double)matches / len2 + (matches - transpositions) / matches) / 3.0;

    int prefix = 0;
    for (int i = 0; i < std::min({len1, len2, 4}); i++) {
        if (s1[i] == s2[i]) prefix++;
        else break;
    }

    return jaro + (prefix * 0.1 * (1.0 - jaro));
}

// Backward-compatible candidate pairs builder
EXPORT_API int build_candidate_pairs_cpp(
    const char* s1_tsv_path,
    const char* target_tsv_path,
    const char* output_tsv_path
) {
    if (!s1_tsv_path || !target_tsv_path || !output_tsv_path) return -1;

    std::ifstream s1_file(s1_tsv_path);
    std::ifstream tgt_file(target_tsv_path);
    std::ofstream out_file(output_tsv_path);

    if (!s1_file.is_open() || !tgt_file.is_open() || !out_file.is_open()) {
        return -1;
    }

    out_file << "source1_entity_id\tcandidate_entity_ids\n";

    std::unordered_map<std::string, std::vector<std::string>> token_index;
    std::string line;

    std::getline(tgt_file, line);
    while (std::getline(tgt_file, line)) {
        std::stringstream ss(line);
        std::string id, name;
        if (std::getline(ss, id, '\t') && std::getline(ss, name, '\t')) {
            std::stringstream name_ss(name);
            std::string word;
            while (name_ss >> word) {
                if (word.length() >= 3) {
                    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                    if (token_index[word].size() < 30) {
                        token_index[word].push_back(id);
                    }
                }
            }
        }
    }

    int pair_count = 0;
    std::getline(s1_file, line);
    while (std::getline(s1_file, line)) {
        std::stringstream ss(line);
        std::string s1_id, s1_name;
        if (std::getline(ss, s1_id, '\t') && std::getline(ss, s1_name, '\t')) {
            std::stringstream name_ss(s1_name);
            std::string word;
            std::unordered_set<std::string> candidates;
            while (name_ss >> word) {
                if (word.length() >= 3) {
                    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                    if (token_index.find(word) != token_index.end()) {
                        for (const auto& cand_id : token_index[word]) {
                            candidates.insert(cand_id);
                            if (candidates.size() >= 50) break;
                        }
                    }
                }
                if (candidates.size() >= 50) break;
            }

            out_file << s1_id << "\t";
            bool first = true;
            for (const auto& cand_id : candidates) {
                if (!first) out_file << ",";
                out_file << cand_id;
                first = false;
                pair_count++;
            }
            out_file << "\n";
        }
    }

    return pair_count;
}

// Memory free helper for returned JSON strings
EXPORT_API void free_string_cpp(char* str) {
    if (str) {
        delete[] str;
    }
}

// HIGH-PERFORMANCE DIRECT TSV RESOLUTION ENGINE
// Reads TSVs directly from disk in C++, partitions by country, executes multi-threaded
// inverted index candidate retrieval and ML-based rule scoring, and writes proper TSV outputs.
EXPORT_API const char* run_entity_resolution_pipeline_cpp(
    const char* s1_tsv_path,
    const char* s2_tsv_path,
    const char* s3_tsv_path,
    const char* candidate_output_path,
    const char* matching_output_path,
    const char* rules_json_str
) {
    auto t_start = std::chrono::steady_clock::now();

    if (!s1_tsv_path || !s2_tsv_path || !s3_tsv_path || !candidate_output_path || !matching_output_path) {
        std::string err = "{\"status\": \"error\", \"message\": \"Null file path provided.\"}";
        char* res = new char[err.size() + 1];
        std::strcpy(res, err.c_str());
        return res;
    }

    ResolutionRules rules;
    if (rules_json_str && std::strlen(rules_json_str) > 0) {
        // Simple fast parsing of rule numbers from json
        std::string rstr(rules_json_str);
        auto parse_double = [&](const std::string& key, double& target) {
            size_t p = rstr.find(key);
            if (p != std::string::npos) {
                size_t colon = rstr.find(':', p);
                if (colon != std::string::npos) {
                    target = std::stod(rstr.substr(colon + 1));
                }
            }
        };
        auto parse_int = [&](const std::string& key, int& target) {
            size_t p = rstr.find(key);
            if (p != std::string::npos) {
                size_t colon = rstr.find(':', p);
                if (colon != std::string::npos) {
                    target = std::stoi(rstr.substr(colon + 1));
                }
            }
        };
        parse_double("\"match_threshold\"", rules.match_threshold);
        parse_double("\"min_candidate_score\"", rules.min_candidate_score);
        parse_double("\"street_num_bonus\"", rules.street_num_bonus);
        parse_double("\"street_num_penalty\"", rules.street_num_penalty);
        parse_double("\"postal_bonus\"", rules.postal_bonus);
        parse_double("\"postal_penalty\"", rules.postal_penalty);
        parse_double("\"addr_bonus\"", rules.addr_bonus);
        parse_double("\"addr_penalty\"", rules.addr_penalty);
        parse_int("\"num_threads\"", rules.num_threads);
        int max_cands = (int)rules.max_candidates_per_entity;
        parse_int("\"max_candidates_per_entity\"", max_cands);
        if (max_cands > 0) rules.max_candidates_per_entity = (size_t)max_cands;
        int max_matches = (int)rules.max_matches_per_entity;
        parse_int("\"max_matches_per_entity\"", max_matches);
        if (max_matches > 0) rules.max_matches_per_entity = (size_t)max_matches;
        int max_posting = (int)rules.max_posting_list_size;
        parse_int("\"max_posting_list_size\"", max_posting);
        if (max_posting > 0) rules.max_posting_list_size = (size_t)max_posting;
    }

    int hw_threads = (int)std::thread::hardware_concurrency();
    if (hw_threads > 0) {
        rules.num_threads = std::min(hw_threads, rules.num_threads > 0 ? rules.num_threads : hw_threads);
    } else {
        rules.num_threads = 4;
    }

    // Step 1: Read all S1 records directly in C++
    std::ifstream s1_file(s1_tsv_path);
    if (!s1_file.is_open()) {
        std::string err = "{\"status\": \"error\", \"message\": \"Failed to open S1 TSV file.\"}";
        char* res = new char[err.size() + 1];
        std::strcpy(res, err.c_str());
        return res;
    }

    std::vector<S1Record> s1_records;
    s1_records.reserve(2000000);

    // Group S1 indices by country
    std::unordered_map<std::string, std::vector<size_t>> s1_by_country;

    std::string line;
    std::getline(s1_file, line); // Skip header

    std::vector<std::string_view> fields;
    while (std::getline(s1_file, line)) {
        if (line.empty()) continue;
        split_tsv_fields(line, fields);
        if (fields.empty()) continue;

        std::string id = std::string(fields[0]);
        std::string_view raw_name = (fields.size() > 1) ? fields[1] : "";
        std::string_view raw_addr = (fields.size() > 2) ? fields[2] : "";
        std::string_view raw_country = (fields.size() > 3) ? fields[3] : "UNKNOWN";

        std::string clean_name = normalize_text_cpp_internal(raw_name);
        auto [root_name, _] = parse_legal_suffix_cpp_internal(clean_name);
        std::string country = normalize_country_cpp_internal(raw_country);
        std::string acronym = extract_acronym_cpp_internal(root_name);
        std::vector<std::string> tokens = extract_tokens_cpp_internal(root_name);
        std::string clean_addr = normalize_text_cpp_internal(raw_addr);
        std::vector<std::string> street_nums = extract_street_numbers_cpp_internal(clean_addr);
        std::string postal = extract_postal_code_cpp_internal(clean_addr);
        std::vector<uint32_t> addr_hashes = extract_address_token_hashes_cpp_internal(clean_addr);

        size_t idx = s1_records.size();
        s1_records.push_back({std::move(id), std::move(root_name), std::move(acronym), country, std::move(tokens), std::move(street_nums), std::move(postal), std::move(addr_hashes)});
        s1_by_country[country].push_back(idx);
    }
    s1_file.close();

    size_t total_s1 = s1_records.size();
    std::cerr << "[C++ pipeline] Stage 1/5 complete: normalized " << total_s1
              << " incoming records" << std::endl;

    // Results storage: S1 index -> candidate IDs and matched IDs
    std::vector<std::vector<std::string>> s1_candidates(total_s1);
    std::vector<std::vector<std::string>> s1_matches(total_s1);

    // Step 2a: Count name-token frequencies before building postings.  The old
    // implementation kept the first N rows for every token, which created a
    // severe order bias on the 5M-row target sources.
    std::unordered_map<std::string, CountryTargetIndex> country_target_indices;
    std::unordered_map<std::string, size_t> target_records_by_country;
    size_t total_s2 = 0;
    size_t total_s3 = 0;
    for (const auto& s1 : s1_records) {
        auto& query = country_target_indices[s1.country];
        if (!s1.root_name.empty()) query.query_root_names.insert(s1.root_name);
        for (const auto& tok : s1.tokens) query.query_tokens.insert(tok);
        if (!s1.postal_code.empty()) query.query_postal_codes.insert(s1.postal_code);
        for (uint32_t h : s1.addr_hashes) query.query_address_hashes.insert(h);
    }

    for (const char* target_path : {s2_tsv_path, s3_tsv_path}) {
        uint8_t src_type = (std::strcmp(target_path, s2_tsv_path) == 0) ? 2 : 3;
        std::ifstream tgt_file(target_path);
        if (!tgt_file.is_open()) continue;

        std::string t_line;
        std::getline(tgt_file, t_line); // Skip header
        while (std::getline(tgt_file, t_line)) {
            if (t_line.empty()) continue;
            if (src_type == 2) {
                total_s2++;
            } else {
                total_s3++;
            }
            split_tsv_fields(t_line, fields);
            if (fields.empty()) continue;

            std::string_view raw_c = (fields.size() > 3) ? fields[3] : "UNKNOWN";
            std::string c_norm = normalize_country_cpp_internal(raw_c);
            auto c_it = country_target_indices.find(c_norm);
            if (c_it == country_target_indices.end()) continue;
            ++target_records_by_country[c_norm];

            CountryTargetIndex& target_index = c_it->second;

            std::string_view raw_name = (fields.size() > 1) ? fields[1] : "";
            std::string clean_name = normalize_text_cpp_internal(raw_name);
            auto [root_name, _] = parse_legal_suffix_cpp_internal(clean_name);
            std::vector<std::string> tokens = extract_tokens_cpp_internal(root_name);
            auto& frequencies = (src_type == 2) ? target_index.token_frequency_s2 : target_index.token_frequency_s3;
            for (const auto& tok : tokens) {
                if (target_index.query_tokens.find(tok) != target_index.query_tokens.end()) {
                    ++frequencies[tok];
                }
            }
        }
        tgt_file.close();
    }

    // Use bounded country-partition reservations.  The old fixed one-million-
    // row reservation per country created large unused allocations and
    // increased memory pressure before scoring began.
    for (const auto& cp : s1_by_country) {
        auto count_it = target_records_by_country.find(cp.first);
        if (count_it != target_records_by_country.end()) {
            // The final index keeps only rows with at least one query key, so
            // the raw country count is an upper bound.  Use a bounded warm
            // reserve to avoid both giant allocations and repeated growth.
            country_target_indices[cp.first].records.reserve(
                std::min<size_t>(count_it->second, 250000)
            );
        }
    }

    std::cerr << "[C++ pipeline] Stage 2/5 complete: counted target blocking-key frequencies"
              << " (S2=" << total_s2 << ", S3=" << total_s3 << ")" << std::endl;

    // Step 2b: Index S2 and S3 in a second pass.  Rare tokens are retained in
    // full, so a matching record is never lost merely because it appears late
    // in a source file.  Address keys provide a second route for DBA/rebrand
    // records whose names share no token with Source 1.
    for (const char* target_path : {s2_tsv_path, s3_tsv_path}) {
        uint8_t src_type = (std::strcmp(target_path, s2_tsv_path) == 0) ? 2 : 3;
        std::ifstream tgt_file(target_path);
        if (!tgt_file.is_open()) continue;

        std::string t_line;
        std::getline(tgt_file, t_line); // Skip header
        while (std::getline(tgt_file, t_line)) {
            if (t_line.empty()) continue;
            split_tsv_fields(t_line, fields);
            if (fields.empty()) continue;

            std::string_view raw_c = (fields.size() > 3) ? fields[3] : "UNKNOWN";
            std::string c_norm = normalize_country_cpp_internal(raw_c);
            auto c_it = country_target_indices.find(c_norm);
            if (c_it == country_target_indices.end()) continue;

            CountryTargetIndex& target_index = c_it->second;
            std::string id = std::string(fields[0]);
            std::string_view raw_name = (fields.size() > 1) ? fields[1] : "";
            std::string_view raw_addr = (fields.size() > 2) ? fields[2] : "";
            std::string clean_name = normalize_text_cpp_internal(raw_name);
            auto [root_name, _] = parse_legal_suffix_cpp_internal(clean_name);
            std::string acronym = extract_acronym_cpp_internal(root_name);
            std::vector<std::string> tokens = extract_tokens_cpp_internal(root_name);
            std::string clean_addr = normalize_text_cpp_internal(raw_addr);
            std::vector<std::string> street_nums = extract_street_numbers_cpp_internal(clean_addr);
            std::string postal = extract_postal_code_cpp_internal(clean_addr);
            std::vector<uint32_t> addr_hashes = extract_address_token_hashes_cpp_internal(clean_addr);

            auto& frequencies = (src_type == 2) ? target_index.token_frequency_s2 : target_index.token_frequency_s3;

            bool has_exact_name_key = !root_name.empty() &&
                target_index.query_root_names.find(root_name) != target_index.query_root_names.end();
            bool has_token_key = false;
            for (const auto& tok : tokens) {
                auto freq_it = frequencies.find(tok);
                if (target_index.query_tokens.find(tok) != target_index.query_tokens.end() &&
                    freq_it != frequencies.end() && freq_it->second <= rules.max_posting_list_size) {
                    has_token_key = true;
                    break;
                }
            }
            bool has_postal_key = !postal.empty() &&
                target_index.query_postal_codes.find(postal) != target_index.query_postal_codes.end();
            bool has_address_key = false;
            for (uint32_t h : addr_hashes) {
                if (target_index.query_address_hashes.find(h) != target_index.query_address_hashes.end()) {
                    has_address_key = true;
                    break;
                }
            }

            // The scorer can only reach a target through one of these blocking
            // routes.  Do not retain millions of unreachable target records.
            if (!has_exact_name_key && !has_token_key && !has_postal_key && !has_address_key) {
                continue;
            }

            uint32_t rec_idx = (uint32_t)target_index.records.size();

            // Exact roots are high-signal. Keep all of them unless the name is
            // genuinely generic; the token frequency filter below handles the
            // latter without arbitrary row-order truncation.
            if (has_exact_name_key) {
                auto& list = (src_type == 2) ? target_index.exact_name_map_s2[root_name] : target_index.exact_name_map_s3[root_name];
                if (list.size() < std::max<size_t>(rules.max_exact_posting_size, 500)) {
                    list.push_back(rec_idx);
                }
            }
            for (const auto& tok : tokens) {
                auto freq_it = frequencies.find(tok);
                if (target_index.query_tokens.find(tok) != target_index.query_tokens.end() &&
                    freq_it != frequencies.end() && freq_it->second <= rules.max_posting_list_size) {
                    auto& post = (src_type == 2) ? target_index.token_inverted_index_s2[tok] : target_index.token_inverted_index_s3[tok];
                    post.push_back(rec_idx);
                }
            }
            if (!postal.empty() && target_index.query_postal_codes.find(postal) != target_index.query_postal_codes.end()) {
                auto& post = (src_type == 2) ? target_index.postal_index_s2[postal] : target_index.postal_index_s3[postal];
                if (post.size() < 2000) post.push_back(rec_idx);
            }
            auto& address_index = (src_type == 2) ? target_index.address_hash_index_s2 : target_index.address_hash_index_s3;
            for (uint32_t h : addr_hashes) {
                if (target_index.query_address_hashes.find(h) == target_index.query_address_hashes.end()) continue;
                auto& post = address_index[h];
                if (post.size() < 1000) post.push_back(rec_idx);
            }

            target_index.records.push_back({std::move(id), std::move(root_name), std::move(acronym), std::move(tokens), std::move(street_nums), std::move(postal), std::move(addr_hashes), src_type});
        }
        tgt_file.close();
    }

    size_t indexed_target_records = 0;
    for (const auto& entry : country_target_indices) {
        indexed_target_records += entry.second.records.size();
    }
    std::cerr << "[C++ pipeline] Stage 3/5 complete: built country-partitioned candidate indexes"
              << " (retained " << indexed_target_records << " keyed target records)"
              << std::endl;

    // Step 3: Multi-threaded matching of S1 records per country
    std::atomic<size_t> processed_s1{0};
    std::cerr << "[C++ pipeline] Stage 4/5 started: candidate sieve, fuzzy scoring, and resolution"
              << std::endl;
    for (const auto& cp : s1_by_country) {
        const std::string& country = cp.first;
        const std::vector<size_t>& s1_indices = cp.second;
        auto c_it = country_target_indices.find(country);
        if (c_it == country_target_indices.end()) continue;

        const CountryTargetIndex& target_index = c_it->second;
        size_t n_targets = target_index.records.size();
        if (n_targets == 0) continue;

        size_t n_country_s1 = s1_indices.size();
        size_t n_threads = std::max(1, rules.num_threads);
        size_t chunk_size = (n_country_s1 + n_threads - 1) / n_threads;

        std::vector<std::thread> workers;
        for (size_t t = 0; t < n_threads; ++t) {
            size_t start_idx = t * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, n_country_s1);
            if (start_idx >= end_idx) continue;

            workers.emplace_back([&, start_idx, end_idx, n_targets]() {
                // Thread-local scratch buffers (allocated ONCE per thread)
                std::vector<int> overlap_counts(n_targets, 0);
                std::vector<uint32_t> touched_s2;
                touched_s2.reserve(256);
                std::vector<uint32_t> touched_s3;
                touched_s3.reserve(256);

                std::vector<std::pair<uint32_t, double>> s2_scores;
                s2_scores.reserve(64);
                std::vector<std::pair<uint32_t, double>> s3_scores;
                s3_scores.reserve(64);

                for (size_t k = start_idx; k < end_idx; ++k) {
                    size_t s1_idx = s1_indices[k];
                    const auto& s1 = s1_records[s1_idx];
                    s2_scores.clear();
                    s3_scores.clear();
                    touched_s2.clear();
                    touched_s3.clear();

                    // ========================================================
                    // 1. Search Source 2 Candidates
                    // ========================================================
                    // Address keys are an independent retrieval route for
                    // rebrands/DBAs whose names have no shared token.
                    if (!s1.postal_code.empty()) {
                        auto it = target_index.postal_index_s2.find(s1.postal_code);
                        if (it != target_index.postal_index_s2.end()) {
                            for (uint32_t tid : it->second) {
                                if (overlap_counts[tid] == 0) touched_s2.push_back(tid);
                                if (overlap_counts[tid] == 0) overlap_counts[tid] = -1;
                            }
                        }
                    }
                    for (uint32_t h : s1.addr_hashes) {
                        auto it = target_index.address_hash_index_s2.find(h);
                        if (it != target_index.address_hash_index_s2.end()) {
                            for (uint32_t tid : it->second) {
                                if (overlap_counts[tid] == 0) touched_s2.push_back(tid);
                                if (overlap_counts[tid] == 0) overlap_counts[tid] = -1;
                            }
                        }
                    }
                    if (!s1.root_name.empty()) {
                        auto it = target_index.exact_name_map_s2.find(s1.root_name);
                        if (it != target_index.exact_name_map_s2.end()) {
                            for (uint32_t tid : it->second) {
                                if (overlap_counts[tid] == 0) touched_s2.push_back(tid);
                                overlap_counts[tid] = 9999;
                            }
                        }
                    }
                    for (const auto& tok : s1.tokens) {
                        auto it = target_index.token_inverted_index_s2.find(tok);
                        if (it != target_index.token_inverted_index_s2.end()) {
                            for (uint32_t tid : it->second) {
                                if (overlap_counts[tid] == 0) {
                                    touched_s2.push_back(tid);
                                    overlap_counts[tid] = 1;
                                } else if (overlap_counts[tid] == -1) {
                                    overlap_counts[tid] = 1;
                                } else if (overlap_counts[tid] > 0 && overlap_counts[tid] < 9900) {
                                    overlap_counts[tid]++;
                                }
                            }
                        }
                    }

                    // Lambda for computing rich multi-field score between s1 and target
                    auto compute_score = [&](const S1Record& s1_rec, const TargetRecord& target, int overlap) -> double {
                        int name_overlap = std::max(0, overlap);
                        size_t name_union_size = s1_rec.tokens.size() + target.tokens.size() - name_overlap;
                        double name_jaccard = (name_union_size > 0) ? ((double)name_overlap / (double)name_union_size) : 0.0;
                        double name_jw = fast_jaro_winkler(s1_rec.root_name.c_str(), target.root_name.c_str());
                        double base_name = 0.0;
                        if (overlap >= 9999 || (!s1_rec.root_name.empty() && s1_rec.root_name == target.root_name)) {
                            base_name = 1.0;
                        } else {
                            // Address-retrieved pairs must still receive a
                            // character-level name signal; the previous code
                            // forced JW to zero when no name token overlapped.
                            if (name_jaccard >= 0.75 || name_jw >= 0.92) {
                                base_name = std::max(name_jaccard, name_jw);
                            } else if (name_jaccard >= 0.50 || name_jw >= 0.85) {
                                base_name = 0.40 * name_jaccard + 0.60 * name_jw;
                            } else {
                                base_name = 0.35 * std::max(name_jaccard, name_jw);
                            }
                        }

                        // Structured Numeric Street Number Check
                        int sn_agree = 0;
                        if (!s1_rec.street_nums.empty() && !target.street_nums.empty()) {
                            bool has_match = false;
                            for (const auto& s1_sn : s1_rec.street_nums) {
                                for (const auto& t_sn : target.street_nums) {
                                    if (s1_sn == t_sn) { has_match = true; break; }
                                }
                                if (has_match) break;
                            }
                            sn_agree = has_match ? 1 : -1;
                        }

                        // Structured Postal Code Check
                        int postal_agree = 0;
                        if (!s1_rec.postal_code.empty() && !target.postal_code.empty()) {
                            postal_agree = (s1_rec.postal_code == target.postal_code) ? 1 : -1;
                        }

                        // Structured Address / Locality Token Check
                        int addr_agree = 0;
                        int addr_overlap = 0;
                        if (!s1_rec.addr_hashes.empty() && !target.addr_hashes.empty()) {
                            for (uint32_t h1 : s1_rec.addr_hashes) {
                                for (uint32_t h2 : target.addr_hashes) {
                                    if (h1 == h2) { addr_overlap++; break; }
                                }
                            }
                            if (addr_overlap >= 1) {
                                addr_agree = 1;
                            } else {
                                addr_agree = -1; // Contradictory localities
                            }
                        }

                        size_t addr_union = s1_rec.addr_hashes.size() + target.addr_hashes.size() - (size_t)addr_overlap;
                        double addr_jaccard = (addr_union > 0) ? ((double)addr_overlap / (double)addr_union) : 0.0;

                        // Missing Address Indicator: check if either record lacks address data
                        bool s1_has_addr = !s1_rec.street_nums.empty() || !s1_rec.postal_code.empty() || !s1_rec.addr_hashes.empty();
                        bool tgt_has_addr = !target.street_nums.empty() || !target.postal_code.empty() || !target.addr_hashes.empty();
                        bool missing_addr = (!s1_has_addr || !tgt_has_addr);

                        // Composite Discriminator Score with Singleton Protections
                        double score = base_name;
                        if (postal_agree == 1) score += rules.postal_bonus;
                        else if (postal_agree == -1) score += rules.postal_penalty;

                        if (sn_agree == 1) score += rules.street_num_bonus;
                        else if (sn_agree == -1) score += rules.street_num_penalty;

                        if (addr_agree == 1) score += rules.addr_bonus;
                        else if (addr_agree == -1) score += rules.addr_penalty;

                        // Graceful degradation when address information is missing
                        if (missing_addr) {
                            score -= 0.15;
                        }

                        // Hard penalties for compound contradictions (prevents spurious singleton matches)
                        if (postal_agree == -1 && addr_agree == -1) score -= 0.30;
                        if (sn_agree == -1 && addr_agree == -1) score -= 0.30;
                        if (postal_agree == -1 && sn_agree == -1) score -= 0.30;

                        // Uncorroborated Non-Exact Name Penalty
                        // If root names are not virtually identical (< 0.98) and there is zero positive corroboration
                        if (base_name < 0.98 && postal_agree <= 0 && sn_agree <= 0 && addr_agree <= 0) {
                            score -= 0.35;
                        }

                        // Full-density validation showed that exact/common
                        // business names are not safe by themselves.  Require
                        // both a meaningful address agreement and a strong
                        // name signal for a production match.  This preserves
                        // DBA/rebrand links while rejecting same-name entities
                        // at different locations.
                        bool address_consistent = addr_jaccard >= 0.30 && postal_agree >= 0 && sn_agree >= 0;
                        bool name_consistent = name_jw >= 0.90 || name_jaccard >= 0.40 || base_name >= 0.98;
                        if (address_consistent && name_consistent) {
                            score = std::max(score, 0.84);
                        } else {
                            score = std::min(score, 0.78);
                        }

                        return std::max(0.0, std::min(1.0, score));
                    };

                    // Score S2 Candidates
                    for (uint32_t tid : touched_s2) {
                        int overlap = overlap_counts[tid];
                        const auto& target = target_index.records[tid];
                        double score = compute_score(s1, target, overlap);
                        if (score >= rules.min_candidate_score) {
                            s2_scores.emplace_back(tid, score);
                        }
                    }

                    // Reset touched S2 in scratch buffer
                    for (uint32_t tid : touched_s2) overlap_counts[tid] = 0;

                    // ========================================================
                    // 2. Search Source 3 Candidates
                    // ========================================================
                    if (!s1.postal_code.empty()) {
                        auto it = target_index.postal_index_s3.find(s1.postal_code);
                        if (it != target_index.postal_index_s3.end()) {
                            for (uint32_t tid : it->second) {
                                if (overlap_counts[tid] == 0) touched_s3.push_back(tid);
                                if (overlap_counts[tid] == 0) overlap_counts[tid] = -1;
                            }
                        }
                    }
                    for (uint32_t h : s1.addr_hashes) {
                        auto it = target_index.address_hash_index_s3.find(h);
                        if (it != target_index.address_hash_index_s3.end()) {
                            for (uint32_t tid : it->second) {
                                if (overlap_counts[tid] == 0) touched_s3.push_back(tid);
                                if (overlap_counts[tid] == 0) overlap_counts[tid] = -1;
                            }
                        }
                    }
                    if (!s1.root_name.empty()) {
                        auto it = target_index.exact_name_map_s3.find(s1.root_name);
                        if (it != target_index.exact_name_map_s3.end()) {
                            for (uint32_t tid : it->second) {
                                if (overlap_counts[tid] == 0) touched_s3.push_back(tid);
                                overlap_counts[tid] = 9999;
                            }
                        }
                    }
                    for (const auto& tok : s1.tokens) {
                        auto it = target_index.token_inverted_index_s3.find(tok);
                        if (it != target_index.token_inverted_index_s3.end()) {
                            for (uint32_t tid : it->second) {
                                if (overlap_counts[tid] == 0) {
                                    touched_s3.push_back(tid);
                                    overlap_counts[tid] = 1;
                                } else if (overlap_counts[tid] == -1) {
                                    overlap_counts[tid] = 1;
                                } else if (overlap_counts[tid] > 0 && overlap_counts[tid] < 9900) {
                                    overlap_counts[tid]++;
                                }
                            }
                        }
                    }

                    // Score S3 Candidates
                    for (uint32_t tid : touched_s3) {
                        int overlap = overlap_counts[tid];
                        const auto& target = target_index.records[tid];
                        double score = compute_score(s1, target, overlap);
                        if (score >= rules.min_candidate_score) {
                            s3_scores.emplace_back(tid, score);
                        }
                    }

                    // Reset touched S3 in scratch buffer
                    for (uint32_t tid : touched_s3) overlap_counts[tid] = 0;

                    // Deterministic Candidate Comparator: score descending, then target entity ID ascending
                    // Strict weak ordering: exact comparison prevents transitivity violations in std::sort
                    auto cand_cmp = [&](const std::pair<uint32_t, double>& a, const std::pair<uint32_t, double>& b) {
                        if (a.second != b.second) {
                            return a.second > b.second;
                        }
                        return target_index.records[a.first].id < target_index.records[b.first].id;
                    };

                    std::sort(s2_scores.begin(), s2_scores.end(), cand_cmp);
                    std::sort(s3_scores.begin(), s3_scores.end(), cand_cmp);

                    // ========================================================
                    // 3. Tripartite Cross-Verification & Transitive Candidate Expansion
                    //    (S1 <-> S2, S2 <-> S3, S1 <-> S3)
                    // ========================================================
                    if (!s2_scores.empty() && s2_scores[0].second >= 0.70) {
                        const auto& top_s2 = target_index.records[s2_scores[0].first];
                        
                        // Transitive candidate discovery from S2 to S3
                        if (!top_s2.root_name.empty()) {
                            auto it = target_index.exact_name_map_s3.find(top_s2.root_name);
                            if (it != target_index.exact_name_map_s3.end()) {
                                for (uint32_t tid : it->second) {
                                    bool exists = false;
                                    for (const auto& p : s3_scores) {
                                        if (p.first == tid) { exists = true; break; }
                                    }
                                    if (!exists) {
                                        const auto& s3_rec = target_index.records[tid];
                                        int actual_overlap = 0;
                                        if (!s1.root_name.empty() && s1.root_name == s3_rec.root_name) {
                                            actual_overlap = 9999;
                                        } else {
                                            for (const auto& tok1 : s1.tokens) {
                                                for (const auto& tok2 : s3_rec.tokens) {
                                                    if (tok1 == tok2) { actual_overlap++; break; }
                                                }
                                            }
                                        }
                                        double trans_score = compute_score(s1, s3_rec, actual_overlap);
                                        if (trans_score >= rules.min_candidate_score) {
                                            s3_scores.emplace_back(tid, trans_score);
                                        }
                                    }
                                }
                            }
                        }

                        // Tripartite mutual agreement boost for S3 candidates (requires name AND address)
                        for (auto& s3_pair : s3_scores) {
                            const auto& s3_rec = target_index.records[s3_pair.first];
                            bool name_agree = (!top_s2.root_name.empty() && top_s2.root_name == s3_rec.root_name);
                            bool postal_agree = (!top_s2.postal_code.empty() && !s3_rec.postal_code.empty() && top_s2.postal_code == s3_rec.postal_code);
                            bool sn_agree = false;
                            if (!top_s2.street_nums.empty() && !s3_rec.street_nums.empty()) {
                                for (const auto& sn2 : top_s2.street_nums) {
                                    for (const auto& sn3 : s3_rec.street_nums) {
                                        if (sn2 == sn3) { sn_agree = true; break; }
                                    }
                                    if (sn_agree) break;
                                }
                            }
                            if (name_agree && (postal_agree || sn_agree)) {
                                s3_pair.second = std::min(1.0, s3_pair.second + 0.15); // Tripartite confirmation boost
                            }
                        }

                        // Re-sort S3 scores after boost with deterministic comparator
                        std::sort(s3_scores.begin(), s3_scores.end(), cand_cmp);
                    }

                    // Transitive expansion from S3 to S2 if S3 had a high-confidence match
                    if (!s3_scores.empty() && s3_scores[0].second >= 0.70) {
                        const auto& top_s3 = target_index.records[s3_scores[0].first];
                        if (!top_s3.root_name.empty()) {
                            auto it = target_index.exact_name_map_s2.find(top_s3.root_name);
                            if (it != target_index.exact_name_map_s2.end()) {
                                for (uint32_t tid : it->second) {
                                    bool exists = false;
                                    for (const auto& p : s2_scores) {
                                        if (p.first == tid) { exists = true; break; }
                                    }
                                    if (!exists) {
                                        const auto& s2_rec = target_index.records[tid];
                                        int actual_overlap = 0;
                                        if (!s1.root_name.empty() && s1.root_name == s2_rec.root_name) {
                                            actual_overlap = 9999;
                                        } else {
                                            for (const auto& tok1 : s1.tokens) {
                                                for (const auto& tok2 : s2_rec.tokens) {
                                                    if (tok1 == tok2) { actual_overlap++; break; }
                                                }
                                            }
                                        }
                                        double trans_score = compute_score(s1, s2_rec, actual_overlap);
                                        if (trans_score >= rules.min_candidate_score) {
                                            s2_scores.emplace_back(tid, trans_score);
                                        }
                                    }
                                }
                            }
                        }

                        // Tripartite mutual agreement boost for S2 candidates
                        for (auto& s2_pair : s2_scores) {
                            const auto& s2_rec = target_index.records[s2_pair.first];
                            bool name_agree = (!top_s3.root_name.empty() && top_s3.root_name == s2_rec.root_name);
                            bool postal_agree = (!top_s3.postal_code.empty() && !s2_rec.postal_code.empty() && top_s3.postal_code == s2_rec.postal_code);
                            bool sn_agree = false;
                            if (!top_s3.street_nums.empty() && !s2_rec.street_nums.empty()) {
                                for (const auto& sn3 : top_s3.street_nums) {
                                    for (const auto& sn2 : s2_rec.street_nums) {
                                        if (sn3 == sn2) { sn_agree = true; break; }
                                    }
                                    if (sn_agree) break;
                                }
                            }
                            if (name_agree && (postal_agree || sn_agree)) {
                                s2_pair.second = std::min(1.0, s2_pair.second + 0.15); // Tripartite confirmation boost
                            }
                        }

                        std::sort(s2_scores.begin(), s2_scores.end(), cand_cmp);
                    }

                    // ========================================================
                    // 4. Candidate & Match Selection (Disciplined, Canonical & Sorted)
                    // ========================================================
                    // Hard Rule: NEVER discard a valid match (score >= match_threshold) due to candidate caps!
                    std::vector<std::string> s2_matches;
                    std::vector<std::string> s3_matches;
                    std::unordered_set<uint32_t> match_tids;

                    size_t s2_match_count = 0;
                    for (const auto& sc : s2_scores) {
                        if (sc.second >= rules.match_threshold) {
                            s2_matches.push_back(target_index.records[sc.first].id);
                            match_tids.insert(sc.first);
                            if (++s2_match_count >= rules.max_matches_per_entity) break;
                        }
                    }
                    size_t s3_match_count = 0;
                    for (const auto& sc : s3_scores) {
                        if (sc.second >= rules.match_threshold) {
                            s3_matches.push_back(target_index.records[sc.first].id);
                            match_tids.insert(sc.first);
                            if (++s3_match_count >= rules.max_matches_per_entity) break;
                        }
                    }

                    // Sort matches canonically: S2 matches ascending, S3 matches ascending
                    std::sort(s2_matches.begin(), s2_matches.end());
                    std::sort(s3_matches.begin(), s3_matches.end());

                    std::vector<std::string> matches;
                    matches.reserve(s2_matches.size() + s3_matches.size());
                    for (const auto& mid : s2_matches) matches.push_back(mid);
                    for (const auto& mid : s3_matches) matches.push_back(mid);

                    // Candidate Selection:
                    // Hard Rule: Every match MUST be in candidates (strict subset property).
                    // If matches count < rules.max_candidates_per_entity, fill remaining slots
                    // with the highest-scoring non-match candidates from both sources.
                    std::vector<std::string> s2_cands = s2_matches;
                    std::vector<std::string> s3_cands = s3_matches;

                    size_t current_cands = matches.size();
                    if (current_cands < rules.max_candidates_per_entity) {
                        std::vector<std::pair<uint32_t, double>> non_matches;
                        for (const auto& sc : s2_scores) {
                            if (match_tids.find(sc.first) == match_tids.end()) {
                                non_matches.push_back(sc);
                            }
                        }
                        for (const auto& sc : s3_scores) {
                            if (match_tids.find(sc.first) == match_tids.end()) {
                                non_matches.push_back(sc);
                            }
                        }
                        std::sort(non_matches.begin(), non_matches.end(), cand_cmp);

                        size_t needed = rules.max_candidates_per_entity - current_cands;
                        size_t take = std::min(needed, non_matches.size());
                        for (size_t i = 0; i < take; ++i) {
                            const auto& rec = target_index.records[non_matches[i].first];
                            if (rec.id.rfind("S2-", 0) == 0) {
                                s2_cands.push_back(rec.id);
                            } else {
                                s3_cands.push_back(rec.id);
                            }
                        }
                    }

                    // Sort candidates canonically: S2 candidates ascending, S3 candidates ascending
                    std::sort(s2_cands.begin(), s2_cands.end());
                    std::sort(s3_cands.begin(), s3_cands.end());

                    std::vector<std::string> cands;
                    cands.reserve(s2_cands.size() + s3_cands.size());
                    for (const auto& cid : s2_cands) cands.push_back(cid);
                    for (const auto& cid : s3_cands) cands.push_back(cid);

                    s1_candidates[s1_idx] = std::move(cands);
                    s1_matches[s1_idx] = std::move(matches);

                    size_t completed = processed_s1.fetch_add(1) + 1;
                    if (completed % 100000 == 0 || completed == total_s1) {
                        std::cerr << "[C++ pipeline] Resolved " << completed << "/"
                                  << total_s1 << " incoming records" << std::endl;
                    }
                }
            });
        }

        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }

    std::cerr << "[C++ pipeline] Stage 4/5 complete: resolved all incoming records"
              << std::endl;
    // Step 5: Stream serializing to output TSVs in strictly sorted S1 order
    std::cerr << "[C++ pipeline] Stage 5/5 started: writing candidate and match TSVs"
              << std::endl;
    std::ofstream cand_out(candidate_output_path);
    std::ofstream match_out(matching_output_path);

    if (!cand_out.is_open() || !match_out.is_open()) {
        std::string err = "{\"status\": \"error\", \"message\": \"Failed to open output TSV files for writing.\"}";
        char* res = new char[err.size() + 1];
        std::strcpy(res, err.c_str());
        return res;
    }

    cand_out << "source1_entity_id\tcandidate_entity_ids\n";
    match_out << "source1_entity_id\tmatched_entity_ids\n";

    // Strictly sort output rows alphabetically by source1_entity_id
    std::vector<size_t> sorted_s1_indices(total_s1);
    std::iota(sorted_s1_indices.begin(), sorted_s1_indices.end(), 0);
    std::sort(sorted_s1_indices.begin(), sorted_s1_indices.end(), [&](size_t a, size_t b) {
        return s1_records[a].id < s1_records[b].id;
    });

    size_t total_cand_links = 0;
    size_t total_matched_links = 0;
    size_t singletons_count = 0;

    for (size_t idx : sorted_s1_indices) {
        const std::string& s1_id = s1_records[idx].id;
        const auto& cands = s1_candidates[idx];
        const auto& matches = s1_matches[idx];

        total_cand_links += cands.size();
        total_matched_links += matches.size();
        if (matches.empty()) {
            singletons_count++;
        }

        // Write candidate pairs TSV
        cand_out << s1_id << "\t";
        for (size_t c = 0; c < cands.size(); ++c) {
            if (c > 0) cand_out << ",";
            cand_out << cands[c];
        }
        cand_out << "\n";

        // Write matching results TSV
        match_out << s1_id << "\t";
        for (size_t m = 0; m < matches.size(); ++m) {
            if (m > 0) match_out << ",";
            match_out << matches[m];
        }
        match_out << "\n";
    }

    cand_out.close();
    match_out.close();
    std::cerr << "[C++ pipeline] Stage 5/5 complete: output files written" << std::endl;

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    // Construct JSON summary response
    std::ostringstream json_ss;
    json_ss << "{"
            << "\"status\": \"success\","
            << "\"total_s1\": " << total_s1 << ","
            << "\"total_s2\": " << total_s2 << ","
            << "\"total_s3\": " << total_s3 << ","
            << "\"total_candidates\": " << total_cand_links << ","
            << "\"total_matches\": " << total_matched_links << ","
            << "\"singletons\": " << singletons_count << ","
            << "\"elapsed_seconds\": " << elapsed_sec << ","
            << "\"threads_used\": " << rules.num_threads << ","
            << "\"candidate_file\": \"" << candidate_output_path << "\","
            << "\"matching_file\": \"" << matching_output_path << "\""
            << "}";

    std::string json_str = json_ss.str();
    char* result_cstr = new char[json_str.size() + 1];
    std::strcpy(result_cstr, json_str.c_str());
    return result_cstr;
}

// Single-entity interactive query resolution via C++
EXPORT_API const char* query_entity_resolution_cpp(
    const char* s1_id,
    const char* raw_name,
    const char* raw_addr,
    const char* raw_country,
    const char* s2_tsv_path,
    const char* s3_tsv_path,
    double match_threshold
) {
    if (!s1_id || !raw_name || !s2_tsv_path || !s3_tsv_path) {
        std::string err = "{\"status\": \"error\", \"message\": \"Missing parameters for query.\"}";
        char* res = new char[err.size() + 1];
        std::strcpy(res, err.c_str());
        return res;
    }

    std::string clean_name = normalize_text_cpp_internal(raw_name);
    auto [root_name, _] = parse_legal_suffix_cpp_internal(clean_name);
    std::string country = normalize_country_cpp_internal(raw_country ? raw_country : "UNKNOWN");
    std::vector<std::string> tokens = extract_tokens_cpp_internal(root_name);
    std::string clean_addr = normalize_text_cpp_internal(raw_addr ? raw_addr : "");
    std::vector<std::string> street_nums = extract_street_numbers_cpp_internal(clean_addr);
    std::string postal = extract_postal_code_cpp_internal(clean_addr);
    std::vector<uint32_t> addr_hashes = extract_address_token_hashes_cpp_internal(clean_addr);

    std::unordered_set<std::string> s1_tok_set(tokens.begin(), tokens.end());
    std::unordered_set<std::string> s1_snum_set(street_nums.begin(), street_nums.end());

    std::vector<std::pair<std::string, double>> candidates;

    // Scan S2 and S3 for matching country
    std::vector<std::string_view> fields;
    for (const char* target_path : {s2_tsv_path, s3_tsv_path}) {
        std::ifstream file(target_path);
        if (!file.is_open()) continue;

        std::string line;
        std::getline(file, line); // Skip header
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            split_tsv_fields(line, fields);
            if (fields.empty()) continue;

            std::string_view tc = (fields.size() > 3) ? fields[3] : "UNKNOWN";
            if (normalize_country_cpp_internal(tc) != country) continue;

            std::string t_id = std::string(fields[0]);
            std::string_view t_name = (fields.size() > 1) ? fields[1] : "";
            std::string_view t_addr = (fields.size() > 2) ? fields[2] : "";

            std::string t_clean = normalize_text_cpp_internal(t_name);
            auto [t_root, _t_leg] = parse_legal_suffix_cpp_internal(t_clean);

            double score = 0.0;
            std::string t_clean_addr = normalize_text_cpp_internal(t_addr);
            std::vector<std::string> t_snums = extract_street_numbers_cpp_internal(t_clean_addr);
            std::string t_postal = extract_postal_code_cpp_internal(t_clean_addr);
            std::vector<uint32_t> t_addr_hashes = extract_address_token_hashes_cpp_internal(t_clean_addr);

            int sn_agree = 0;
            if (!s1_snum_set.empty() && !t_snums.empty()) {
                bool matched = false;
                for (const auto& ts : t_snums) {
                    if (s1_snum_set.find(ts) != s1_snum_set.end()) { matched = true; break; }
                }
                sn_agree = matched ? 1 : -1;
            }

            int postal_agree = 0;
            if (!postal.empty() && !t_postal.empty()) {
                postal_agree = (postal == t_postal) ? 1 : -1;
            }

            int addr_overlap = 0;
            if (!addr_hashes.empty() && !t_addr_hashes.empty()) {
                for (uint32_t h1 : addr_hashes) {
                    for (uint32_t h2 : t_addr_hashes) {
                        if (h1 == h2) { addr_overlap++; break; }
                    }
                }
            }
            int addr_agree = 0;
            if (addr_hashes.size() >= 2 && t_addr_hashes.size() >= 2) {
                addr_agree = (addr_overlap >= 1) ? 1 : -1;
            } else if (addr_overlap >= 1) {
                addr_agree = 1;
            }

            if (!root_name.empty() && root_name == t_root) {
                score = 1.0;
                if (postal_agree == 1) score += 0.15;
                else if (postal_agree == -1) score -= 0.35;
                if (sn_agree == 1) score += 0.15;
                else if (sn_agree == -1) score -= 0.35;
                if (addr_agree == 1) score += 0.15;
                else if (addr_agree == -1) score -= 0.35;
                if (postal_agree == -1 && addr_agree == -1) score -= 0.25;
                score = std::max(0.0, std::min(1.0, score));
            } else {
                std::string t_acronym = extract_acronym_cpp_internal(t_root);
                std::string s1_acronym = extract_acronym_cpp_internal(root_name);
                bool is_acronym_match = (!s1_acronym.empty() && s1_acronym == t_root) || (!t_acronym.empty() && t_acronym == root_name);

                std::vector<std::string> t_tokens = extract_tokens_cpp_internal(t_root);
                int overlap = 0;
                for (const auto& tok : t_tokens) {
                    if (s1_tok_set.find(tok) != s1_tok_set.end()) overlap++;
                }
                if (overlap > 0 || is_acronym_match) {
                    size_t union_len = s1_tok_set.size();
                    for (const auto& tt : t_tokens) {
                        if (s1_tok_set.find(tt) == s1_tok_set.end()) union_len++;
                    }
                    double jaccard = (union_len > 0) ? ((double)overlap / (double)union_len) : 0.0;
                    double jw = fast_jaro_winkler(root_name.c_str(), t_root.c_str());
                    double base_name = is_acronym_match ? 0.88 : std::max(jaccard, jw);

                    score = base_name;
                    if (postal_agree == 1) score += 0.15;
                    else if (postal_agree == -1) score -= 0.35;

                    if (sn_agree == 1) score += 0.15;
                    else if (sn_agree == -1) score -= 0.35;

                    if (addr_agree == 1) score += 0.15;
                    else if (addr_agree == -1) score -= 0.35;

                    if (postal_agree == -1 && addr_agree == -1) score -= 0.25;
                    if (sn_agree == -1 && addr_agree == -1) score -= 0.25;
                    if (postal_agree == -1 && sn_agree == -1) score -= 0.25;
                    score = std::max(0.0, std::min(1.0, score));
                }
            }

            if (score >= 0.35) {
                candidates.emplace_back(std::move(t_id), score);
            }
        }
        file.close();
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    std::ostringstream ss;
    ss << "{\"s1_id\": \"" << s1_id << "\", \"candidates\": [";
    for (size_t i = 0; i < std::min((size_t)25, candidates.size()); ++i) {
        if (i > 0) ss << ",";
        ss << "{\"id\": \"" << candidates[i].first << "\", \"score\": " << candidates[i].second << "}";
    }
    std::vector<std::string> query_matches;
    for (const auto& cand : candidates) {
        if (cand.second >= match_threshold) {
            query_matches.push_back(cand.first);
        }
    }
    std::sort(query_matches.begin(), query_matches.end());

    ss << "], \"matches\": [";
    for (size_t i = 0; i < query_matches.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "\"" << query_matches[i] << "\"";
    }
    ss << "]}";

    std::string res_str = ss.str();
    char* out = new char[res_str.size() + 1];
    std::strcpy(out, res_str.c_str());
    return out;
}

} // extern "C"

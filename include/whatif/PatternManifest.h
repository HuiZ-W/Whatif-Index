#pragma once
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace whatif {
inline std::vector<std::string> manifestCsvRow(const std::string &line) {
  std::vector<std::string> fields;
  std::string value;
  bool quoted = false;
  for (std::size_t i=0;i<line.size();++i) {
    const char c=line[i];
    if (c=='"') {
      if (quoted && i+1<line.size() && line[i+1]=='"') { value+='"'; ++i; }
      else quoted=!quoted;
    } else if (c==',' && !quoted) { fields.push_back(value); value.clear(); }
    else if (c!='\r') value+=c;
  }
  if (quoted) throw std::runtime_error("unterminated manifest CSV quote");
  fields.push_back(value); return fields;
}
inline std::map<std::string,std::vector<std::string>> loadPatternSchemas(const std::filesystem::path &path) {
  std::map<std::string,std::vector<std::string>> schemas;
  if (!std::filesystem::exists(path)) return schemas;
  std::ifstream input(path);
  if(!input) throw std::runtime_error("cannot open pattern manifest");
  std::string line;
  std::getline(input,line);
  auto header=manifestCsvRow(line);
  auto vi=std::find(header.begin(),header.end(),"variable_id"), ri=std::find(header.begin(),header.end(),"roles");
  if(vi==header.end() || ri==header.end()) throw std::runtime_error("manifest requires variable_id and roles");
  const auto v=vi-header.begin(), r=ri-header.begin();
  while(std::getline(input,line)) {
    if(line.empty()) continue;
    auto fields=manifestCsvRow(line);
    if(fields.size()!=header.size() || fields[v].empty()) throw std::runtime_error("invalid manifest row");
    std::istringstream parts(fields[r]); std::string role; std::vector<std::string> roles;
    while(std::getline(parts,role,'|')) { if(role.empty()) throw std::runtime_error("empty manifest role"); roles.push_back(role); }
    std::sort(roles.begin(),roles.end());
    if(roles.empty() || std::adjacent_find(roles.begin(),roles.end())!=roles.end() ||
       !schemas.emplace(fields[v],roles).second) throw std::runtime_error("duplicate/empty pattern schema");
  }
  if(schemas.empty()) throw std::runtime_error("empty pattern manifest");
  return schemas;
}
} // namespace whatif

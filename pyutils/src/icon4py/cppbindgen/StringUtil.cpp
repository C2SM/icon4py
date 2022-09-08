//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "StringUtil.h"
#include <sstream>
#include <numeric>

namespace dawn {

extern std::string decimalToOrdinal(int dec) {
  std::string decimal = std::to_string(dec);
  std::string suffix;

  switch(decimal.back()) {
  case '1':
    suffix = "st";
    break;
  case '2':
    suffix = "nd";
    break;
  case '3':
    suffix = "rd";
    break;
  default:
    suffix = "th";
  }

  if(dec > 10)
    suffix = "th";

  return decimal + suffix;
}

std::string indent(const std::string& string, int amount) {
  // This could probably be done faster (it's not really speed-critical though)
  std::istringstream iss(string);
  std::ostringstream oss;
  std::string spacer(amount, ' ');
  bool firstLine = true;
  for(std::string line; std::getline(iss, line);) {
    if(!firstLine)
      oss << spacer;
    oss << line;
    if(!iss.eof())
      oss << "\n";
    firstLine = false;
  }
  return oss.str();
}

bool equalsLower(const std::string& a, const std::string& b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                    [](char a, char b) { return tolower(a) == tolower(b); });
}

bool startWithLower(std::string str, std::string match) {
  // Convert str to lower case
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  // Convert match to lower case
  std::transform(match.begin(), match.end(), match.begin(), ::tolower);
  if(str.find(match) == 0)
    return true;
  else
    return false;
}

bool endsWithLower(std::string str, std::string match) {
  // Convert str to lower case
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  // Convert match to lower case
  std::transform(match.begin(), match.end(), match.begin(), ::tolower);
  if(str.find(match) + match.size() == str.size())
    return true;
  else
    return false;
}

static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

std::vector<std::string> tokenize(const std::string &str, char separator) {
  std::stringstream ss(str);
  std::vector<std::string> tokens;
  std::string token;
  while (getline(ss, token, separator)) {
    trim(token);
    tokens.push_back(token);
  }
  return tokens;
};

//https://stackoverflow.com/questions/5689003/how-to-implode-a-vector-of-strings-into-a-string-the-elegant-way
std::string join(const std::vector<std::string>& strlist, char separator) {
  if (strlist.empty()) return std::string();

  return std::accumulate(
    std::next(strlist.begin()),
      strlist.end(),
      strlist[0],
      [&separator](auto result, const auto &value) {
          return result + separator + value;
      });
}

} // namespace dawn

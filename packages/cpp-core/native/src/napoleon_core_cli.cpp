#include "napoleon_core.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::uint32_t parse_seed(const std::string& value) {
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value);
  } catch (const std::exception&) {
    throw std::runtime_error("seed must be an integer between 0 and 4294967295");
  }

  if (parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("seed must be an integer between 0 and 4294967295");
  }

  return static_cast<std::uint32_t>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::uint32_t seed = 0;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--seed" && index + 1 < argc) {
        seed = parse_seed(argv[++index]);
      } else if (arg == "--snapshot") {
        continue;
      } else {
        throw std::runtime_error("usage: napoleon_core_cli --snapshot --seed <uint32>");
      }
    }

    napoleon::GameState state = napoleon::create_initial_game(seed);
    for (std::string line; std::getline(std::cin, line);) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      napoleon::apply_action(state, napoleon::parse_action_line(line));
    }

    std::cout << napoleon::canonical_snapshot_json(state) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

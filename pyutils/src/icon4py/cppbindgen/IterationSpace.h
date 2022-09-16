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

#pragma once

#include "LocationType.h"

#include "Assert.h"
#include <exception>
#include <stdexcept>

namespace dawn {
namespace ast {

struct UnstructuredIterationSpace {
  NeighborChain Chain;
  bool IncludeCenter = false;
  std::optional<LocationType> NewSparseRoot;

  UnstructuredIterationSpace(std::vector<LocationType>&& chain, bool includeCenter)
      : Chain(chain), IncludeCenter(includeCenter) {
    if(includeCenter && chain.front() != chain.back()) {
      throw std::logic_error("including center is only allowed if the end "
                             "location is the same as the starting location");
    }
    if(chain.size() == 0) {
      throw std::logic_error("neighbor chain needs to have at least one member");
    }
    if(!chainIsValid()) {
      throw std::logic_error("invalid neighbor chain (repeated element in succession, use "
                             "expaneded notation (e.g. C->C becomes C->E->C\n");
    }
  }
  UnstructuredIterationSpace(std::vector<LocationType>&& chain, bool includeCenter,
                             std::optional<LocationType> newSparseRoot)
      : UnstructuredIterationSpace(std::move(chain), includeCenter) {
    NewSparseRoot = newSparseRoot;
  }
  UnstructuredIterationSpace(std::vector<LocationType>&& chain)
      : UnstructuredIterationSpace(std::move(chain), false) {}

  operator std::vector<LocationType>() const { return Chain; }

  bool chainIsValid() const {
    for(int chainIdx = 0; chainIdx < Chain.size() - 1; chainIdx++) {
      if(Chain[chainIdx] == Chain[chainIdx + 1]) {
        return false;
      }
    }
    return true;
  }

  bool isNewSparse() const { return NewSparseRoot.has_value(); }

  bool operator==(const UnstructuredIterationSpace& other) const {
    return Chain == other.Chain && IncludeCenter == other.IncludeCenter;
  }
};

} // namespace ast
} // namespace dawn

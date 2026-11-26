// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_TENSOR_DUMP_H_
#define IMSRG_OPERATOR_TENSOR_DUMP_H_
/// @file
/// Reader and writer of the raw tensor reference dumps (`# imsrg-tensor-dump
/// raw v0`, `*.tdump`, design/15 "Tensor reference dumps"), the format of the
/// M15 goldens in tests/golden/tensor/. Provisional, like the format: the
/// tensor OperatorDump v2 will replace it.
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"

namespace imsrg::op {

/// Reads a raw tensor dump into an operator on `ms`. Orbits are mapped by (n,
/// l, 2j, 2tz), so `ms` may be larger than the dump's basis. The spec comes
/// from the `spec` record (particle_rank 2 unless no two-body section; the
/// file must say `reduced 1` for non-scalar specs). One-body records set the
/// single element (a, b); two-body records set the stored element of their
/// block (ch_bra <= ch_ket for tensors). Channel labels of the block lines
/// are checked against `ms`.
/// @throws InputError ("<name>:<line>: <message>") on a malformed record, an
///         unknown orbit, a block that the spec does not store, or a ket not
///         in its channel.
/// @throws NotImplementedError for specs `Validate` rejects.
Operator ReadTensorDump(const std::filesystem::path& path,
                        std::shared_ptr<const basis::ModelSpace> ms);

/// Stream version of `ReadTensorDump`; `name` labels error messages.
Operator ReadTensorDump(std::istream& in, std::string_view name,
                        std::shared_ptr<const basis::ModelSpace> ms);

/// Writes `op` in the raw v0 layout (`%.17g`, exact zeros omitted; one-body
/// entries a then b; two-body entries block by block in storage order, bra
/// then ket local index; blocks without a nonzero element omitted).
/// `operator_name` fills the `operator` record.
/// @throws InputError if the file cannot be written.
void WriteTensorDump(const std::filesystem::path& path, const Operator& op,
                     const std::string& operator_name = "op");

/// Stream version of `WriteTensorDump`.
void WriteTensorDump(std::ostream& out, const Operator& op,
                     const std::string& operator_name = "op");

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_TENSOR_DUMP_H_

// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

/// @file
/// The host/device residency state machine of Python operators and the
/// registry of live objects that own device data (design/18 sections 2.2,
/// 4.2, 4.3).

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "imsrg/exec/runtime.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;
namespace {

struct Registry {
  std::mutex mutex;
  std::vector<Releasable*> live;
};

// Heap-allocated and never deleted (it outlives every static destructor).
Registry& GetRegistry() {
  static Registry* r = new Registry;
  return *r;
}

int InitialPolicy() {
  const char* e = std::getenv("IMSRG_RESIDENCY");
  if (e != nullptr && std::strcmp(e, "strict") == 0)
    return static_cast<int>(ResidencyPolicy::kStrict);
  if (e != nullptr && std::strcmp(e, "implicit") == 0)
    return static_cast<int>(ResidencyPolicy::kImplicit);
  return static_cast<int>(ResidencyPolicy::kAuto);
}

std::atomic<int>& PolicyStorage() {
  static std::atomic<int> p{InitialPolicy()};
  return p;
}

}  // namespace

Releasable::Releasable() {
  auto& r = GetRegistry();
  std::lock_guard lock(r.mutex);
  r.live.push_back(this);
}

Releasable::Releasable(const Releasable& /*other*/) : Releasable() {}

Releasable::~Releasable() {
  auto& r = GetRegistry();
  std::lock_guard lock(r.mutex);
  const auto it = std::find(r.live.begin(), r.live.end(), this);
  if (it != r.live.end()) r.live.erase(it);
}

int LiveObjects() noexcept {
  auto& r = GetRegistry();
  std::lock_guard lock(r.mutex);
  return static_cast<int>(
      std::count_if(r.live.begin(), r.live.end(),
                    [](const Releasable* o) { return !o->released; }));
}

namespace {

std::atomic<int> g_in_flight{0};
std::atomic<bool> g_releasing{false};

}  // namespace

int CallsInFlight() noexcept { return g_in_flight.load(); }

InFlight::InFlight() {
  // Dekker-style handshake with ReleaseAllLive (sequentially consistent).
  ++g_in_flight;
  if (g_releasing.load()) {
    --g_in_flight;
    throw StaleHostError(
        "the imsrg runtime is being finalized; the call was not started");
  }
}

InFlight::~InFlight() { --g_in_flight; }

void BusyFlag::Enter() {
  std::lock_guard lock(mutex_);
  const auto me = std::this_thread::get_id();
  if (depth_ > 0 && owner_ != me)
    throw imsrg::RuntimeError("object in use by another thread");
  owner_ = me;
  ++depth_;
}

void BusyFlag::Leave() noexcept {
  std::lock_guard lock(mutex_);
  if (--depth_ == 0) owner_ = std::thread::id();
}

void ThrowIfSignalPending() {
  if (PyErr_CheckSignals() != 0) throw nb::python_error();
}

PyModelSpace::~PyModelSpace() {
  const bool shared = ms && ms.use_count() > 1;
  ms.reset();
  if (shared) PurgeWorkspaceCache();
}

void RequireSameModelSpace(const op::Operator& a, const op::Operator& b) {
  if (!op::SameModelSpace(a, b))
    throw ModelSpaceError("the operands live on different model spaces");
}

void RequireFile(const std::filesystem::path& path) {
  std::error_code ec;
  if (!std::filesystem::exists(path, ec))
    throw MissingFileError("file not found: '" + path.string() + "'");
}

int ReleaseAllLive() noexcept {
  auto& r = GetRegistry();
  std::lock_guard lock(r.mutex);
  g_releasing.store(true);
  struct Reset {
    ~Reset() { g_releasing.store(false); }
  } reset;
  if (g_in_flight.load() > 0) return -1;
  int blocked = 0;
  for (const Releasable* o : r.live)
    if (!o->released && !o->CanRelease()) ++blocked;
  if (blocked > 0) return blocked;
  for (Releasable* o : r.live) {
    if (o->released) continue;
    o->Release();
    o->released = true;
  }
  return 0;
}

void SetResidencyPolicy(ResidencyPolicy p) {
  PolicyStorage().store(static_cast<int>(p));
}

ResidencyPolicy GetResidencyPolicy() {
  return static_cast<ResidencyPolicy>(PolicyStorage().load());
}

bool ResidencyEnforced() {
  switch (GetResidencyPolicy()) {
    case ResidencyPolicy::kStrict:
      return true;
    case ResidencyPolicy::kImplicit:
      return false;
    case ResidencyPolicy::kAuto:
      break;
  }
  return !exec::DefaultMemoryIsHostAccessible();
}

const PyOperator& Alive(const PyOperator& o) {
  if (o.released)
    throw StaleHostError(
        "the device data of this operator were released by "
        "Runtime.finalize(force=True); the operator cannot be used any more");
  return o;
}

const op::Operator& Input(const PyOperator& o) {
  Alive(o);
  if (o.residency == Residency::kHostCurrent) {
    // Python hands out non-const objects; the push changes where the data
    // live, not their value (design/18 section 2.2).
    auto& m = const_cast<PyOperator&>(o);
    m.SyncToDevice();
    // A live writable view keeps the host side authoritative: its later
    // edits must be pushed before the next call as well.
    if (m.writable_views->load() == 0) m.residency = Residency::kBoth;
  }
  return o;
}

void PrepareHostAccess(PyOperator& o, bool writable) {
  Alive(o);
  if (o.residency == Residency::kDeviceCurrent) {
    if (ResidencyEnforced())
      throw StaleHostError(
          "the host mirror of this operator is stale (its data were last "
          "written on the device): call op.sync_to_host() first");
    o.SyncToHost();
    o.residency = Residency::kBoth;
  }
  if (writable) o.residency = Residency::kHostCurrent;
}

namespace {

/// Payload of a view guard capsule.
struct ViewGuard {
  nb::object owner;
  std::shared_ptr<std::atomic<int>> count;
  std::shared_ptr<std::atomic<int>> writable;
};

}  // namespace

nb::object MakeViewGuard(nb::handle self, bool writable) {
  auto& o = nb::cast<PyOperator&>(self);
  auto* g = new ViewGuard{nb::borrow(self), o.views,
                          writable ? o.writable_views : nullptr};
  ++*g->count;
  if (g->writable) ++*g->writable;
  return nb::capsule(g, [](void* p) noexcept {
    auto* guard = static_cast<ViewGuard*>(p);
    --*guard->count;
    if (guard->writable) --*guard->writable;
    delete guard;
  });
}

void MarkModifiedInPlace(PyOperator& o) {
  if (o.writable_views->load() > 0) {
    o.SyncToHost();
    o.residency = Residency::kHostCurrent;
  } else {
    o.residency = Residency::kDeviceCurrent;
  }
}

}  // namespace imsrg::py

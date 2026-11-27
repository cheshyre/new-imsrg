// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/termination.h"

#include <csignal>

namespace imsrg::driver {
namespace {

// The only state a signal handler may share with the program (design/00 D75):
// written by the handler, read between flow steps. sig_atomic_t is trivially
// destructible and async-signal-safe.
volatile std::sig_atomic_t termination_requested = 0;

void OnTerminationSignal(int /*signum*/) { termination_requested = 1; }

}  // namespace

void InstallTerminationHandler() { std::signal(SIGTERM, OnTerminationSignal); }

void RestoreDefaultTerminationHandler() { std::signal(SIGTERM, SIG_DFL); }

bool TerminationRequested() noexcept { return termination_requested != 0; }

void RequestTermination() noexcept { termination_requested = 1; }

void ClearTerminationRequest() noexcept { termination_requested = 0; }

}  // namespace imsrg::driver

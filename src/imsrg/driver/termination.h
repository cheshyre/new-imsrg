// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_TERMINATION_H_
#define IMSRG_DRIVER_TERMINATION_H_
/// @file
/// Termination requests (design/00 D75): Slurm sends SIGTERM some minutes
/// before the end of a job (`--signal=B:TERM@300`). The handler only sets a
/// flag; `driver::Run` checks it between Magnus steps (combined over all MPI
/// ranks), writes a checkpoint and throws `InterruptedError`, which the
/// `imsrg` executable maps to exit status `kInterruptedExitStatus`.
///
/// The flag is the one piece of mutable process-wide state of the driver: a
/// signal handler can communicate with the program in no other way.

namespace imsrg::driver {

/// Exit status of the `imsrg` executable after an `InterruptedError`
/// (EX_TEMPFAIL of sysexits.h: the run can be continued with
/// `checkpoint.restart = true`).
inline constexpr int kInterruptedExitStatus = 75;

/// Installs the SIGTERM handler (it sets the termination flag and does
/// nothing else). Called by the executable's `main`, never by library
/// initialization. Idempotent.
void InstallTerminationHandler();

/// Restores the default SIGTERM disposition (tests).
void RestoreDefaultTerminationHandler();

/// True once a termination was requested (SIGTERM received, or
/// `RequestTermination`) and not cleared. Async-signal-safe, lock-free.
bool TerminationRequested() noexcept;

/// Sets the termination flag as the handler does (tests, embedding code).
void RequestTermination() noexcept;

/// Clears the termination flag (tests, or before a new run in the same
/// process).
void ClearTerminationRequest() noexcept;

}  // namespace imsrg::driver

#endif  // IMSRG_DRIVER_TERMINATION_H_

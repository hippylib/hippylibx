# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

from mpi4py import MPI

def master_print(comm: MPI.Comm, *args, **kwargs) -> None:
    if comm.rank == 0:
        print(*args, **kwargs)

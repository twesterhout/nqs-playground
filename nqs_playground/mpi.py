#!/usr/bin/env python3

# Copyright Tom Westerhout (c) 2019
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
from mpi4py import MPI


def parallel_for(func, begin, end, make_recv_buffer, dtype, comm=MPI.COMM_WORLD, root=0):
    """
    A parallel version of ``[func(_, i) for i in range(begin, end)]``.

    :param func:
        A callable that receives two ``int``s and returns a buffer (i.e. a type
        supporting Python's buffer interface). ``func`` will be called with the
        process' rank and index (from ``[begin, end)``).
    :param begin:
        Starting index.
    :param end:
        Stop index.
    :param make_recv_buffer:
        Results produced by ``func`` are communicated using MPI. To use fast
        versions of the routines, we need to pre-allocate the "receiving
        buffer". ``make_recv_buffer`` should do exactly that.
    :param dtype:
        MPI datatype of buffers returned by ``func``.
    :param comm:
        MPI communicator to use.
    :param root:
        Rank of the process to which to send all the results.
    """
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    def get_local_chunk():
        """
        Evenly divides the work between MPI processes. Processes are
        assigned contiguous ranges of indices. If the work can't be
        distributed evenly, we make sure that the root process has slighty
        less on his plate.
        """
        size = end - begin
        rest = size % comm_size
        chunk_size = size // comm_size
        # NOTE: The rest is magic :)
        flag = rank < rest + (root < rest)
        local_begin = (
            begin
            + rank * chunk_size
            + flag * (rank - (root < rank))
            + (not flag) * rest
        )
        local_end = local_begin + chunk_size + (flag and rank != root)
        return local_begin, local_end

    local_begin, local_end = get_local_chunk()
    # print("{}: {}".format(rank, list(range(local_begin, local_end))))
    # Share the chunk sizes. We interleave this communication with computation
    counts = numpy.empty(comm_size, dtype=numpy.int32) if rank == root else None
    request = comm.Igather(
        numpy.array([local_end - local_begin], dtype=numpy.int32), counts, root=root
    )

    if rank != root:
        if local_end > local_begin:
            # Do some computation first giving MPI time to complete the Igather
            # request
            r = func(rank, local_begin)
            # Calls to func will be quite expensive in practive so waiting for
            # one call should given MPI enough time to send a few integers.
            request.Wait()
            # We keep an array of results to make sure that buffers aren't
            # garbage collected before Isend finishes.
            results = [r]
            request = [comm.Isend([r, dtype], dest=root)]
            for i in range(local_begin + 1, local_end):
                r = func(rank, i)
                request.append(comm.Isend([r, dtype], dest=root))
                results.append(r)
            for req in request:
                req.Wait()
        return None
    else:
        # If possible, do some work first giving MPI time to complete the
        # Igather request
        if local_end > local_begin:
            r = func(rank, local_begin)
        request.Wait()
        request = []
        results = []
        # We allocate all the buffers and issue Irecv operations so that
        # other processes can start sending us results
        for i in range(comm_size):
            if i == root:
                results += [None for _ in range(counts[i])]
            elif counts[i] > 0:
                buffers = [make_recv_buffer() for _ in range(counts[i])]
                request += [
                    comm.Irecv([buffers[k], dtype], source=i) for k in range(counts[i])
                ]
                results += buffers
        if local_end > local_begin:
            results[local_begin - begin] = r
            for i in range(local_begin + 1, local_end):
                results[i - begin] = func(rank, i)
        for req in request:
            req.Wait()
        return results

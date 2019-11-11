#include "pool.hpp"

TCM_NAMESPACE_BEGIN

Pool::Pool(size_t const chunk_size, size_t const number_chunks)
    : _pool{chunk_size, number_chunks}, _lock{}
{}

auto Pool::malloc() -> void*
{
    GuardT guard{_lock};
    auto*  p = _pool.malloc();
    if (TCM_UNLIKELY(p == nullptr)) { throw std::bad_alloc{}; }
    return p;
}

auto Pool::free(void* p) noexcept -> void
{
    if (TCM_UNLIKELY(p == nullptr)) { return; }
    GuardT guard{_lock};
    _pool.free(p);
}

TCM_NAMESPACE_END

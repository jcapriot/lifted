#ifndef LIFTED_COMMON_H_
#define LIFTED_COMMON_H_

#include <cstddef>
#include <vector>
#include <concepts>
#include <array>
#include <algorithm>
#include <functional>
#include <type_traits>

#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>
#include <functional>
#include <new>

// Reserved:
//    0 : Lazy Wavelet (Just a deinterleave operation)
//    1-99 for Daubechies
//    100-199 for Symlets
//    200-299 for Coiflets
//    10000-19999 for Biorthogonal
//    20000-29999 for Reverse Biorthogonal
//    100000+ for Others (CDF5_3, CDF9_7, etc...)
// Note: Do NOT change these numbers:
#ifndef LIFTED_WAVELETS
#define LIFTED_WAVELETS \
    X(Lazy, 0) \
    X(Daubechies1, 1) \
    X(Daubechies2, 2) \
    X(Daubechies3, 3) \
    X(Daubechies4, 4) \
    X(Daubechies5, 5) \
    X(Daubechies6, 6) \
    X(Daubechies7, 7) \
    X(Daubechies8, 8) \
    X(Daubechies9, 9) \
    X(Daubechies10, 10) \
    X(Symlet4, 104) \
    X(Symlet5, 105) \
    X(Symlet6, 106) \
    X(Coiflet2, 202) \
    X(Coiflet3, 203) \
    X(Bior1_3, 10103) \
    X(Bior1_5, 10105) \
    X(Bior2_2, 10202) \
    X(Bior2_4, 10204) \
    X(Bior2_6, 10206) \
    X(Bior2_8, 10208) \
    X(Bior3_1, 10301) \
    X(Bior3_3, 10303) \
    X(Bior3_5, 10305) \
    X(Bior3_7, 10307) \
    X(Bior3_9, 10309) \
    X(Bior4_2, 10402) \
    X(Bior4_4, 10404) \
    X(Bior4_6, 10406) \
    X(Bior5_5, 10505) \
    X(Bior6_8, 10608) \
	X(CDF5_3, 100503) \
	X(CDF9_7, 100907)
#endif


// Note: Do NOT change the assiged numbers, and increase them
// sequentially for jump table indexing
#ifndef LIFTED_BOUNDARY_CONDITIONS
#define LIFTED_BOUNDARY_CONDITIONS \
    X(Zero, 0) \
    X(Constant, 1) \
    X(Periodic, 2) \
	X(Symmetric, 3) \
	X(Reflect, 4)
#endif

#ifndef LIFTED_N_BC_TYPES
#define LIFTED_N_BC_TYPES 5
#endif

// Note: Do NOT change the assiged numbers, and increase them
// sequentially for jump table indexing
#ifndef LIFTED_TRANSFORM_TYPES
#define LIFTED_TRANSFORM_TYPES \
    X(Forward, 0) \
    X(Inverse, 1) \
	X(ForwardAdjoint, 2) \
	X(InverseAdjoint, 3)
#endif
#ifndef LIFTED_N_TRANSFORM_TYPES
#define LIFTED_N_TRANSFORM_TYPES 4
#endif

namespace lifted {
    
using std::size_t;
using std::ptrdiff_t;

using size_v = std::vector<size_t>;
using stride_v = std::vector<ptrdiff_t>;
using stride_t = stride_v::value_type;

enum class Wavelet : std::uint32_t {
	#define X(name, val) name = val,
	LIFTED_WAVELETS
	#undef X
};

enum class BoundaryCondition : std::uint32_t {
	#define X(name, val) name = val,
	LIFTED_BOUNDARY_CONDITIONS
	#undef X
};

enum class Transform : std::uint32_t {
    #define X(name, val) name = val,
	LIFTED_TRANSFORM_TYPES
	#undef X
};

namespace detail {
    using std::array;
    using std::vector;

    template<typename To, typename... From>
    concept AllConvertible = (std::convertible_to<From, To> && ...);

    template<typename V>
    static inline size_t prod(const V& shape)
    {
        size_t res = 1;
        for (auto sz : shape)
            res *= sz;
        return res;
    }

    namespace threading {

#ifdef _MSC_VER

inline void* aligned_alloc(size_t align, size_t size) {
    void* ptr = _aligned_malloc(size, align);
	if (!ptr) throw std::bad_alloc();
	return ptr;
}

inline void aligned_dealloc(void* ptr) {
	if (ptr) _aligned_free(ptr);
}
#else
inline void* aligned_alloc(size_t align, size_t size)
{
	align = std::max(align, alignof(max_align_t));
	void* ptr = malloc(size + align);
	if (!ptr) throw std::bad_alloc();
	void* res = reinterpret_cast<void*>
		((reinterpret_cast<uintptr_t>(ptr) & ~(uintptr_t(align - 1))) + uintptr_t(align));
	(reinterpret_cast<void**>(res))[-1] = ptr;
	return res;
	}

inline void aligned_dealloc(void* ptr)
{
	if (ptr) free((reinterpret_cast<void**>(ptr))[-1]);
}
#endif

    static inline size_t thread_count(size_t nthreads, const size_v& shape,
        size_t axis, size_t vlen)
    {
        if (nthreads == 1) return 1;
        size_t size = prod(shape);
        size_t parallel = size / (shape[axis] * vlen);
        if (shape[axis] < 1000)
            parallel /= 4;
        size_t max_threads = nthreads == 0 ?
            std::thread::hardware_concurrency() : nthreads;
        return std::max(size_t(1), std::min(parallel, max_threads));
}

    inline size_t& thread_id()
    {
        static thread_local size_t thread_id_ = 0;
        return thread_id_;
    }
    inline size_t& num_threads()
    {
        static thread_local size_t num_threads_ = 1;
        return num_threads_;
    }
    static const size_t max_threads = std::max(1u, std::thread::hardware_concurrency());

    class latch
    {
        std::atomic<size_t> num_left_;
        std::mutex mut_;
        std::condition_variable completed_;
        using lock_t = std::unique_lock<std::mutex>;

    public:
        latch(size_t n) : num_left_(n) {}

        void count_down()
        {
            lock_t lock(mut_);
            if (--num_left_)
                return;
            completed_.notify_all();
        }

        void wait()
        {
            lock_t lock(mut_);
            completed_.wait(lock, [this] { return is_ready(); });
        }
        bool is_ready() { return num_left_ == 0; }
    };

    template <typename T> class concurrent_queue
    {
        std::queue<T> q_;
        std::mutex mut_;
        std::atomic<size_t> size_;
        using lock_t = std::lock_guard<std::mutex>;

    public:

        void push(T val)
        {
            lock_t lock(mut_);
            ++size_;
            q_.push(std::move(val));
        }

        bool try_pop(T& val)
        {
            if (size_ == 0) return false;
            lock_t lock(mut_);
            // Queue might have been emptied while we acquired the lock
            if (q_.empty()) return false;

            val = std::move(q_.front());
            --size_;
            q_.pop();
            return true;
        }

        bool empty() const { return size_ == 0; }
    };

    // C++ allocator with support for over-aligned types
    template <typename T> struct aligned_allocator
    {
        using value_type = T;
        template <class U>
        aligned_allocator(const aligned_allocator<U>&) {}
        aligned_allocator() = default;

        T* allocate(size_t n)
        {
            void* mem = aligned_alloc(alignof(T), n * sizeof(T));
            return static_cast<T*>(mem);
        }

        void deallocate(T* p, size_t /*n*/)
        {
            aligned_dealloc(p);
        }
    };

    class thread_pool
    {
        // A reasonable guess, probably close enough for most hardware
        static constexpr size_t cache_line_size = 64;
        struct alignas(cache_line_size) worker
        {
            std::thread thread;
            std::condition_variable work_ready;
            std::mutex mut;
            std::atomic_flag busy_flag = ATOMIC_FLAG_INIT;
            std::function<void()> work;

            void worker_main(
                std::atomic<bool>& shutdown_flag,
                std::atomic<size_t>& unscheduled_tasks,
                concurrent_queue<std::function<void()>>& overflow_work)
            {
                using lock_t = std::unique_lock<std::mutex>;
                bool expect_work = true;
                while (!shutdown_flag || expect_work)
                {
                    std::function<void()> local_work;
                    if (expect_work || unscheduled_tasks == 0)
                    {
                        lock_t lock(mut);
                        // Wait until there is work to be executed
                        work_ready.wait(lock, [&] { return (work || shutdown_flag); });
                        local_work.swap(work);
                        expect_work = false;
                    }

                    bool marked_busy = false;
                    if (local_work)
                    {
                        marked_busy = true;
                        local_work();
                    }

                    if (!overflow_work.empty())
                    {
                        if (!marked_busy && busy_flag.test_and_set())
                        {
                            expect_work = true;
                            continue;
                        }
                        marked_busy = true;

                        while (overflow_work.try_pop(local_work))
                        {
                            --unscheduled_tasks;
                            local_work();
                        }
                    }

                    if (marked_busy) busy_flag.clear();
                }
            }
        };

        concurrent_queue<std::function<void()>> overflow_work_;
        std::mutex mut_;
        std::vector<worker, aligned_allocator<worker>> workers_;
        std::atomic<bool> shutdown_;
        std::atomic<size_t> unscheduled_tasks_;
        using lock_t = std::lock_guard<std::mutex>;

        void create_threads()
        {
            lock_t lock(mut_);
            size_t nthreads = workers_.size();
            for (size_t i = 0; i < nthreads; ++i)
            {
                try
                {
                    auto* worker = &workers_[i];
                    worker->busy_flag.clear();
                    worker->work = nullptr;
                    worker->thread = std::thread([worker, this]
                        {
                            worker->worker_main(shutdown_, unscheduled_tasks_, overflow_work_);
                        });
                }
                catch (...)
                {
                    shutdown_locked();
                    throw;
                }
            }
        }

        void shutdown_locked()
        {
            shutdown_ = true;
            for (auto& worker : workers_)
                worker.work_ready.notify_all();

            for (auto& worker : workers_)
                if (worker.thread.joinable())
                    worker.thread.join();
        }

    public:
        explicit thread_pool(size_t nthreads) :
            workers_(nthreads)
        {
            create_threads();
        }

        thread_pool() : thread_pool(max_threads) {}

        ~thread_pool() { shutdown(); }

        void submit(std::function<void()> work)
        {
            lock_t lock(mut_);
            if (shutdown_)
                throw std::runtime_error("Work item submitted after shutdown");

            ++unscheduled_tasks_;

            // First check for any idle workers and wake those
            for (auto& worker : workers_)
                if (!worker.busy_flag.test_and_set())
                {
                    --unscheduled_tasks_;
                    {
                        lock_t lock(worker.mut);
                        worker.work = std::move(work);
                    }
                    worker.work_ready.notify_one();
                    return;
                }

            // If no workers were idle, push onto the overflow queue for later
            overflow_work_.push(std::move(work));
        }

        void shutdown()
        {
            lock_t lock(mut_);
            shutdown_locked();
        }

        void restart()
        {
            shutdown_ = false;
            create_threads();
        }
    };

    inline thread_pool& get_pool()
    {
        static thread_pool pool;
#ifdef LIFTED_PTHREADS
        static std::once_flag f;
        std::call_once(f,
            [] {
                pthread_atfork(
                    +[] { get_pool().shutdown(); },  // prepare
                    +[] { get_pool().restart(); },   // parent
                    +[] { get_pool().restart(); }    // child
                );
            });
#endif

        return pool;
    }

    /** Map a function f over nthreads */
    template <typename Func>
    void thread_map(size_t nthreads, Func f)
    {
        if (nthreads == 0)
            nthreads = max_threads;

        if (nthreads == 1)
        {
            f(); return;
        }

        auto& pool = get_pool();
        latch counter(nthreads);
        std::exception_ptr ex;
        std::mutex ex_mut;
        for (size_t i = 0; i < nthreads; ++i)
        {
            pool.submit(
                [&f, &counter, &ex, &ex_mut, i, nthreads] {
                    thread_id() = i;
                    num_threads() = nthreads;
                    try { f(); }
                    catch (...)
                    {
                        std::lock_guard<std::mutex> lock(ex_mut);
                        ex = std::current_exception();
                    }
                    counter.count_down();
                });
        }
        counter.wait();
        if (ex)
            std::rethrow_exception(ex);
    }
}

    
    class arr_info
    {
    protected:
        size_v shp;
        stride_v str;

    public:
        arr_info(const size_v& shape_, const stride_v& stride_)
            : shp(shape_), str(stride_) {}
        size_t ndim() const { return shp.size(); }
        size_t size() const { return prod(shp); }
        const size_v& shape() const { return shp; }
        size_t shape(size_t i) const { return shp[i]; }
        const stride_v& stride() const { return str; }
        const ptrdiff_t& stride(size_t i) const { return str[i]; }
    };

    template<typename T> class cndarr : public arr_info
    {
    protected:
        const T* d;

    public:
        cndarr(const T* data_, const size_v& shape_, const stride_v& stride_)
            : arr_info(shape_, stride_),
            d(data_) {}
        const T& operator[](ptrdiff_t ofs) const
        {
            return d[ofs];
        }

        cndarr get_subset(const size_v& start, const size_v& end) {
            size_v new_shape = end;
            const T* new_d = d;
            for (size_t i = 0; i < start.size(); ++i)
            {
                new_d += start[i] * str[i];
                new_shape[i] -= start[i];
            }
            return cndarr(new_d, new_shape, str);
        }
    };

    template<typename T> class ndarr : public cndarr<T>{
    public:
        ndarr(T* data_, const size_v& shape_, const stride_v& stride_)
            : cndarr<T>::cndarr(const_cast<const T*>(data_), shape_, stride_)
        {}
        T& operator[](ptrdiff_t ofs)
        {
            return *const_cast<T*>(cndarr<T>::d + ofs);
        }
        ndarr get_subset(const size_v& start, const size_v& end) {
            size_v new_shape = end;
            T* new_d = cndarr<T>::d;
            for (size_t i = 0; i < start.size(); ++i)
            {
                new_d += start[i] * cndarr<T>::str[i];
                new_shape[i] -= start[i];
            }
            return ndarr(new_d, new_shape, cndarr<T>::str);
        }
    };

    constexpr size_t dynamic_size = static_cast<size_t>(-1);
    
    template<size_t N=dynamic_size> class multi_iter
    {
    private:
        size_v pos;
        const arr_info& iarr, & oarr;
        ptrdiff_t p_ii, p_i[N], str_i, p_oi, p_o[N], str_o;
        size_t idim, rem;

        void advance_i()
        {
            for (size_t i = 0; i < pos.size(); ++i)
            {
                if (i == idim) continue;
                p_ii += iarr.stride(i);
                p_oi += oarr.stride(i);
                if (++pos[i] < iarr.shape(i))
                    return;
                pos[i] = 0;
                p_ii -= ptrdiff_t(iarr.shape(i)) * iarr.stride(i);
                p_oi -= ptrdiff_t(oarr.shape(i)) * oarr.stride(i);
            }
        }

    public:
        multi_iter(const arr_info& iarr_, const arr_info& oarr_, size_t idim_)
            : pos(iarr_.ndim(), 0), iarr(iarr_), oarr(oarr_), p_ii(0),
            str_i(iarr.stride(idim_)), p_oi(0), str_o(oarr.stride(idim_)),
            idim(idim_), rem(iarr.size() / iarr.shape(idim))
        {
            auto nshares = threading::num_threads();
            if (nshares == 1) return;
            if (nshares == 0) throw std::runtime_error("can't run with zero threads");
            auto myshare = threading::thread_id();
            if (myshare >= nshares) throw std::runtime_error("impossible share requested");
            size_t nbase = rem / nshares;
            size_t additional = rem % nshares;
            size_t lo = myshare * nbase + ((myshare < additional) ? myshare : additional);
            size_t hi = lo + nbase + (myshare < additional);
            size_t todo = hi - lo;

            size_t chunk = rem;
            for (size_t i = 0; i < pos.size(); ++i)
            {
                if (i == idim) continue;
                chunk /= iarr.shape(i);
                size_t n_advance = lo / chunk;
                pos[i] += n_advance;
                p_ii += ptrdiff_t(n_advance) * iarr.stride(i);
                p_oi += ptrdiff_t(n_advance) * oarr.stride(i);
                lo -= n_advance * chunk;
            }
            rem = todo;
        }
        void advance(size_t n)
        {
            if (rem < n) throw std::runtime_error("underrun");
            for (size_t i = 0; i < n; ++i)
            {
                p_i[i] = p_ii;
                p_o[i] = p_oi;
                advance_i();
            }
            rem -= n;
        }
        ptrdiff_t iofs(size_t i) const { return p_i[0] + ptrdiff_t(i) * str_i; }
        ptrdiff_t iofs(size_t j, size_t i) const { return p_i[j] + ptrdiff_t(i) * str_i; }
        ptrdiff_t oofs(size_t i) const { return p_o[0] + ptrdiff_t(i) * str_o; }
        ptrdiff_t oofs(size_t j, size_t i) const { return p_o[j] + ptrdiff_t(i) * str_o; }
        size_t length_in() const { return iarr.shape(idim); }
        size_t length_out() const { return oarr.shape(idim); }
        ptrdiff_t stride_in() const { return str_i; }
        ptrdiff_t stride_out() const { return str_o; }
        size_t remaining() const { return rem; }
        bool i_across_contiguous() const { return p_i[0] + ptrdiff_t(N) - 1 == p_i[N-1]; }
        bool o_across_contiguous() const { return p_o[0] + ptrdiff_t(N) - 1 == p_o[N-1]; }
        bool i_along_contiguous() const { return str_i == 1; }
        bool o_along_contiguous() const { return str_o == 1; }
    };

    template<>
    class multi_iter<dynamic_size>
    {
    private:
        size_v pos;
        const arr_info& iarr, & oarr;
        ptrdiff_t p_ii, str_i, p_oi, str_o;
        stride_v p_i, p_o;
        size_t idim, rem;
        const size_t N;

        void advance_i()
        {
            for (size_t i = 0; i < pos.size(); ++i)
            {
                if (i == idim) continue;
                p_ii += iarr.stride(i);
                p_oi += oarr.stride(i);
                if (++pos[i] < iarr.shape(i))
                    return;
                pos[i] = 0;
                p_ii -= ptrdiff_t(iarr.shape(i)) * iarr.stride(i);
                p_oi -= ptrdiff_t(oarr.shape(i)) * oarr.stride(i);
            }
        }

    public:
        multi_iter(const arr_info& iarr_, const arr_info& oarr_, size_t idim_, const size_t n)
            : pos(iarr_.ndim(), 0), iarr(iarr_), oarr(oarr_), p_ii(0),
            str_i(iarr.stride(idim_)), p_oi(0), str_o(oarr.stride(idim_)),
            p_i(n, 0), p_o(n, 0),
            idim(idim_), rem(iarr.size() / iarr.shape(idim)),
            N(n)
        {
            auto nshares = threading::num_threads();
            if (nshares == 1) return;
            if (nshares == 0) throw std::runtime_error("can't run with zero threads");
            auto myshare = threading::thread_id();
            if (myshare >= nshares) throw std::runtime_error("impossible share requested");
            size_t nbase = rem / nshares;
            size_t additional = rem % nshares;
            size_t lo = myshare * nbase + ((myshare < additional) ? myshare : additional);
            size_t hi = lo + nbase + (myshare < additional);
            size_t todo = hi - lo;

            size_t chunk = rem;
            for (size_t i = 0; i < pos.size(); ++i)
            {
                if (i == idim) continue;
                chunk /= iarr.shape(i);
                size_t n_advance = lo / chunk;
                pos[i] += n_advance;
                p_ii += ptrdiff_t(n_advance) * iarr.stride(i);
                p_oi += ptrdiff_t(n_advance) * oarr.stride(i);
                lo -= n_advance * chunk;
            }
            rem = todo;
        }
        void advance(size_t n)
        {
            if (rem < n) throw std::runtime_error("underrun");
            for (size_t i = 0; i < n; ++i)
            {
                p_i[i] = p_ii;
                p_o[i] = p_oi;
                advance_i();
            }
            rem -= n;
        }
        ptrdiff_t iofs(size_t i) const { return p_i[0] + ptrdiff_t(i) * str_i; }
        ptrdiff_t iofs(size_t j, size_t i) const { return p_i[j] + ptrdiff_t(i) * str_i; }
        ptrdiff_t oofs(size_t i) const { return p_o[0] + ptrdiff_t(i) * str_o; }
        ptrdiff_t oofs(size_t j, size_t i) const { return p_o[j] + ptrdiff_t(i) * str_o; }
        size_t length_in() const { return iarr.shape(idim); }
        size_t length_out() const { return oarr.shape(idim); }
        ptrdiff_t stride_in() const { return str_i; }
        ptrdiff_t stride_out() const { return str_o; }
        size_t remaining() const { return rem; }
        bool i_across_contiguous() const { return p_i[0] + ptrdiff_t(N) - 1 == p_i[N-1]; }
        bool o_across_contiguous() const { return p_o[0] + ptrdiff_t(N) - 1 == p_o[N-1]; }
        bool i_along_contiguous() const { return str_i == 1; }
        bool o_along_contiguous() const { return str_o == 1; }
    };

    template <size_t N, typename Func>
    constexpr void static_for(Func&& f) {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (f.template operator()<Is>(), ...);
        }(std::make_index_sequence<N>{});
    }

    // Boundary conditions
    struct ZeroBoundary {
        static inline size_t index_of(const ptrdiff_t i, const size_t n) {
            if ((i < 0) || (i >= ptrdiff_t(n))) {
                return n;
            }
            else {
                return i;
            }
        }
    };

    struct ConstantBoundary {
        static inline size_t index_of(const ptrdiff_t i, const size_t n) {
            if (i < 0) {
                return 0;
            }
            else if (i >= ptrdiff_t(n)) {
                return n - 1;
            }
            else {
                return i;
            }
        }
    };

    struct PeriodicBoundary {
        static inline size_t index_of(const ptrdiff_t i, const size_t n) {
            return i % n;
        }
    };

    struct SymmetricBoundary {
        static inline size_t index_of(const ptrdiff_t i, const size_t n) {
            ptrdiff_t io = i;
            while ((io >= ptrdiff_t(n)) || (io < 0)) {
                if (io < 0) {
                    io = -(io + 1);
                }
                else {
                    io = 2 * (n - 1) - (io - 1);
                }
            }
            return io;
        }
    };

    struct ReflectBoundary {
        static inline size_t index_of(const ptrdiff_t i, const size_t n) {
            ptrdiff_t io = i;
            while ((io >= ptrdiff_t(n)) || (io < 0)) {
                if (io < 0) {
                    io = -io;
                }
                else {
                    io = 2 * (n - 1) - io;
                }
            }
            return io;
        }
    };

    // UpdateType enum to specify whether we are updating even or odd indices
    enum class UpdateType {
        UpdateEvens,
        UpdateOdds,
    };

    struct Along{};
    struct Across{};

    template<typename T>
    concept VecDir = std::same_as<T, Along> || std::same_as<T, Across>;

    enum class UpdateOperation{
        add,
        sub
    };

    template<size_t N>
    class OffsetData {
    public:
    
        const ptrdiff_t offset;
        const size_t n_front;
        const size_t n_back;
        
        const ptrdiff_t offset_r;
        const size_t n_front_r;
        const size_t n_back_r;

        constexpr OffsetData(const ptrdiff_t offset_) : 
            offset(offset_),
            n_front((offset < 0) ? -offset : 0),
            n_back((max_offset() < 0) ? 0 : max_offset()),
            offset_r(-max_offset()),
            n_front_r((offset_r < 0) ? -offset_r : 0),
            n_back_r((max_offset_r() < 0) ? 0 : max_offset_r())
        {}

    private:
        constexpr ptrdiff_t max_offset() const {return ptrdiff_t(N) - 1 + offset;}
        constexpr ptrdiff_t max_offset_r() const {return ptrdiff_t(N) - 1 + offset_r;}
    };

    template<typename T, size_t N>
    class GenericUpdateStep : protected OffsetData<N> {
		static_assert(N != 0, "Size must be greater than 0");
    private:
        using Base = OffsetData<N>;
    public:
        using Base::offset;
        using Base::n_front;
        using Base::n_back;
    
        using Base::offset_r;
        using Base::n_front_r;
        using Base::n_back_r;

        using type = T;
		const static size_t n_vals = N;
		const UpdateType update_type;
		const array<T, N> vals;
		const array<T, N> vals_r;

    public:
		constexpr GenericUpdateStep() : Base(0), update_type(UpdateType::UpdateEvens), vals{}, vals_r{} {}

        template<typename U>
        requires AllConvertible<T, U>
		constexpr GenericUpdateStep(const UpdateType sd, const ptrdiff_t offset_, const array<U, N>& arr) : 
            Base(offset_),
			update_type(sd),
            vals(to_std_array(arr)),
            vals_r(to_rev_array(vals))
		{}

        template<typename U>
        requires AllConvertible<T, U>
        GenericUpdateStep(const UpdateType sd, const ptrdiff_t offset_, const vector<U>& vec) : 
            Base(offset_), 
            update_type(sd),
            vals(to_std_array(vec)),
            vals_r(to_rev_array(vals))
        {}

    private:

        template <typename Container>
        constexpr static array<T, N> to_std_array(const Container& container) {
            if (container.size() != N) {
                throw std::invalid_argument("Container has wrong size for array");
            }
            array<T, N> arr;
            std::copy(container.begin(), container.end(), arr.begin());
            return arr;
        }

        template <typename Container>
        constexpr static array<T, N> to_rev_array(const Container& container) {
            if (container.size() != N) {
                throw std::invalid_argument("Container has wrong size for array");
            }
            array<T, N> arr;
            std::reverse_copy(container.begin(), container.end(), arr.begin());
            return arr;
        }
    };

    template<typename T, UpdateOperation PM=UpdateOperation::add>
    class UnitUpdateStep : protected OffsetData<1>{
    private:
        using Base = OffsetData<1>;
    public:
        using Base::offset;
        using Base::n_front;
        using Base::n_back;
    
        using Base::offset_r;
        using Base::n_front_r;
        using Base::n_back_r;

        using type = T;
		const static size_t n_vals = 1;
        const static UpdateOperation pm = PM;
		const UpdateType update_type;

        constexpr UnitUpdateStep(const UpdateType sd, const ptrdiff_t offset_) :
            Base(offset_),
            update_type(sd) {}
    };

    template<typename T, size_t N>
    class RepeatUpdateStep : protected OffsetData<N>{
		static_assert(N != 0, "Size must be greater than 0");
        using Base = OffsetData<N>;

    public:
        using Base::offset;
        using Base::n_front;
        using Base::n_back;
        using Base::offset_r;
        using Base::n_front_r;
        using Base::n_back_r;

        const static size_t n_vals = N;
        const UpdateType update_type;
        const T val;

        template<typename U>
        requires AllConvertible<T, U>
        constexpr RepeatUpdateStep(const UpdateType sd, const ptrdiff_t offset_, const U val_) :
            Base(offset_),
            update_type(sd),
            val(val_){}
    };

    template<typename T>
    class ScaleStep {
    public:
        const T s_mul;
        const T s_div;
    
        const T d_div;
        const T d_mul;

        using type = T;

		constexpr ScaleStep() : s_mul(1), s_div(1), d_div(1), d_mul(1) {};
		
		template<typename U>
        requires AllConvertible<T, U>
		constexpr ScaleStep(const U s_mul_d_div = 1) :
            s_mul(static_cast<T>(s_mul_d_div)),
            s_div(static_cast<T>(1.0L/s_mul_d_div)),
            d_div(static_cast<T>(s_mul_d_div)),
            d_mul(static_cast<T>(1.0L/s_mul_d_div)){}

		template<typename U1, typename U2>
        requires AllConvertible<T, U1, U2>
		constexpr ScaleStep(const U1 s_mul_, const U2 d_div_) :
            s_mul(static_cast<T>(s_mul_)),
            s_div(static_cast<T>(1.0L/s_mul_)),
            d_div(static_cast<T>(d_div_)),
            d_mul(static_cast<T>(1.0L/d_div_)){};
    };

    // Structs defining the transform type
    struct Forward{
        constexpr static bool forward_step() {return true;}
    };
    struct Inverse{
        constexpr static bool forward_step() {return false;}
    };
    struct ForwardAdjoint{
        constexpr static bool forward_step() {return false;}
    };
    struct InverseAdjoint{
        constexpr static bool forward_step() {return true;}
    };

    template<typename T>
    concept ForwardStepLoop = std::same_as<T, Forward> || std::same_as<T, InverseAdjoint>;

    template<typename T>
    concept ReverseStepLoop = std::same_as<T, ForwardAdjoint> || std::same_as<T, Inverse>;

    template<typename T>
    concept NormalTransform = std::same_as<T, Forward> || std::same_as<T, Inverse>;

    template<typename T>
    concept AdjointTransform = std::same_as<T, ForwardAdjoint> || std::same_as<T, InverseAdjoint>;

    template<typename T>
    concept ForwardTransform = std::same_as<T, Forward> || std::same_as<T, ForwardAdjoint>;

    template<typename T>
    concept InverseTransform = std::same_as<T, Inverse> || std::same_as<T, InverseAdjoint>;

    template<typename T, typename... Us>
    requires AllConvertible<T, Us...>
	constexpr auto update_s(const ptrdiff_t offset, Us... values) {
		return GenericUpdateStep<T, sizeof...(Us)>(
            UpdateType::UpdateEvens, offset, array<T, sizeof...(Us)>{{static_cast<T>(values)...}}
        );
	}

    template<typename T, typename... Us>
    requires AllConvertible<T, Us...>
	constexpr auto update_d(const ptrdiff_t offset, Us... values) {
		return GenericUpdateStep<T, sizeof...(Us)>(
            UpdateType::UpdateOdds, offset, array<T, sizeof...(Us)>{static_cast<T>(values)...}
        );
	}

	template<typename T, UpdateOperation PM>
	constexpr auto unit_update_s(const ptrdiff_t offset) {
		return UnitUpdateStep<T, PM>(UpdateType::UpdateEvens, offset);
	}

    template<typename T, UpdateOperation PM>
	constexpr auto unit_update_d(const ptrdiff_t offset) {
		return UnitUpdateStep<T, PM>(UpdateType::UpdateOdds, offset);
	}

    template<typename T, size_t N, typename U>
	constexpr auto repeat_update_s(const ptrdiff_t offset, const U val) {
		return RepeatUpdateStep<T, N>(UpdateType::UpdateEvens, offset, static_cast<T>(val));
	}

    template<typename T, size_t N, typename U>
	constexpr auto repeat_update_d(const ptrdiff_t offset, const U val) {
		return RepeatUpdateStep<T, N>(UpdateType::UpdateOdds, offset, static_cast<T>(val));
	}

    // Invokes f.template operator()<k>() for each index
    template <std::size_t N, typename F, std::size_t... I>
    constexpr auto tuple_generate_impl(F&& f, std::index_sequence<I...>) {
        return std::make_tuple(f.template operator()<I>()...);
    }

    template <std::size_t N, typename F>
    constexpr auto tuple_generate(F&& f) {
        return tuple_generate_impl<N>(std::forward<F>(f), std::make_index_sequence<N>{});
    }

    template<BoundaryCondition BC> struct boundary_from_enum;
    #define X(name, val) template<> struct boundary_from_enum<BoundaryCondition::name> { using type=name##Boundary;};
    LIFTED_BOUNDARY_CONDITIONS
    #undef X

    template<BoundaryCondition BC>
    using boundary_from_enum_t = typename boundary_from_enum<BC>::type;

    template<Transform OP> struct transform_from_enum;
    #define X(name, val) template<> struct transform_from_enum<Transform::name> { using type=name;};
    LIFTED_TRANSFORM_TYPES
    #undef X

    template<Transform OP>
    using transform_from_enum_t = typename transform_from_enum<OP>::type;

    constexpr static array wavelet_enum_array{
		#define X(name, value) Wavelet::name,
		LIFTED_WAVELETS
		#undef X
    };

	static inline auto wavelet_enum_to_index = []{
		std::unordered_map<Wavelet, size_t> m;
		for(size_t i = 0; i < wavelet_enum_array.size(); ++i)
			m[wavelet_enum_array[i]] = i;
		return m;
	}();

}

struct SeperableTransform{};
struct NonSeperableTransform{};

template<typename T>
concept DimSeperability = std::same_as<T, SeperableTransform> || std::same_as<T, NonSeperableTransform>;

// Boundary Conditions
using detail::ZeroBoundary;
using detail::ConstantBoundary;
using detail::PeriodicBoundary;
using detail::SymmetricBoundary;
using detail::ReflectBoundary;

template<typename T>
concept BoundCond = std::same_as<T, ZeroBoundary> ||
    std::same_as<T, ConstantBoundary> ||
    std::same_as<T, PeriodicBoundary> ||
    std::same_as<T, SymmetricBoundary> ||
    std::same_as<T, ReflectBoundary>;

template<typename T>
concept TransformType = detail::NormalTransform<T> || detail::AdjointTransform<T>;

static inline size_t max_level(const size_t width, size_t n) {
    if (width == 0) return 0;
    if (n < width - 1) return 0;
    size_t lvl = 0;
    while (n >= 2 * (width - 1)) {
        lvl += 1;
        n = (n + 1) / 2;  // If n is even, this is equivalent to n/2, if it is odd, then this is n - n/2
    }
    return lvl;
}

static inline size_t max_level(const size_t width, const size_v& shape, const size_v& axes) {
    size_t min_n = shape[axes[0]];
    for (size_t i = 1; i < shape.size(); ++i)
        if (shape[axes[i]] < min_n) min_n = shape[axes[i]];
    return max_level(width, min_n);
}

}

#endif
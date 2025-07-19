#ifndef NDARRAY_H
#define NDARRAY_H

#ifndef __cplusplus
#error This file is C++ and requires a C++ compiler.
#endif

#if !(__cplusplus >= 201103L || _MSVC_LANG+0L >= 201103L)
#error This file requires at least C++20 support.
#endif

#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
#include <utility>

#ifndef WAVELETS_NO_MULTITHREADING
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>
#include <functional>
#include <new>

#ifdef WAVELETS_PTHREADS
#  include <pthread.h>
#endif
#endif

namespace ndarray {

namespace detail {
using std::cout;
using std::endl;

using std::size_t;
using std::ptrdiff_t;

using size_v = std::vector<size_t>;
using stride_v = std::vector<ptrdiff_t>;

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

template<typename V>
static inline size_t prod(const V& shape)
{
    size_t res = 1;
    for (auto sz : shape)
        res *= sz;
    return res;
}

template<typename T, size_t ALIGN=64> class aligned_array
{
protected:
	T* p;
	size_t sz;

	static T* ralloc(size_t num)
	{
		if (num == 0) return nullptr;
		void* ptr = aligned_alloc(alignment, num * sizeof(T));
		return static_cast<T*>(ptr);
	}
	static void dealloc(T* ptr)
	{
		aligned_dealloc(ptr);
	}

public:
	using value_type = T;
    constexpr static size_t alignment = std::max(ALIGN, alignof(T));

    aligned_array() : p(0), sz(0) {}
    aligned_array(size_t n) : p(ralloc(n)), sz(n) {}
	aligned_array(aligned_array&& other) : p(other.p), sz(other.sz) {other.p = nullptr; other.sz = 0;}
	~aligned_array() {dealloc(p);}

	void resize(size_t n)
	{
		if (n == sz) return;
        T* temp = ralloc(n);
		dealloc(p);
		p = temp;
		sz = n;
	}

    // copy assignment
    aligned_array& operator=(const aligned_array& other)
    {
        if (this == &other)
            return *this;

        resize(other.sz);
        std::copy_n(other.p, other.sz, p);
        return *this;
    }

    aligned_array& operator=(aligned_array&& other)
    {
        if (this == &other)
            return *this;
        dealloc(p);
        p = std::exchange(other.p, nullptr);
        sz = std::exchange(other.sz, 0);
        return *this;
    }

	T& operator[](size_t idx) { return p[idx]; }
	const T& operator[](size_t idx) const { return p[idx]; }

    T* begin() { return p; }
    const T* cbegin() const { return p; }
    T* end() { return p + sz; }
    const T* cend() const { return p + sz; }


	T* data() { return p; }
	const T* data() const { return p; }

	size_t size() const { return sz; }

	friend std::ostream& operator<<(std::ostream& os, const aligned_array& obj)
	{
        os << "aligned_array<" << typeid(T).name() << ">, shape=(" << obj.sz << ")(" << endl;
		os << "{ ";
		size_t n = obj.size();
		for (size_t i = 0; i < n; ++i) {
			os << obj[i];
			if (i < n - 1) {
				os << ", ";
			}
		}
		os << "})";
		return os;
	}
};

namespace threading {

#ifdef WAVELETS_NO_MULTITHREADING

    constexpr inline size_t thread_id() { return 0; }
    constexpr inline size_t num_threads() { return 1; }

    template <typename Func>
    void thread_map(size_t /* nthreads */, Func f)
    {
        f();
    }

    static size_t thread_count(size_t /*nthreads*/, const size_v&/*shape*/,
        size_t /*axis*/, size_t /*vlen*/)
    { return 1; }

#else

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
#ifdef POCKETFFT_PTHREADS
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

#endif

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

template<typename T> class ndarr : public cndarr<T>
{
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

constexpr static size_t dynamic_size = static_cast<size_t>(-1);

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
    multi_iter(const arr_info& iarr_, const arr_info& oarr_, size_t idim_, size_t const n)
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

class simple_iter
{
private:
    size_v pos;
    const arr_info& arr;
    ptrdiff_t p;
    size_t rem;

public:
    simple_iter(const arr_info& arr_)
        : pos(arr_.ndim(), 0), arr(arr_), p(0), rem(arr_.size()) {}
    void advance()
    {
        --rem;
        for (int i_ = int(pos.size()) - 1; i_ >= 0; --i_)
        {
            auto i = size_t(i_);
            p += arr.stride(i);
            if (++pos[i] < arr.shape(i))
                return;
            pos[i] = 0;
            p -= ptrdiff_t(arr.shape(i)) * arr.stride(i);
        }
    }
    ptrdiff_t ofs() const { return p; }
    size_t remaining() const { return rem; }
};

class rev_iter
{
private:
    size_v pos;
    const arr_info& arr;
    std::vector<char> rev_axis;
    std::vector<char> rev_jump;
    size_t last_axis, last_size;
    size_v shp;
    ptrdiff_t p, rp;
    size_t rem;

public:
    rev_iter(const arr_info& arr_, const size_v& axes)
        : pos(arr_.ndim(), 0), arr(arr_), rev_axis(arr_.ndim(), 0),
        rev_jump(arr_.ndim(), 1), p(0), rp(0)
    {
        for (auto ax : axes)
            rev_axis[ax] = 1;
        last_axis = axes.back();
        last_size = arr.shape(last_axis) / 2 + 1;
        shp = arr.shape();
        shp[last_axis] = last_size;
        rem = 1;
        for (auto i : shp)
            rem *= i;
    }
    void advance()
    {
        --rem;
        for (int i_ = int(pos.size()) - 1; i_ >= 0; --i_)
        {
            auto i = size_t(i_);
            p += arr.stride(i);
            if (!rev_axis[i])
                rp += arr.stride(i);
            else
            {
                rp -= arr.stride(i);
                if (rev_jump[i])
                {
                    rp += ptrdiff_t(arr.shape(i)) * arr.stride(i);
                    rev_jump[i] = 0;
                }
            }
            if (++pos[i] < shp[i])
                return;
            pos[i] = 0;
            p -= ptrdiff_t(shp[i]) * arr.stride(i);
            if (rev_axis[i])
            {
                rp -= ptrdiff_t(arr.shape(i) - shp[i]) * arr.stride(i);
                rev_jump[i] = 1;
            }
            else
                rp -= ptrdiff_t(shp[i]) * arr.stride(i);
        }
    }
    ptrdiff_t ofs() const { return p; }
    ptrdiff_t rev_ofs() const { return rp; }
    size_t remaining() const { return rem; }
};

template<typename T, size_t ALIGN = 64> class aligned_ndarray : aligned_array<T, ALIGN>
{
protected:
    using base = aligned_array<T, ALIGN>;
public:
    using base::operator[];
    using base::alignment;
    using typename base::value_type;
    using base::data;

    aligned_ndarray() : base(), shp(), str() {}

    aligned_ndarray(const size_v& shape) :
        shp(shape), str(prod(shape)), base(prod(shape))
    {
        set_strides();
    }

    aligned_ndarray(const std::initializer_list<size_t> shape) :
        shp(shape), str(prod(shape)), base(prod(shape))
    {
        set_strides();
    }

    aligned_ndarray(aligned_ndarray&& other) : base(other), shp(other.shp), str(other.str)
    {
        other.shp = size_v(); other.str = stride_v();
    }

    size_t ndim() const { return shp.size(); }
    size_t size() const { return prod(shp); }
    const size_v& shape() const { return shp; }
    size_t shape(size_t i) const { return shp[i]; }
    const stride_v& stride() const { return str; }
    const ptrdiff_t& stride(size_t i) const { return str[i]; }

    cndarr<T> as_cndarray() {
        auto str_bytes = stride_v(str);
        for (auto i = 0u; i < ndim(); ++i)
            str_bytes[i] *= sizeof(T);
        auto out = cndarr<T>(data(), shp, str_bytes);
        return out;
    }

    ndarr<T> as_ndarray() {
        auto str_bytes = stride_v(str);
        for (auto i = 0u; i < ndim(); ++i)
            str_bytes[i] *= sizeof(T);
        auto out = ndarr<T>(data(), shp, str_bytes);
        return out;
    }

    template<typename... Is>
    T& operator() (Is... inds) {
        return base::data()[get_flat_index(inds...)];
    }

    template<typename... Is>
    const T& operator() (Is... inds) const {
        return base::data()[get_flat_index(inds...)];
    }

    template<typename V>
    void reshape(V& shape)
    {
        size_t n = prod(shape);
        shp = shape;
        str = stride_v(shp.size());
        base::resize(n);
        set_strides();
    }

    aligned_ndarray& operator=(aligned_ndarray& other)
    {
        reshape(other.shp);
        std::memcpy(base::p, other.p, other.sz * sizeof(T));
        return *this;
    }

    aligned_ndarray& operator=(aligned_ndarray&& other)
    {
        base::dealloc(base::p);
        base::p = other.p;
        base::sz = other.sz;
        shp = other.shp;
        str = other.str;

        other.shp = size_v();
        other.str = stride_v();
        other.p = nullptr; other.sz = 0;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const aligned_ndarray& obj)
    {
        auto pos = size_v(obj.ndim(), 0);
        auto p = ptrdiff_t(0);
        auto rem = obj.size();

        // simple_iter it(obj);
        os << "aligned_ndarray<" << typeid(T).name() << ">, shape=(";

        auto nd = obj.ndim();
        for (size_t i = 0; i < nd; ++i) {
            os << obj.shp[i];
            if (i < nd - 1) {
                os << ", ";
            }
        }
        os << ")(" << std::endl;

        os << std::string(nd, '{');
        auto indent = std::string(nd - 1, ' ');
        while (rem-- > 0) {
            os << obj[p];
            for (int i_ = int(nd) - 1; i_ >= 0; --i_)
            {
                auto i = size_t(i_);
                p += obj.stride(i);
                if (pos[i] < obj.shape(i) - 1) {
                    if (i == nd - 1) {
                        os << ", ";
                    }
                    else {
                        os << ",";
                        for (int k = 0; k < nd - i - 1; ++k)
                            os << endl;
                        os << std::string(i + 1, ' ');
                        os << std::string(nd - i - 1, '{');
                    }
                }
                if (++pos[i] < obj.shape(i)) {
                    break;
                }
                os << "}";
                pos[i] = 0;
                p -= ptrdiff_t(obj.shape(i)) * obj.stride(i);
            }
        }
        os << ")";
        return os;
    }

private:
    size_v shp;
    stride_v str;

    void set_strides() {
        str[ndim() - 1] = 1; // strides here are in data_type units.
        for (size_t i = ndim() - 1; i-- > 0;) {
            str[i] = shp[i + 1] * str[i + 1];
        }
    }

    template<typename... Is>
    size_t get_flat_index(Is... inds) const {
        size_v idx_arr = { static_cast<size_t>(inds)... };
        if (idx_arr.size() != ndim()) throw std::invalid_argument("Inconsistent number of passed indices");
        size_t flat_index = 0;
        for (size_t ax = 0; ax < ndim(); ++ax)
            flat_index += str[ax] * idx_arr[ax];
        return flat_index;
    }
};

}

using detail::aligned_array;
using detail::aligned_ndarray;
using detail::cndarr;
using detail::ndarr;
using detail::multi_iter;
using detail::size_v;
using detail::stride_v;
using detail::prod;

}

#endif
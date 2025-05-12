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

namespace ndarray {

namespace detail {

using std::size_t;
using std::ptrdiff_t;
using std::cout;
using std::endl;

using shape_t = std::vector<size_t>;
using stride_t = std::vector<ptrdiff_t>;

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
	aligned_array(aligned_array&& other)
		: p(other.p), sz(other.sz)
	{
		other.p = nullptr; other.sz = 0;
	}
	~aligned_array() { dealloc(p); }

	void resize(size_t n)
	{
		if (n == sz) return;
		dealloc(p);
		p = ralloc(n);
		sz = n;
	}

    aligned_array& operator=(aligned_array& other)
    {
        resize(other.sz);
        std::memcpy(p, other.p, other.sz * sizeof(T));
        return *this;
    }

    aligned_array& operator=(aligned_array&& other)
    {
        dealloc(p);
        p = other.p;
        sz = other.sz;
        other.p = nullptr; other.sz = 0;
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

class arr_info
{
protected:
	shape_t shp;
	stride_t str;

public:
	arr_info(const shape_t& shape_, const stride_t& stride_)
		: shp(shape_), str(stride_) {}
	size_t ndim() const { return shp.size(); }
	size_t size() const { return prod(shp); }
	const shape_t& shape() const { return shp; }
	size_t shape(size_t i) const { return shp[i]; }
	const stride_t& stride() const { return str; }
	const ptrdiff_t& stride(size_t i) const { return str[i]; }
};

template<typename T> class cndarr : public arr_info
{
protected:
	const char* d;

public:
	cndarr(const void* data_, const shape_t& shape_, const stride_t& stride_)
		: arr_info(shape_, stride_),
		d(reinterpret_cast<const char*>(data_)) {}
	const T& operator[](ptrdiff_t ofs) const
	{
		return *reinterpret_cast<const T*>(d + ofs);
	}
};

template<typename T> class ndarr : public cndarr<T>
{
public:
	ndarr(void* data_, const shape_t& shape_, const stride_t& stride_)
		: cndarr<T>::cndarr(const_cast<const void*>(data_), shape_, stride_)
	{}
	T& operator[](ptrdiff_t ofs)
	{
		return *reinterpret_cast<T*>(const_cast<char*>(cndarr<T>::d + ofs));
	}
};

template<size_t N> class multi_iter
{
private:
    shape_t pos;
    const arr_info& iarr, & oarr;
    ptrdiff_t p_ii, p_i[N], str_i, p_oi, p_o[N], str_o;
    size_t idim, rem;

    void advance_i()
    {
        for (int i_ = int(pos.size()) - 1; i_ >= 0; --i_)
        {
            auto i = size_t(i_);
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
        auto nshares = 1; //threading::num_threads();
        if (nshares == 1) return;
        if (nshares == 0) throw std::runtime_error("can't run with zero threads");
        auto myshare = 0; // threading::thread_id();
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
};

class simple_iter
{
private:
    shape_t pos;
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
    shape_t pos;
    const arr_info& arr;
    std::vector<char> rev_axis;
    std::vector<char> rev_jump;
    size_t last_axis, last_size;
    shape_t shp;
    ptrdiff_t p, rp;
    size_t rem;

public:
    rev_iter(const arr_info& arr_, const shape_t& axes)
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

    aligned_ndarray(const shape_t& shape) : 
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
        other.shp = shape_t(); other.str = stride_t();
    }

    size_t ndim() const { return shp.size(); }
    size_t size() const { return prod(shp); }
    const shape_t& shape() const { return shp; }
    size_t shape(size_t i) const { return shp[i]; }
    const stride_t& stride() const { return str; }
    const ptrdiff_t& stride(size_t i) const { return str[i]; }

    cndarr<T> as_cndarray() {
        auto str_bytes = stride_t(str);
        for (auto i = 0u; i < ndim(); ++i)
            str_bytes[i] *= sizeof(T);
        auto out = cndarr<T>(data(), shp, str_bytes);
        return out;
    }

    ndarr<T> as_ndarray() {
        auto str_bytes = stride_t(str);
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
    const T& operator() (Is... inds) const{
        return base::data()[get_flat_index(inds...)];
    }

    template<typename V>
    void reshape(V& shape)
    {
        size_t n = prod(shape);
        shp = shape;
        str = stride_t(shp.size());
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

        other.shp = shape_t();
        other.str = stride_t();
        other.p = nullptr; other.sz = 0;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream & os, const aligned_ndarray& obj)
    {
        auto pos = shape_t(obj.ndim(), 0);
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
    shape_t shp;
    stride_t str;

    void set_strides() {
        str[ndim() - 1] = 1; // strides here are in data_type units.
        for (size_t i = ndim() - 1; i-- > 0;) {
            str[i] = shp[i + 1] * str[i + 1];
        }
    }

    template<typename... Is>
    size_t get_flat_index(Is... inds) const {
        shape_t idx_arr = { static_cast<size_t>(inds)... };
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
using detail::ndarr;
using detail::shape_t;
using detail::stride_t;
}

#endif
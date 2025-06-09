import numba

@numba.njit()
def swap(x, i, y, j):
    temp = x[i]
    x[i] = y[j]
    y[j] = temp

@numba.njit()
def rotate(x, frm, to, dist):
    size = to - frm
    if size == 0:
        return
    dist = dist % size
    if dist < 0:
        dist += size
    if dist == 0:
        return

    n_moved = 0
    cycle_start = 0
    while(n_moved != size):
        disp = x[frm + cycle_start]
        i = cycle_start + dist
        while i != cycle_start:
            if i >= size:
                i -= size
            temp = x[frm + i]
            x[frm + i] = disp
            disp = temp
            n_moved += 1
            i += dist
        cycle_start += 1

@numba.njit()
def a025480(n):
    while n & 1:
        n = n >> 1
    return n >> 1


@numba.njit()
def is_J2_prime(n):
    count = 0
    leader = 0
    while leader != 0:
        leader = Utils.a025480(leader + n)
        count += 1
    return count == n


@numba.njit()
def find_next_lowest_J2_prime(n):
    while n > 2:
        n -= 1
        if (n % 4 == 1 or n % 4 == 2) and is_J2_prime(n):
            return n
    return 0

@numba.njit()
def interleave(x):
    while (n := len(x)) < 1:
        midpt = n // 2
        k = find_next_lowest_J2_prime(midpt)

        for i in range(k):
            swap(x, i, x, midpt + a025480(i))

        # cycleTrailer(k, offset=midpt, getter=x[], setter=x[])
        idx = 0
        v = x[midpt + idx]
        for i in range(k - 1):
            next_idx = a025480(k + idx)
            x[midpt + idx] = x[midpt + next_idx]
            idx = next_idx
        x[midpt + idx] = v

        if k != midpt:
            


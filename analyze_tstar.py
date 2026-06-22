"""Analyze the S3 fixed-point transition table T*."""
import numpy as np
from collections import Counter

T = np.load("results/s3_fixed_point_T_star.npy")
n = len(T)  # 729 = 3^6
q, v = 3, 6

def to_ternary(s):
    digits = []
    for _ in range(v):
        digits.append(s % q)
        s //= q
    return tuple(reversed(digits))

def from_ternary(digits):
    s = 0
    for d in digits:
        s = s * q + d
    return s

states = [to_ternary(i) for i in range(n)]

print("="*60)
print("T* STRUCTURAL ANALYSIS")
print("="*60)

# 1. Verify single cycle
print(f"\nStates: {n}, q={q}, v={v}")
print(f"Is permutation: {sorted(T.tolist()) == list(range(n))}")
visited = set()
c = 0
cycle_len = 0
while c not in visited:
    visited.add(c)
    c = int(T[c])
    cycle_len += 1
print(f"Single cycle length: {cycle_len}")

# 2. How does T* act on each coordinate?
print("\n--- Per-coordinate action ---")
for coord in range(v):
    changes = []
    for s in range(n):
        old = states[s][coord]
        new = states[int(T[s])][coord]
        changes.append((new - old) % q)
    cnt = Counter(changes)
    print(f"  x[{coord}]: delta distribution = {dict(sorted(cnt.items()))}")

# 3. Is T* affine over F_3? Check if T*(s) = A*s + b (mod 3)
print("\n--- Affine test over F_3^6 ---")
zero_state = from_ternary((0,)*v)
b = to_ternary(int(T[zero_state]))
print(f"  T*(000000) = {b}  (candidate offset b)")

# Check if T*(s) - b is linear
is_linear = True
# Build candidate matrix from basis vectors
A = np.zeros((v, v), dtype=int)
for j in range(v):
    basis = [0]*v
    basis[j] = 1
    s = from_ternary(basis)
    img = to_ternary(int(T[s]))
    col = [(img[i] - b[i]) % q for i in range(v)]
    A[:, j] = col

print(f"  Candidate matrix A:\n{A}")

# Verify A*s + b = T*(s) for all states
n_match = 0
n_fail = 0
for s in range(n):
    sv = np.array(states[s], dtype=int)
    predicted = (A @ sv + np.array(b, dtype=int)) % q
    actual = np.array(states[int(T[s])], dtype=int)
    if np.array_equal(predicted, actual):
        n_match += 1
    else:
        n_fail += 1
        if n_fail <= 3:
            print(f"  FAIL at state {states[s]}: predicted {tuple(predicted)} != actual {tuple(actual)}")

print(f"  Affine match: {n_match}/{n} ({100*n_match/n:.1f}%)")
if n_match == n:
    print("  *** T* IS AFFINE over F_3^6 ***")
    print(f"  T*(x) = A*x + b (mod 3)")
    print(f"  A = \n{A}")
    print(f"  b = {b}")
    # Check if A has order dividing 729 (as expected for single cycle)
    M = np.eye(v, dtype=int)
    for k in range(1, 730):
        M = (M @ A) % q
        if np.array_equal(M, np.eye(v, dtype=int)):
            print(f"  Order of A in GL(6,F_3): {k}")
            break
    det = int(round(np.linalg.det(A))) % q
    print(f"  det(A) mod 3 = {det}")
else:
    print("  T* is NOT affine.")

# 4. Hamming distance analysis
print("\n--- Hamming distance: state vs T*(state) ---")
hamming = []
for s in range(n):
    h = sum(1 for i in range(v) if states[s][i] != states[int(T[s])][i])
    hamming.append(h)
cnt = Counter(hamming)
print(f"  Hamming distance distribution: {dict(sorted(cnt.items()))}")
print(f"  Mean Hamming distance: {np.mean(hamming):.3f}")

# 5. Sum conservation
print("\n--- Sum (mod 3) conservation ---")
sum_changes = []
for s in range(n):
    old_sum = sum(states[s]) % q
    new_sum = sum(states[int(T[s])]) % q
    sum_changes.append((new_sum - old_sum) % q)
cnt = Counter(sum_changes)
print(f"  Sum change distribution: {dict(sorted(cnt.items()))}")
if len(cnt) == 1:
    print(f"  *** Sum mod 3 shifts by {list(cnt.keys())[0]} every step ***")

# 6. Coordinate-pair correlations
print("\n--- Coordinate pair correlations in T* ---")
for i in range(v):
    for j in range(i+1, v):
        # Does knowing (x[i], x[j]) determine (T(x)[i], T(x)[j])?
        mapping = {}
        deterministic = True
        for s in range(n):
            key = (states[s][i], states[s][j])
            val = (states[int(T[s])][i], states[int(T[s])][j])
            if key in mapping and mapping[key] != val:
                deterministic = False
                break
            mapping[key] = val
        if deterministic:
            print(f"  (x[{i}],x[{j}]) -> (T(x)[{i}],T(x)[{j}]) is DETERMINISTIC")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

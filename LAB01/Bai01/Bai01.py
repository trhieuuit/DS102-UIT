
import numpy as np

# Ham can tim cuc tri
def f(x):
    x1, x2 = x
    return -(x1 - 2)**2 - 2*(x2 - 3)**2 + 10


def numerical_gradient(f, x, h=1e-5):
    x = np.array(x, dtype=float)
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return grad


# Ham Hessian bang sai phan huu han
def numerical_hessian(f, x, h=1e-4):
    x = np.array(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))

    fx = f(x)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        # Dao ham rieng bac hai theo x_i
        H[i, i] = (
            f(x_plus) - 2 * fx + f(x_minus)
        ) / (h ** 2)

        # Dao ham rieng hon hop
        for j in range(i + 1, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h
            x_pp[j] += h

            x_pm[i] += h
            x_pm[j] -= h

            x_mp[i] -= h
            x_mp[j] += h

            x_mm[i] -= h
            x_mm[j] -= h

            value = (
                f(x_pp) - f(x_pm)
                - f(x_mp) + f(x_mm)
            ) / (4 * h ** 2)

            H[i, j] = value
            H[j, i] = value

    return H


# Thuat toan Newton-Raphson
def newton_raphson(f, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        grad = numerical_gradient(f, x)

        if np.linalg.norm(grad) < tol:
            break

        H = numerical_hessian(f, x)

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            print("Hessian suy bien.")
            return x, False

        x = x - delta

    else:
        print("Da dat so lan lap toi da.")
        return x, False

    return x, True


# Phan loai diem dung
def classify_extremum(f, x):
    H = numerical_hessian(f, x)
    eigenvalues = np.linalg.eigvalsh(H)
    eps = 1e-5

    if np.all(eigenvalues > eps):
        return "Cuc tieu"
    elif np.all(eigenvalues < -eps):
        return "Cuc dai"
    elif np.any(eigenvalues > eps) and np.any(eigenvalues < -eps):
        return "Diem yen ngua"
    else:
        return "Khong du dieu kien ket luan"





# Diem khoi tao
x0 = np.array([0.0, 0.0])

# Thuc hien Newton-Raphson
x_star, success = newton_raphson(f, x0)

if success:
    print("Diem dung:", x_star)
    print("Gia tri ham:", f(x_star))
    print("Phan loai:", classify_extremum(f, x_star))
else:
    print("Chua tim duoc diem dung hoi tu.")

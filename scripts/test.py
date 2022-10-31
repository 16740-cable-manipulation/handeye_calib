import numpy as np


def compute_rotation(checker_center, trans):
    vz = checker_center - trans
    vz = vz / np.linalg.norm(vz)
    vx = np.array([1.0, 0.0, 0.0])
    vy = np.cross(vz, vx)
    vy = vy / np.linalg.norm(vy)
    R = np.vstack((vx, vy, vz)).T
    return R


# a list of poses the ee should go to
dx = 10
dy = 5
dz = 2
nx = 3
ny = 3
nz = 3
cx, cy, cz = (0, 0, 0)
x_ = np.linspace(cx - dx, cx + dx, nx)
y_ = np.linspace(cy - dy, cy + dy, ny)
z_ = np.linspace(cz + dz, cz + 2 * dz, nz)

x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
# at each (x,y,z) point, compute the rotation
translations = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
rotations = [
    compute_rotation(np.array([0, 0, 0]), trans) for trans in translations
]
print(len(rotations), len(translations))
a = [
    {"R": rotations[i], "t": translations[i]} for i in range(len(translations))
]
print(a[2])


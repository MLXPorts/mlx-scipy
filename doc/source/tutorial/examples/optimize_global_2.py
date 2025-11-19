import mlx.core as mx
import matplotlib.pyplot as plt

def eggholder(x):
    return (-(x[1] + 47) * mx.sin(mx.sqrt(abs(x[0]/2 + (x[1]  + 47))))
            -x[0] * mx.sin(mx.sqrt(abs(x[0] - (x[1]  + 47)))))

bounds = [(-512, 512), (-512, 512)]

x = mx.arange(-512, 513)
y = mx.arange(-512, 513)
xgrid, ygrid = mx.meshgrid(x, y)
xy = mx.stack([xgrid, ygrid])

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, eggholder(xy), cmap='terrain')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('eggholder(x, y)')

fig.tight_layout()
plt.show()

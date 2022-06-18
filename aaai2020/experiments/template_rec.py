from jinja2 import Template
import numpy as np

class RegionNode:
    def __init__(
        self, num_buckets=3, eps=0.01,
        lx=-1, hx=1,
        ly=-1, hy=1,
    ):
        # 6 7 8
        # 3 4 5
        # 0 1 2
        self.children = [None for _ in range(num_buckets ** 2)]

        self.eps = 0.1
        self.B = num_buckets

        self.lx = lx
        self.hx = hx
        self.ly = ly
        self.hy = hy

        self.xs = np.linspace(lx, hx, self.B+1)
        self.xs[-1] += eps
        self.ys = np.linspace(ly, hy, self.B+1)
        self.ys[-1] += eps

        self.num_dots = 0

    def single_region(self, x, xs):
        in_region = (xs[:-1] <= x) & (x < xs[1:])
        if not in_region.any():
            raise ValueError
        if in_region.sum() > 1:
            raise ValueError
        return in_region.argmax()

    def x_region(self, x):
        return self.single_region(x, self.xs)

    def y_region(self, y):
        return self.single_region(y, self.ys)

    def xy_region(self, xy):
        x, y = xy
        x_region = self.x_region(x)
        y_region = self.y_region(y)
        return x_region, y_region

    def add(self, xy):
        self.num_dots += 1
        x_region, y_region = self.xy_region(xy)
        #flat_region = self.B * x_region + y_region
        flat_region = self.B * y_region + x_region

        node = self.children[flat_region]
        if node is None:
            # singleton node
            self.children[flat_region] = xy
        elif not isinstance(node, RegionNode):
            # convert singleton node into real node
            old_xy = node
            # overwrite singleton
            node = RegionNode(
                num_buckets = self.B,
                eps = self.eps,
                lx = self.xs[x_region],
                hx = self.xs[x_region+1],
                ly = self.ys[y_region],
                hy = self.ys[y_region+1],
            )
            # add singleton back
            node.add(old_xy)
            # add new point
            node.add(xy)
            self.children[flat_region] = node
        else:
            node.add(xy)


def main():
    import numpy as np
    import jax
    from jax import random
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib import cm

    lv = -1
    hv = 1
    B = 2

    key = random.PRNGKey(0)
    fig, axes = plt.subplots(5,5)
    N = len(axes.flat)
    uk, key = random.split(key)
    xys = random.uniform(uk, minval=lv, maxval=hv, shape=(N, 4,2))
    for i, ax in enumerate(axes.flat):
        xy = xys[i]
        ax.scatter(xy[:,0], xy[:,1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_title(f"{i}")
    plt.savefig("img/dots.png")

    for n in range(N):
        xy = xys[n]
        root = RegionNode(num_buckets=B)
        root.add(xy[0])
        root.add(xy[1])
        root.add(xy[2])
        root.add(xy[3])
        import pdb; pdb.set_trace()



if __name__ == "__main__":
    main()


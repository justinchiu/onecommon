from jinja2 import Template
import numpy as np

import matplotlib.patches as patches
import matplotlib.lines as lines


class RegionNode:
    def __init__(
        self,
        num_buckets = 3,
        eps = 1e-7,
        absolute_bucket = True,
        inner_buckets = None,
        lx = -1, hx = 1,
        ly = -1, hy = 1,
    ):
        # 6 7 8
        # 3 4 5
        # 0 1 2
        self.children = [None for _ in range(num_buckets ** 2)]

        self.eps = 0.1
        self.B = num_buckets
        self.inner_B = num_buckets if inner_buckets is None else inner_buckets
        self.absolute_bucket = absolute_bucket

        self.lx = lx
        self.hx = hx
        self.ly = ly
        self.hy = hy

        self.xs = np.linspace(lx, hx, self.B+1)
        self.xs_pad = np.linspace(lx, hx, self.B+1)
        self.xs_pad[-1] += eps
        self.ys = np.linspace(ly, hy, self.B+1)
        self.ys_pad = np.linspace(ly, hy, self.B+1)
        self.ys_pad[-1] += eps

        self.num_dots = 0

    def lines(self):
        # draw lines for inner boundaries
        for x in self.xs[1:-1]:
            yield lines.Line2D((x, x), (self.ly, self.hy))
        for y in self.ys[1:-1]:
            yield lines.Line2D((self.lx, self.hx), (y, y))

    def items(self):
        yield self
        for child in self.children:
            if isinstance(child, RegionNode):
                yield from child.items()

    def single_region(self, x, xs):
        in_region = (xs[:-1] <= x) & (x < xs[1:])
        if not in_region.any():
            raise ValueError
        if in_region.sum() > 1:
            raise ValueError
        return in_region.argmax()

    def x_region(self, x):
        return self.single_region(x, self.xs_pad)

    def y_region(self, y):
        return self.single_region(y, self.ys_pad)

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

            # OPTION 1: segment based on TREE REGIONS
            node = RegionNode(
                num_buckets = self.inner_B,
                eps = self.eps,
                absolute_bucket = self.absolute_bucket,
                lx = self.xs[x_region],
                hx = self.xs[x_region+1],
                ly = self.ys[y_region],
                hy = self.ys[y_region+1],
            ) if self.absolute_bucket else RegionNode(
                # OPTION 2: segment based in positions of nodes
                # POSSIBLY BREAKS DOWN DEPENDING ON ORDERING OF ADD
                # RELIES ON ASSM THAT ONLY 2 NODES IN REGION
                num_buckets = self.inner_B,
                eps = self.eps,
                absolute_region = self.absolute_bucket,
                lx = min(xy[0], old_xy[0]),
                hx = max(xy[0], old_xy[0]),
                ly = min(xy[1], old_xy[1]),
                hy = max(xy[1], old_xy[1]),
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
    #B = 3
    inner_B = None
    #inner_B = 2
    absolute_bucket = False

    key = random.PRNGKey(0)
    fig, axes = plt.subplots(5,5)
    N = len(axes.flat)
    uk, key = random.split(key)
    xys = random.uniform(uk, minval=lv, maxval=hv, shape=(N, 4,2))

    """
    n = 6
    xy = xys[n]
    root = RegionNode(
        num_buckets = B,
        lx = xy[:,0].min(),
        hx = xy[:,0].max(),
        ly = xy[:,1].min(),
        hy = xy[:,1].max(),
    )
    root.add(xy[0])
    root.add(xy[1])
    root.add(xy[2])
    root.add(xy[3])

    print(xy)
    for node in root.items():
        for line in node.lines():
            print(line.get_xydata())
            import pdb; pdb.set_trace()
    """

    for n in range(N):
        xy = xys[n]
        root = RegionNode(
            num_buckets = B,
            absolute_bucket = absolute_bucket,
            lx = xy[:,0].min(),
            hx = xy[:,0].max(),
            ly = xy[:,1].min(),
            hy = xy[:,1].max(),
        )
        root.add(xy[0])
        root.add(xy[1])
        root.add(xy[2])
        root.add(xy[3])

        ax = axes.flat[n]
        ax.scatter(xy[:,0], xy[:,1])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_xlim(-1,1)
        #ax.set_ylim(-1,1)
        ax.set_title(f"{n}")

        for node in root.items():
            for line in node.lines():
                ax.add_line(line)
    plt.savefig(f"img/B{B}-IB{inner_buckets}-A{absolute_bucket}-dots.png")



if __name__ == "__main__":
    main()


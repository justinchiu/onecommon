from jinja2 import Template
import numpy as np

import matplotlib.patches as patches
import matplotlib.lines as lines

from template import (
    mention_1,
    mention_2,
    mention_2a,
    mention_3,
    mention_41,
    mention_42,
    mention_43,
    dot_template,
    spatial_dot_template,
    render_2,
    size_map,
    color_map,
    get_sign,
    check_triangle
)

def render_2_dots(dots, flip_y=True):
    sc = np.array([
        [dots[0].size, dots[0].color],
        [dots[1].size, dots[1].color],
    ])
    xy = np.vstack((dots[0].xy, dots[1].xy))
    return render_2(xy, sc, flip_y, concise=True)

class Dot:
    def __init__(self, id, size, color, xy):
        self.id = id
        self.size = size
        self.color = color
        self.xy = xy


class RegionNode:
    def __init__(
        self,
        num_buckets = 3,
        eps = 1e-3,
        absolute_bucket = True,
        inner_buckets = None,
        lx = -1, hx = 1,
        ly = -1, hy = 1,
        flip_y = True,
    ):
        self.children = [None for _ in range(num_buckets ** 2)]

        self.flip_y = flip_y
        self.top_word = "bottom" if flip_y else "top"
        self.bottom_word = "top" if flip_y else "bottom"

        # 6 7 8
        # 3 4 5
        # 0 1 2
        self.region_map_3 = [
            f"{self.bottom_word}-left",   # 0
            f"{self.bottom_word}-middle", # 1
            f"{self.bottom_word}-right",  # 2
            f"middle-left",               # 3
            f"middle",                    # 4
            f"middle-right",              # 5
            f"{self.top_word}-left",      # 6
            f"{self.top_word}-middle",    # 7
            f"{self.top_word}-right",     # 8
        ]

        # 2 3
        # 0 1
        self.region_map_2 = [
            f"{self.bottom_word}-left",   # 0
            f"{self.bottom_word}-right",  # 1
            f"{self.top_word}-left",      # 2
            f"{self.top_word}-right",     # 3
        ]

        self.eps = eps
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
        self.dots = {r: [] for r in range(self.B**2)}

    def desc(self):
        # flat template, first attempt at generating regions
        non_empty_regions, children, num_dots = list(zip(*[
            (
                region,
                child,
                child.num_dots if isinstance(child, RegionNode) else 1,
            )
            for region, child in enumerate(self.children)
            if child is not None
        ]))

        number_descriptions = [
            "is 1 dot" if x == 1 else f"are {x} dots"
            for x in num_dots
        ]
        region_descriptions = [
            self.region_map_3[region]
            for region in non_empty_regions
        ]
        flattened_children = [
            [node for nodes in child.dots.values() for node in nodes]
                if isinstance(child, RegionNode)
                else [child]
            for child in children
        ]

        triangle = False
        dot_list = sum(flattened_children, [])
        if len(self.xs) == 4:
            xy = [dot_list[i].xy for i in range(4)]
            for i in range(4):
                compl = list(set([0, 1, 2, 3]) - set([i]))
                res = check_triangle(xy[i], xy[compl[0]], xy[compl[1]], xy[compl[2]])
                if res:
                    dot4 = i
                    dot1, dot2, dot3 = compl[0], compl[1], compl[2]
                    triangle = True
                    break

        if triangle:
            child_descs = [
                dot_template.render(
                    size = size_map[children.size],
                    color = color_map[children.color],
                ) 
                for children in dot_list
            ]
            if len(region_descriptions) < 4:
                new_regions = []
                for i in range(len(region_descriptions)):
                    if len(flattened_children[i]) > 1:
                        new_regions += [region_descriptions[i]] * len(flattened_children[i])
                    else:
                        new_regions.append(region_descriptions[i])
                region_descriptions = new_regions

            desc = Template(
                "Do you see a triangle with a dot inside? "
                "The triangle has a {{dot1f}} dot in {{dot1r}}, "
                "a {{dot2f}} dot in {{dot2r}}, "
                "and a {{dot3f}} dot in {{dot3r}}. "
                "The dot inside the triangle is {{dot4f}} in {{dot4r}}."
            ).render(
                dot1f = child_descs[dot1],
                dot1r = region_descriptions[dot1],
                dot2f = child_descs[dot2],
                dot2r = region_descriptions[dot2],
                dot3f = child_descs[dot3],
                dot3r = region_descriptions[dot3],
                dot4f = child_descs[dot4],
                dot4r = region_descriptions[dot4],
            )
        else:
            # WARNING: will break if there are 3 in a single 2nd level region
            child_descs = [
                dot_template.render(
                    size = size_map[children[0].size],
                    color = color_map[children[0].color],
                ) if len(children) == 1 else render_2_dots(children, flip_y=self.flip_y)
                for children in flattened_children
            ]

            desc = Template(
                "Does this fit any configuration of {{ndots}} dots? "
                "{% for desc in descs %}"
                "There {{desc[0]}} in the {{desc[1]}}: {{desc[2]}}. "
                "{% endfor %}"
            ).render(
                ndots = self.num_dots,
                descs = zip(number_descriptions, region_descriptions, child_descs),
            )
        return desc


    def template(self):
        n = self.num_dots
        if n == 1:
            import pdb; pdb.set_trace()
        elif n == 2:
            import pdb; pdb.set_trace()
        elif n == 3:
            import pdb; pdb.set_trace()
        else:
            raise ValueError("Too many dots in lower level")

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
        return in_region.argmax().item()

    def x_region(self, x):
        return self.single_region(x, self.xs_pad)

    def y_region(self, y):
        return self.single_region(y, self.ys_pad)

    def xy_region(self, xy):
        x, y = xy
        x_region = self.x_region(x)
        y_region = self.y_region(y)
        return x_region, y_region

    def add(self, dot):
        self.num_dots += 1
        xy = dot.xy
        x_region, y_region = self.xy_region(xy)
        #flat_region = self.B * x_region + y_region
        flat_region = self.B * y_region + x_region

        # some book-keeping for easy access to dots
        self.dots[flat_region].append(dot)

        node = self.children[flat_region]
        if node is None:
            # create singleton node
            self.children[flat_region] = dot
        elif not isinstance(node, RegionNode):
            # convert singleton node into RegionNode
            old_dot = node
            old_xy = old_dot.xy
            # overwrite singleton

            # OPTION 1: segment based on absolute TREE REGIONS
            node = RegionNode(
                num_buckets = self.inner_B,
                eps = self.eps,
                absolute_bucket = self.absolute_bucket,
                flip_y = self.flip_y,
                lx = self.xs[x_region],
                hx = self.xs[x_region+1],
                ly = self.ys[y_region],
                hy = self.ys[y_region+1],
            ) if self.absolute_bucket else RegionNode(
                # OPTION 2: segment based in positions of nodes
                # BREAKS DOWN DEPENDING ON ORDERING OF ADD
                # RELIES ON ASSM THAT ONLY 2 NODES IN REGION
                num_buckets = self.inner_B,
                eps = self.eps,
                absolute_bucket = self.absolute_bucket,
                flip_y = self.flip_y,
                lx = min(xy[0], old_xy[0]),
                hx = max(xy[0], old_xy[0]),
                ly = min(xy[1], old_xy[1]),
                hy = max(xy[1], old_xy[1]),
            )

            # add singleton back
            node.add(old_dot)
            # add new point
            node.add(dot)
            self.children[flat_region] = node
        else:
            # already a RegionNode
            node.add(dot)

def render(n, sc, xy, flip_y=True):
    if n == 2:
        return render_2(xy, sc, flip_y=flip_y)

    root = RegionNode(
        num_buckets = 3,
        inner_buckets = 2,
        absolute_bucket = True,
        flip_y = flip_y,
        lx = xy[:,0].min(),
        hx = xy[:,0].max(),
        ly = xy[:,1].min(),
        hy = xy[:,1].max(),
    )
    # convert to dots
    dots = [Dot(i, sc[i,0], sc[i,1], xy[i]) for i in range(n)]
    for dot in dots:
        root.add(dot)
    return root.desc()


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
    inner_B = None

    B = 3
    inner_B = 2
   
    absolute_bucket = True
    #absolute_bucket = False

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

    xy = xys[0]
    dots = [Dot(0,0,0,xy[0]), Dot(1,2,2,xy[1])]
    out = render_2_dots(dots, flip_y=False)

    for n in range(N):
        xy = xys[n]
        root = RegionNode(
            num_buckets = B,
            inner_buckets = inner_B,
            absolute_bucket = absolute_bucket,
            flip_y = False,
            lx = xy[:,0].min(),
            hx = xy[:,0].max(),
            ly = xy[:,1].min(),
            hy = xy[:,1].max(),
        )
        for i, xyi in enumerate(xy):
            root.add(Dot(i, 1, 1, xyi))

        desc = root.desc()
        print(f"{n}: {desc}")

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
    plt.savefig(f"img/B{B}-IB{inner_B}-A{absolute_bucket}-dots.png")



if __name__ == "__main__":
    main()

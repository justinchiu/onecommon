
class Dot:
    def __init__(self, item):
        for k,v in item.items():
            setattr(self, k, v)

    def html(self, shift=0, value=None):
        x = self.x + shift
        y = self.y
        r = self.size
        f = self.color
        label = (f'<text x="{x+12}" y="{y-12}" font-size="18">{self.id}</text>'
            if value is None
            else f'<text x="{x+12}" y="{y-12}" font-size="18">{self.id} ({value:.2f})</text>'
        )
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f}" /> {label}'

    def select_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 2
        f = self.color # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="red" stroke-width="3" stroke-dasharray="3,3"  />'

    def intersect_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 4
        f = self.color # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="blue" stroke-width="3" stroke-dasharray="3,3"  />'

    def __repr__(self):
        return f"Dot {self.id}: ({self.x}, {self.y}) r={self.size} f={self.color}"

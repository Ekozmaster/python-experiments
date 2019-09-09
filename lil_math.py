import math


class Vector3:
    x = y = z = 0.0

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    # Algebra Operators
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)

    def __mul__(self, other=1.0):
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __str__(self):
        return "({0}, {1}, {2})".format(self.x, self.y, self.z)

    def normalize(self):
        mag = self.magnitude()
        self.x /= mag
        self.y /= mag
        self.z /= mag

    def normalized(self):
        v = Vector3(self.x, self.y, self.z)
        v.normalize()
        return v

    def magnitude(self):
        return math.sqrt(self.sqr_magnitude())

    def sqr_magnitude(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z


def test():
    a = Vector3(1,1,0)
    b = Vector3(0.7071, 0.7071, 0)
    print(a.normalized())
    print(a)

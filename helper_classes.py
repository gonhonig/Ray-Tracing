from __future__ import annotations
from abc import abstractmethod
import numpy as np
from numpy import ndarray


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity

    # This function returns the ray that goes from a point to the light source
    @abstractmethod
    def get_light_ray(self, intersection_point) -> Ray:
        pass

    # This function returns the distance from a point to the light source
    @abstractmethod
    def get_distance_from_light(self, intersection):
        pass

    # This function returns the light intensity at a point
    @abstractmethod
    def get_intensity(self, intersection):
        pass


class DirectionalLight(LightSource):
    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection_point):
        return Ray(intersection_point, -self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.dot(intersection, self.direction)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection):
        return Ray(intersection, self.position - intersection)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(self.position - intersection)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # calculate and return the light intensity based on kc, kl, kq
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl * d + self.kq * (d ** 2) + 1e-4)


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = normalize(np.array(direction))
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, self.position - intersection)

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(self.position - intersection)

    def get_intensity(self, intersection):
        v = normalize(self.position - intersection)
        d = self.get_distance_from_light(intersection)
        return (self.intensity * max(0, v.dot(self.direction))) / (self.kc + self.kl * d + self.kq * (d ** 2))


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects: list[Object3D]) -> (Object3D, float, ndarray):
        intersections = None
        nearest_object: Object3D = None
        min_distance = np.inf
        nearest_object_normal = None

        for obj in objects:
            intersected_obj, distance, normal = obj.intersect(self)
            if distance is not None and distance < min_distance:
                min_distance = distance
                nearest_object = intersected_obj
                nearest_object_normal = normal

        return nearest_object, min_distance, nearest_object_normal


class Object3D:
    def __init__(self):
        self.ambient = None
        self.diffuse = None
        self.specular = None
        self.shininess = None
        self.reflection = None
        self.refraction = None

    def set_material(self, ambient, diffuse, specular, shininess, reflection, refraction = 0):
        self.ambient = np.array(ambient, dtype=np.float64)
        self.diffuse = np.array(diffuse, dtype=np.float64)
        self.specular = np.array(specular, dtype=np.float64)
        self.shininess = shininess
        self.reflection = reflection
        self.refraction = refraction

    @abstractmethod
    def intersect(self, ray: Ray) -> (Object3D, float, ndarray):
        pass


class Plane(Object3D):
    def __init__(self, normal, point):
        super(Object3D).__init__()
        self.normal = normalize(np.array(normal))
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)

        if t > 0:
            return self, t, self.normal
        else:
            return None, None, None


class Triangle(Object3D):
    r"""
        C
        /\
       /  \
    A /____\ B

    The front face of the triangle is A -> B -> C.

    """

    def __init__(self, a, b, c):
        super().__init__()
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()
        self.area = self.compute_area()

    # Computes normal to the triangle surface. Pay attention to its direction!
    def compute_normal(self):
        ab = self.b - self.a
        ac = self.c - self.a
        return normalize(np.cross(ab, ac))

    def intersect(self, ray: Ray):
        v = self.a - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)

        if t <= 0:
            return None, None, None

        p = ray.origin + t * ray.direction
        pa = self.a - p
        pb = self.b - p
        pc = self.c - p
        alpha = np.dot(self.normal, np.cross(pb, pc)) / (2 * self.area)
        beta = np.dot(self.normal, np.cross(pc, pa)) / (2 * self.area)
        gamma = 1 - alpha - beta

        if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
            return self, t, self.normal

        return None, None, None

    def compute_area(self):
        ab = self.b - self.a
        ac = self.c - self.a
        return np.linalg.norm(np.cross(ab, ac)) / 2


class Diamond(Object3D):
    r"""
                D
                /\*\
               /==\**\
             /======\***\
           /==========\***\
         /==============\****\
       /==================\*****\
    A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
       \==================/****/
         \==============/****/
           \==========/****/
             \======/***/
               \==/**/
                \/*/
                 E

    Similar to Triangle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        C -> A -> D
        E -> B -> A
        E -> C -> B
        E -> A -> C
    """

    def __init__(self, v_list):
        super().__init__()
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        t_idx = [
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
            [4, 1, 0],
            [4, 2, 1],
            [4, 0, 2]]

        return [Triangle(*self.v_list[i]) for i in t_idx]

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        nearest_triangle: Object3D = None
        min_distance = np.inf
        nearest_triangle_normal = None

        for triangle in self.triangle_list:
            _, distance, normal = triangle.intersect(ray)
            if distance is not None and distance < min_distance:
                min_distance = distance
                nearest_triangle = triangle
                nearest_triangle_normal = normal

        return nearest_triangle, min_distance, nearest_triangle_normal


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        super().__init__()
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        oc = ray.origin - self.center
        d = ray.direction
        a = 1
        b = 2 * np.dot(oc, d)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None, None, None

        t1, t2 = np.roots([a, b, c])
        t1 = t1 if t1 > 0 else np.inf
        t2 = t2 if t2 > 0 else np.inf
        t = min(t1, t2)

        if t == np.inf:
            return None, None, None

        hit = ray.origin + t * ray.direction
        normal = normalize(hit - self.center)

        return self, t, normal


def get_color(ray: Ray, ambient, lights: list[LightSource], objects: list[Object3D], level: int):
    color = np.zeros(3, dtype=np.float64)
    obj, t, N = ray.nearest_intersected_object(objects)

    if obj is None:
        return color

    if np.dot(ray.direction, N) > 0:
        N = -N

    hit = ray.origin + t * ray.direction + N * 1e-4
    color = obj.ambient * ambient

    for light in lights:
        shadow_ray = light.get_light_ray(hit)
        shadow_obj, shadow_t, shadow_N = shadow_ray.nearest_intersected_object(objects)
        shadow_factor = 1

        if shadow_obj is not None:
            shadow_hit = shadow_ray.origin + shadow_t * shadow_ray.direction + shadow_N * 1e-4
            light_distance = light.get_distance_from_light(hit)
            shadow_distance = np.linalg.norm(hit - shadow_hit)
            if shadow_distance < light_distance:
                if shadow_obj.refraction == 0:
                    continue
                shadow_factor = shadow_obj.refraction

        intensity = shadow_factor * light.get_intensity(hit)
        L = shadow_ray.direction
        R = reflected(-L, N)

        diffuse = obj.diffuse * intensity * max(0, np.dot(L, N))
        specular = obj.specular * intensity * max(0, np.dot(R, ray.direction) ** obj.shininess)
        color += diffuse + specular

    if level > 1:
        if obj.reflection > 0:
            reflected_ray = Ray(hit, normalize(reflected(ray.direction, N)))
            color += obj.reflection * get_color(reflected_ray, ambient, lights, objects, level - 1)
        if obj.refraction > 0:
            hit_inner = ray.origin + t * ray.direction - N * 1e-4
            refracted_ray = Ray(hit_inner, ray.direction)
            color = (1 - obj.refraction) * color + obj.refraction * get_color(refracted_ray, ambient, lights, objects, level - 1)

    return  color
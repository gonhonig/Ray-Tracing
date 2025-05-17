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
        return Ray(intersection_point, -normalize(self.direction))

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
        return Ray(intersection, normalize(self.position - intersection))

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
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(self.position - intersection)

    def get_intensity(self, intersection):
        v = normalize(intersection - self.position)
        d = self.get_distance_from_light(intersection)
        return (self.intensity * v.dot(self.direction)) / (self.kc + self.kl * d + self.kq * (d ** 2))


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

    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

    @abstractmethod
    def intersect(self, ray: Ray) -> (Object3D, float, ndarray):
        pass


class Plane(Object3D):
    def __init__(self, normal, point):
        super(Object3D).__init__()
        self.normal = np.array(normal)
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

    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """

    def __init__(self, v_list):
        super().__init__()
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
            [0, 1, 3],
            [1, 2, 3],
            [0, 3, 2],
            [4, 1, 0],
            [4, 2, 1],
            [2, 4, 0]]
        # TODO
        return l

    def apply_materials_to_triangles(self):
        # TODO
        pass

    def intersect(self, ray: Ray):
        # TODO
        pass


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        super().__init__()
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        # TODO
        pass


def get_color(ray: Ray, ambient, lights: list[LightSource], objects: list[Object3D], level: int):
    color = np.zeros(3)
    obj, t, N = ray.nearest_intersected_object(objects)

    if obj is None:
        return color

    hit = ray.origin + t * ray.direction
    color = obj.ambient * ambient

    for light in lights:
        shadow_ray = light.get_light_ray(hit + N * 1e-4)
        shadow_obj, shadow_t, _ = shadow_ray.nearest_intersected_object(objects)
        light_distance = light.get_distance_from_light(hit)
        shadow_distance = light.get_distance_from_light(shadow_ray.origin + shadow_t * shadow_ray.direction)

        if shadow_obj is not None and shadow_distance < light_distance:
            continue

        intensity = light.get_intensity(hit)
        L = normalize(shadow_ray.direction)
        R = reflected(-L, N)

        diffuse = obj.diffuse * intensity * max(0, np.dot(L, N))
        specular = obj.specular * intensity * max(0, np.dot(R, normalize(ray.direction)) ** obj.shininess)
        color += diffuse + specular

    return  color
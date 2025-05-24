from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            # This is the main loop where each pixel color is computed.
            color = get_color(ray, ambient, lights, objects, max_depth)
            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])

    floor = Plane([0, 1, 0], [0, -1, 0])
    floor.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)
    background = Plane([0, 0, 1], [0, 0, -3])
    background.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 1000, 0.5)

    v_list = np.array(
        [
            [-0.5, -0.042, -0.998],
            [-0.034, 0.192, -0.145],
            [0.484, 0.131, -0.998],
            [-0.104, 0.951, -0.828],
            [0.23, -0.733, -0.591]
        ])

    diamond = Diamond(v_list)
    diamond.set_material([0.1, 0.4, 0.7], [1, 0, 0], [0.7, 0.3, 0.3], 10, 0.5)
    diamond.apply_materials_to_triangles()

    sphere = Sphere([0, 0, 0], 0.55)
    sphere.set_material([1, 0, 0], [0, 0.3, 0.7], [0.5, 0.5, 0.5], 200, 0, 0.8)

    light_a = PointLight(intensity=np.array([1, 1, 1]), position=np.array([1, 1.5, 1]), kc=0.1, kl=0.1, kq=0.1)
    light_b = SpotLight(intensity=np.array([0, 1, 0]), position=np.array([-0.5, 0.5, 0]), direction=([0, 0, 1]),
                        kc=0.1, kl=0.1, kq=0.1)

    lights = [light_a, light_b]
    objects = [floor, sphere, diamond, background]

    return camera, lights, objects


def render_triangle_mesh(triangles, screen_size, max_depth):
    camera = np.array([0, 0, 1])
    ambient = np.array([0.5, 0.5, 0.5])

    floor = Plane([0, 1, 0], [0, -1, 0])
    floor.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)

    light_a = PointLight(intensity=np.array([1, 1, 1]), position=np.array([1, 1.5, 1]), kc=0.1, kl=0.1, kq=0.1)
    light_b = PointLight(intensity=np.array([0.5, 0.5, 0.5]), position=np.array([-3, -1, 3]), kc=0.1, kl=0.1, kq=0.1)
    lights = [light_a, light_b]

    objects = triangles + [floor]

    return render_scene(camera, ambient, lights, objects, screen_size, max_depth)
import numpy as np
import pyglet
from PIL import Image
from pyglet.gl import *


class App:
    def __init__(self, background):
        self.world = World()
        self.win = pyglet.window.Window()
        self.camera = Camera(self.win)
        self.background = background

    def mainLoop(self):
        while not self.win.has_exit:
            self.win.dispatch_events()
            glClearColor(127, 0, 127, 127)
            self.win.clear()
            self.camera.twoD()
            self.world.draw()
            self.background.draw()


class World:
    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();


class Camera:
    def __init__(self, win):
        self.win = win

    def twoD(self):
        pass
        # glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()
        # gluOrtho2D(0, self.win.width, 0, self.win.height)


class Background:
    def __init__(self, pix):
        self.pix = pix.flatten()
        self.indices = np.indices(env_pixels.shape[:2]).flatten(order='F')

    def draw(self):
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        pyglet.graphics.draw(len(self.pix) / 3, pyglet.gl.GL_POINTS,
                             ('v2i', self.indices),
                             ('c3b', self.pix)
                             )


def temp_test_image_draw(pixels):
    window = pyglet.window.Window(pixels.shape[1], pixels.shape[0])

    pix_reshaped = np.flipud(pixels).flatten()
    x_indices, y_indices = np.indices(pixels.shape[:2])
    x_indices = x_indices.flatten()
    y_indices = y_indices.flatten()
    indices_reshaped = np.empty((x_indices.size + y_indices.size), dtype=x_indices.dtype)
    indices_reshaped[0::2] = y_indices
    indices_reshaped[1::2] = x_indices

    label = pyglet.text.Label('Hello, world',
                              font_name='Times New Roman',
                              font_size=36,
                              x=window.width // 2, y=window.height // 2,
                              anchor_x='center', anchor_y='center')

    vertex_list = pyglet.graphics.vertex_list(len(pix_reshaped) / 3,
                                              ('v2i', indices_reshaped),
                                              ('c3B', pix_reshaped)
                                              )

    @window.event
    def on_draw():
        window.clear()
        vertex_list.draw(pyglet.gl.GL_POINTS)
        label.draw()

    pyglet.app.run()

def draw_patch(src_pixels, target_indices):
    pass

if __name__ == "__main__":
    import sql_model, file_handling
    import sys
    img_name = 'small_butterfly'
    K = 50
    target_image = sql_model.get_target_image(img_name, K)
    temp_test_image_draw(target_image.pixels)
    for patch in target_image.segments:
        labelled_indices = np.where(target_image.labels == patch.id)
        patch_indices = (labelled_indices[0] - np.min(labelled_indices[0]),
                         labelled_indices[1] - np.min(labelled_indices[1]))

        best_visit_loss = sys.float_info.max
        best_visit = None
        for visit in patch.visits:
            if visit.loss < best_visit_loss:
                best_visit_loss = visit.loss
                best_visit = visit
        translated_x = patch_indices[0] + best_visit.x
        translated_y = patch_indices[1] + best_visit.y
        src_pix = file_handling.get_source_image(best_visit.source)

        best_patch_pix = src_pix[translated_x, translated_y]
        print best_patch_pix.shape
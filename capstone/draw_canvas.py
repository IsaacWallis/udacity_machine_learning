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


if __name__ == "__main__":
    import sql_model, file_handling
    import sys
    import vbo

    img_name = 'small_butterfly'
    K = 50
    target_image = sql_model.get_target_image(img_name, K)
    canvas_image = np.copy(target_image.pixels)
    vbo.setup_window(target_image.pixels.shape)
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
        canvas_image[labelled_indices] = best_patch_pix
    vbo.draw_target_image(canvas_image)

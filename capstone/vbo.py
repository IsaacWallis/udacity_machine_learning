from pyglet.gl import *
import numpy as np


class VBO():
    def __init__(self):
        self.buffer = (GLuint)(0)
        glGenBuffers(1, self.buffer)

    def data(self, data):
        data_gl = (GLfloat * len(data))(*data)
        glBindBuffer(GL_ARRAY_BUFFER_ARB, self.buffer)
        glBufferData(GL_ARRAY_BUFFER_ARB, len(data) * 4,
                     data_gl, GL_STATIC_DRAW)

    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER_ARB, self.buffer)

    def vertex(self):
        self.bind()
        glVertexPointer(2, GL_FLOAT, 0, 0)

    def color(self):
        self.bind()
        glColorPointer(3, GL_FLOAT, 0, 0)


window = None
vbo = VBO()
color = VBO()


def setup_window(shape):
    global window
    window = pyglet.window.Window(shape[1], shape[0])

    @window.event
    def on_draw():
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        vbo.vertex()
        color.color()

        glDrawArrays(GL_POINTS, 0, shape[0] * shape[1] * 2)


def draw_target_image(pixels):
    pix_reshaped = np.flipud(pixels).flatten() / 256.

    x_indices, y_indices = np.indices(pixels.shape[:2])
    x_indices = x_indices.flatten()
    y_indices = y_indices.flatten()
    indices_reshaped = np.empty((x_indices.size + y_indices.size), dtype=x_indices.dtype)
    indices_reshaped[0::2] = y_indices
    indices_reshaped[1::2] = x_indices
    indices_reshaped = indices_reshaped.astype(np.float32)

    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_VERTEX_ARRAY)

    vbo.data(indices_reshaped)
    color.data(pix_reshaped)

    pyglet.app.run()


    # import sql_model
    # img_name = 'small_butterfly'
    # K = 50
    # target_image = sql_model.get_target_image(img_name, K)
    # draw_target_image(target_image.pixels)

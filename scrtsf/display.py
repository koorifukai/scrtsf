"""Movable, resizable optical display windows."""

import cv2

from . import visual


class OpticalWindow:
    def __init__(self, title, screen):
        self.title = title
        self.size = (min(960, screen[0]), min(720, screen[1]))
        self.shown = False
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, *self.size)

    def closed(self):
        return self.shown and cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1

    def show(self, profile, packet, label):
        previous = self.size
        if self.shown:
            _, _, width, height = cv2.getWindowImageRect(self.title)
            if width >= 320 and height >= 240:
                self.size = (width, height)
        cv2.imshow(self.title, visual.render(packet, profile, self.size, label))
        self.shown = True
        return self.size != previous

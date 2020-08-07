import math
import cv2


class Circle:
    def __init__(self, coord):
        """
        Instantiate circle from bounding box.
        Uses half of box diagonal as radius. Result circle covers whole obj.
        :param coord: [y1, y2, x1, x2]
        """
        self.p_y = math.ceil((coord[0] + coord[1]) / 2)
        self.p_x = math.ceil((coord[2] + coord[3]) / 2)
        self.r = math.ceil(math.sqrt(((self.p_x - coord[2]) ** 2) + ((self.p_y - coord[0]) ** 2)))


def visualize_frame(img, state_dict, window_name='frame'):
    """
    Visualize parsed frame state: circle the agent and box the other objects.
    :param img: *BGR* img (same format as read by cv2).
    :param state_dict: state dict: {'agent': [y1, y2, x1, x2],
                                    '<other modules>': [[y1, y2, x1, x2] for each obj]}
    :param window_name: name of window to be displayed.
    :return: None
    """
    state = state_dict.copy()
    box_img = img.copy()

    # circle agent
    if None not in state['agent']:
        circ = Circle(state['agent'])
        cv2.circle(box_img, (math.ceil(circ.p_x), math.ceil(circ.p_y)), math.ceil(circ.r), (0, 0, 255))
    state.pop('agent')

    # box objects
    for mod in state:
        if state[mod] is not None:
            for obj in state[mod]:
                if None not in obj:
                    cv2.rectangle(box_img, (int(obj[2]) - 1, int(obj[0]) - 1), (int(obj[3]) + 1, int(obj[1]) + 1),
                                  (0, 0, 255), 1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 840, 640)
    cv2.imshow(window_name, box_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

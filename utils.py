import logging
import base64
from PIL import Image
from io import BytesIO


class CustomLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        logging.basicConfig(
            format='%(asctime)s || %(name)s || %(levelname)s || %(message)s',
            level=logging.INFO,
            # filename="test_log.log",
            # filemode='a',
            handlers=[
                logging.FileHandler(self.log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)


########################### Concatenate images ###########################


def load_image(base64_image):
    img_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(img_data))
    return image


def parse_resolution(resolution_str):
    # try to parse a string into a resolution tuple for the grid output
    try:
        width, height = map(int, resolution_str.split(','))
        return width, height
    except Exception as e:
        raise argparse.ArgumentTypeError("Resolution must be w,h.") from e


def concatenate_images_vertical(images, dist_images):
    # calc max width from imgs
    width = max(img.width for img in images)
    # calc total height of imgs + dist between them
    total_height = sum(img.height for img in images) + dist_images * (len(images) - 1)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (width, total_height), (0, 0, 0))

    # init var to track current height pos
    current_height = 0
    for img in images:
        # paste img in new_img at current height
        new_img.paste(img, (0, current_height))
        # update current height for next img
        current_height += img.height + dist_images

    return new_img


def concatenate_images_horizontal(images, dist_images):
    # calc total width of imgs + dist between them
    total_width = sum(img.width for img in images) + dist_images * (len(images) - 1)
    # calc max height from imgs
    height = max(img.height for img in images)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    # init var to track current width pos
    current_width = 0
    for img in images:
        # paste img in new_img at current width
        new_img.paste(img, (current_width, 0))
        # update current width for next img
        current_width += img.width + dist_images

    return new_img


def concatenate_images_grid(images, dist_images, output_size):
    num_images = len(images)
    # calc grid size based on amount of input imgs
    grid_size = max(2, ceil(sqrt(num_images)))

    cell_width = (output_size[0] - dist_images * (grid_size - 1)) // grid_size
    cell_height = (output_size[1] - dist_images * (grid_size - 1)) // grid_size

    # create new img with output_size, black bg
    new_img = Image.new('RGB', output_size, (0, 0, 0))

    for index, img in enumerate(images):
        # calc img aspect ratio
        img_ratio = img.width / img.height
        # calc target aspect ratio per cell
        target_ratio = cell_width / cell_height

        # resize img to fit in cell
        if img_ratio > target_ratio:
            new_width = cell_width
            new_height = int(cell_width / img_ratio)
        else:
            new_width = int(cell_height * img_ratio)
            new_height = cell_height

        # resize img using lanczos filter
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        row = index // grid_size
        col = index % grid_size

        # calc x, y offsets for img positioning
        x_offset = col * (cell_width + dist_images) + (cell_width - new_width) // 2
        y_offset = row * (cell_height + dist_images) + (cell_height - new_height) // 2

        # paste resized img in calc pos
        new_img.paste(resized_img, (x_offset, y_offset))

    return new_img


# https://github.com/haotian-liu/LLaVA/issues/874
# https://github.com/mapluisch/LLaVA-CLI-with-multiple-images
def concatenate_images(images, strategy, dist_images, grid_resolution):
    if strategy == 'vertical':
        return concatenate_images_vertical(images, dist_images)
    elif strategy == 'horizontal':
        return concatenate_images_horizontal(images, dist_images)
    elif strategy == 'grid':
        return concatenate_images_grid(images, dist_images, grid_resolution)
    else:
        raise ValueError("Invalid concatenation strategy specified")

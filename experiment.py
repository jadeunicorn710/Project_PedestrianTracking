"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import ps5
import os
import numpy as np

# I/O directories
input_dir = "input_images"
output_dir = "./"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


# Helper code
def run_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))



def run_particle_filter5(filter_class, imgs_dir, template_rect1, template_rect2, template_rect3,
                         save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect1 (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # initialize the frame number
    frame_num = 0

    # Initialize templates for all targets
    template1 = None
    pf1 = None
    template2 = None
    pf2 = None
    template3 = None
    pf3 = None


    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # object 1 appears prior to frame 59
        if frame_num <= 59:
            # Extract template and initialize (one-time only)
            if template1 is None:
                template1 = frame[int(template_rect1['y']):
                                 int(template_rect1['y'] + template_rect1['h']),
                           int(template_rect1['x']):
                                 int(template_rect1['x'] + template_rect1['w'])]

                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template1)

                pf1 = filter_class(frame, template1, **kwargs)

            # Process frame
            pf1.process(frame)

        # object 2 appears prior to frame 45
        if frame_num <= 45:
            # Extract template and initialize (one-time only)
            if template2 is None:
                template2 = frame[int(template_rect2['y']):
                                  int(template_rect2['y'] + template_rect2['h']),
                            int(template_rect2['x']):
                            int(template_rect2['x'] + template_rect2['w'])]

                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template2)

                pf2 = filter_class(frame, template2, **kwargs)

            # Process frame
            pf2.process(frame)

        # object 3 appears after frame 28
        if frame_num >= 28:
            # Extract template and initialize (one-time only)
            if template3 is None:
                template3 = frame[int(template_rect3['y']):
                                  int(template_rect3['y'] + template_rect3['h']),
                            int(template_rect3['x']):
                            int(template_rect3['x'] + template_rect3['w'])]

                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template3)

                pf3 = filter_class(frame, template3, **kwargs)

            # Process frame
            pf3.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            if frame_num <= 59:
                pf1.render(out_frame)
            if frame_num <= 45:
                pf2.render(out_frame)
            if frame_num >= 28:
                pf3.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            if frame_num <= 59:
                pf1.render(frame_out)
            if frame_num <= 45:
                pf2.render(frame_out)
            if frame_num >= 28:
                pf3.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))



def run_particle_filter6(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))
        # target is in frame without occlusion prior to frame number 116 and after number frame 127
        if frame_num <= 117 or frame_num >= 128:
            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):
                                 int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):
                                 int(template_rect['x'] + template_rect['w'])]

                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)

                pf = filter_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))


def run_kalman_filter(kf, imgs_dir, noise, sensor, save_frames={},
                      template_loc=None):

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))


def run_kalman_filter5(kf1, kf2, kf3, imgs_dir, noise, sensor, template_loc1, template_loc2, template_loc3,
                       save_frames={}):

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # initialize the frame number
    frame_num = 0

    # initialize templates for all targets
    template1 = None
    template2 = None
    template3 = None


    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        pass

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # object 1 appears prior to frame 59
        if frame_num <= 59:
            # Extract template and initialize (one-time only)
            if template1 is None:
                template1 = frame[template_loc1['y']:
                                 template_loc1['y'] + template_loc1['h'],
                                 template_loc1['x']:
                                 template_loc1['x'] + template_loc1['w']]

            # Sensor
            if sensor == "hog":
                (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                        padding=(8, 8), scale=1.05)

                if len(weights) > 0:
                    max_w_id = np.argmax(weights)
                    z_x, z_y, z_w, z_h = rects[max_w_id]

                    z_x += z_w // 2
                    z_y += z_h // 2

                    z_x += np.random.normal(0, noise['x'])
                    z_y += np.random.normal(0, noise['y'])

                    z_x1 = z_x
                    z_y1 = z_y

                    z_w1 = z_w
                    z_h1 = z_h


            elif sensor == "matching":
                corr_map1 = cv2.matchTemplate(frame, template1, cv2.TM_SQDIFF)
                z_y1, z_x1 = np.unravel_index(np.argmin(corr_map1), corr_map1.shape)

                z_w1 = template_loc1['w']
                z_h1 = template_loc1['h']

                z_x1 += z_w1 // 2 + np.random.normal(0, noise['x'])
                z_y1 += z_h1 // 2 + np.random.normal(0, noise['y'])

            x1, y1 = kf1.process(z_x1, z_y1)

        # object 2 appears prior to frame 45
        if frame_num <= 45:
            # Extract template and initialize (one-time only)
            if template2 is None:
                template2 = frame[template_loc2['y']:
                                 template_loc2['y'] + template_loc2['h'],
                                 template_loc2['x']:
                                 template_loc2['x'] + template_loc2['w']]

            # Sensor
            if sensor == "hog":
                (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                        padding=(8, 8), scale=1.05)

                if len(weights) > 0:
                    max_w_id = np.argmax(weights)
                    z_x, z_y, z_w, z_h = rects[max_w_id]

                    z_x += z_w // 2
                    z_y += z_h // 2

                    z_x += np.random.normal(0, noise['x'])
                    z_y += np.random.normal(0, noise['y'])

                    z_x2 = z_x
                    z_y2 = z_y

                    z_w2 = z_w
                    z_h2 = z_h


            elif sensor == "matching":
                corr_map2 = cv2.matchTemplate(frame, template2, cv2.TM_SQDIFF)
                z_y2, z_x2 = np.unravel_index(np.argmin(corr_map2), corr_map2.shape)

                z_w2 = template_loc2['w']
                z_h2 = template_loc2['h']

                z_x2 += z_w2 // 2 + np.random.normal(0, noise['x'])
                z_y2 += z_h2 // 2 + np.random.normal(0, noise['y'])

            x2, y2 = kf2.process(z_x2, z_y2)


        # object 3 appears after frame 28
        if frame_num >= 28:
            # Extract template and initialize (one-time only)
            if template3 is None:
                template3 = frame[template_loc3['y']:
                                 template_loc3['y'] + template_loc3['h'],
                                 template_loc3['x']:
                                 template_loc3['x'] + template_loc3['w']]

            # Sensor
            if sensor == "hog":
                (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                        padding=(8, 8), scale=1.05)

                if len(weights) > 0:
                    max_w_id = np.argmax(weights)
                    z_x, z_y, z_w, z_h = rects[max_w_id]

                    z_x += z_w // 2
                    z_y += z_h // 2

                    z_x += np.random.normal(0, noise['x'])
                    z_y += np.random.normal(0, noise['y'])

                    z_x3 = z_x
                    z_y3 = z_y

                    z_w3 = z_w
                    z_h3 = z_h


            elif sensor == "matching":
                corr_map3 = cv2.matchTemplate(frame, template3, cv2.TM_SQDIFF)
                z_y3, z_x3 = np.unravel_index(np.argmin(corr_map3), corr_map3.shape)

                z_w3 = template_loc3['w']
                z_h3 = template_loc3['h']

                z_x3 += z_w3 // 2 + np.random.normal(0, noise['x'])
                z_y3 += z_h3 // 2 + np.random.normal(0, noise['y'])

            x3, y3 = kf3.process(z_x3, z_y3)


        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            # visualize target 1
            if frame_num <= 59:
                cv2.circle(out_frame, (int(z_x1), int(z_y1)), 20, (0, 0, 255), 2)
                cv2.circle(out_frame, (int(x1), int(y1)), 10, (255, 0, 0), 2)
                cv2.rectangle(out_frame, (int(z_x1) - z_w1 // 2, int(z_y1) - z_h1 // 2),
                              (int(z_x1) + z_w1 // 2, int(z_y1) + z_h1 // 2),
                              (0, 0, 255), 2)

            # visualize target 2
            if frame_num <= 45:
                cv2.circle(out_frame, (int(z_x2), int(z_y2)), 20, (0, 0, 255), 2)
                cv2.circle(out_frame, (int(x2), int(y2)), 10, (255, 0, 0), 2)
                cv2.rectangle(out_frame, (int(z_x2) - z_w2 // 2, int(z_y2) - z_h2 // 2),
                              (int(z_x2) + z_w2 // 2, int(z_y2) + z_h2 // 2),
                              (0, 0, 255), 2)

            # visualize target 3
            if frame_num >= 28:
                cv2.circle(out_frame, (int(z_x3), int(z_y3)), 20, (0, 0, 255), 2)
                cv2.circle(out_frame, (int(x3), int(y3)), 10, (255, 0, 0), 2)
                cv2.rectangle(out_frame, (int(z_x3) - z_w3 // 2, int(z_y3) - z_h3 // 2),
                              (int(z_x3) + z_w3 // 2, int(z_y3) + z_h3 // 2),
                              (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            if frame_num <= 59:
                cv2.circle(frame_out, (int(x1), int(y1)), 10, (255, 0, 0), 2)
            if frame_num <= 45:
                cv2.circle(frame_out, (int(x2), int(y2)), 10, (255, 0, 0), 2)
            if frame_num >= 28:
                cv2.circle(frame_out, (int(x3), int(y3)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = 0.2 * np.eye(4)  # Process noise array
    R = 0.4 * np.eye(4)  # Measurement noise array

    kf = ps5.KalmanFilter(template_loc['x'], template_loc['y'])

    save_frames = {10: os.path.join(output_dir, 'ps5-1-b-1.png'),
                   30: os.path.join(output_dir, 'ps5-1-b-2.png'),
                   59: os.path.join(output_dir, 'ps5-1-b-3.png'),
                   99: os.path.join(output_dir, 'ps5-1-b-4.png')}

    run_kalman_filter(kf,
                      os.path.join(input_dir, "circle"),
                      NOISE_2,
                      "matching",
                      save_frames,
                      template_loc)


def part_1c():
    print("Part 1c")

    init_pos = {'x': 311, 'y': 217}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = 0.01 * np.eye(4)  # Process noise array
    R = 0.04 * np.eye(4)  # Measurement noise array

    kf = ps5.KalmanFilter(init_pos['x'], init_pos['y'])

    save_frames = {10: os.path.join(output_dir, 'ps5-1-c-1.png'),
                   33: os.path.join(output_dir, 'ps5-1-c-2.png'),
                   84: os.path.join(output_dir, 'ps5-1-c-3.png'),
                   159: os.path.join(output_dir, 'ps5-1-c-4.png')}

    run_kalman_filter(kf,
                      os.path.join(input_dir, "walking"),
                      NOISE_1,
                      "hog",
                      save_frames)


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {10: os.path.join(output_dir, 'ps5-2-a-1.png'),
                   30: os.path.join(output_dir, 'ps5-2-a-2.png'),
                   59: os.path.join(output_dir, 'ps5-2-a-3.png'),
                   99: os.path.join(output_dir, 'ps5-2-a-4.png')}

    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(ps5.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "circle"),
                        template_loc,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_loc)  # Add more if you need to


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {10: os.path.join(output_dir, 'ps5-2-b-1.png'),
                   33: os.path.join(output_dir, 'ps5-2-b-2.png'),
                   84: os.path.join(output_dir, 'ps5-2-b-3.png'),
                   99: os.path.join(output_dir, 'ps5-2-b-4.png')}

    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(ps5.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate_noisy"),
                        template_loc,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_loc)  # Add more if you need to


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {22: os.path.join(output_dir, 'ps5-3-a-1.png'),
                   50: os.path.join(output_dir, 'ps5-3-a-2.png'),
                   160: os.path.join(output_dir, 'ps5-3-a-3.png')}

    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.2  # Set a value for alpha

    run_particle_filter(ps5.AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "pres_debate"),
                        # input video
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect)  # Add more if you need to


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {40: os.path.join(output_dir, 'ps5-4-a-1.png'),
                   100: os.path.join(output_dir, 'ps5-4-a-2.png'),
                   240: os.path.join(output_dir, 'ps5-4-a-3.png'),
                   300: os.path.join(output_dir, 'ps5-4-a-4.png')}

    num_particles = 500  # Define the number of particles
    sigma_md = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(ps5.MDParticleFilter,
                        os.path.join(input_dir, "pedestrians"),
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_md,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    #================================================================================
    # 1. Kalman filter
    #================================================================================

    # define initial position of templates for each targets when they are in frame
    # # full body
    # template_rect1 = {'x': 96, 'y': 280, 'w': 98, 'h': 330}
    # template_rect2 = {'x': 310, 'y': 290, 'w': 68, 'h': 200}
    # template_rect3 = {'x': 28, 'y': 290, 'w': 56, 'h': 240}

    # # upper body
    # template_rect1 = {'x': 100, 'y': 220, 'w': 90, 'h': 160}
    # template_rect2 = {'x': 310, 'y': 246, 'w': 68, 'h': 100}
    # template_rect3 = {'x': 28, 'y': 240, 'w': 56, 'h': 140}

    # trial and error
    template_rect1 = {'x': 78, 'y': 148, 'w': 78, 'h': 148}
    template_rect2 = {'x': 302, 'y': 258, 'w': 32, 'h': 66}
    template_rect3 = {'x': 4, 'y': 204, 'w': 42, 'h': 96}


    # # visualize bounding boxes at targets 1 & 2 initial state
    # target1_2 = cv2.imread('input_images/TUD-Campus/000001.jpg')
    # # print(target1.shape)
    #
    # # visualize bounding box at target 3 initial state
    # target3 = cv2.imread('input_images/TUD-Campus/000028.jpg')
    #
    # cv2.rectangle(target1_2, (int(template_rect1['x'] - template_rect1['w'] / 2), int(template_rect1['y'] - template_rect1['h'] / 2)),
    #               (int(template_rect1['x'] + template_rect1['w'] / 2), int(template_rect1['y'] + template_rect1['h'] / 2)),
    #               (224, 255, 255), 2)
    # cv2.rectangle(target1_2, (int(template_rect2['x'] - template_rect2['w'] / 2), int(template_rect2['y'] - template_rect2['h'] / 2)),
    #               (int(template_rect2['x'] + template_rect2['w'] / 2), int(template_rect2['y'] + template_rect2['h'] / 2)),
    #               (224, 255, 255), 2)
    # cv2.rectangle(target3, (int(template_rect3['x'] - template_rect3['w'] / 2), int(template_rect3['y'] - template_rect3['h'] / 2)),
    #               (int(template_rect3['x'] + template_rect3['w'] / 2), int(template_rect3['y'] + template_rect3['h'] / 2)),
    #               (224, 255, 255), 2)
    #
    # cv2.imwrite('target1 & 2_bounding_box.png', target1_2)
    # cv2.imwrite('target3_bounding_box.png', target3)


    # print("Part 5: Kalman filter option:")
    #
    # # Define process and measurement arrays if you want to use other than the
    # # default. Pass them to KalmanFilter.
    # Q = 0.2 * np.eye(4)  # Process noise array
    # R = 0.4 * np.eye(4)  # Measurement noise array
    #
    # # define filter class for each target
    # kf1 = ps5.KalmanFilter(template_rect1['x'], template_rect1['y'])
    # kf2 = ps5.KalmanFilter(template_rect2['x'], template_rect2['y'])
    # kf3 = ps5.KalmanFilter(template_rect3['x'], template_rect3['y'])
    # # kf = ps5.KalmanFilter
    #
    # save_frames = {29: os.path.join(output_dir, 'ps5-5-b-1.png'),
    #                56: os.path.join(output_dir, 'ps5-5-b-2.png'),
    #                70: os.path.join(output_dir, 'ps5-5-b-3.png')}
    #
    # # Kalman filter with matching sensor
    # run_kalman_filter5(kf1, kf2, kf3,
    #                   os.path.join(input_dir, "TUD-Campus"),
    #                   NOISE_2,
    #                   "matching",
    #                    template_rect1, template_rect2, template_rect3,
    #                   save_frames)


    # # Kalman filter with HoG sensor (Do NOT use this one)
    # run_kalman_filter5(kf1, kf2, kf3,
    #                   os.path.join(input_dir, "TUD-Campus"),
    #                   NOISE_1,
    #                   "hog",
    #                    template_rect1, template_rect2, template_rect3,
    #                   save_frames)


    #================================================================================
    # 2. Particle filter
    #================================================================================

    print("Part 5: Appearance Model Particle filter option:")

    save_frames = {29: os.path.join(output_dir, 'ps5-5-a-1.png'),
                   56: os.path.join(output_dir, 'ps5-5-a-2.png'),
                   70: os.path.join(output_dir, 'ps5-5-a-3.png')}

    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.03  # Set a value for alpha

    run_particle_filter5(ps5.AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "TUD-Campus"),
                        # input video
                        template_rect1, template_rect2, template_rect3,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect1)  # Add more if you need to

    # raise NotImplementedError


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    print("Part 6: Appearance Model Particle filter:")

    # define initial position of templates for the target
    # # full body
    # template_rect = {'x': 110, 'y': 116, 'w': 40, 'h': 170}

    # # upper body
    # template_rect = {'x': 110, 'y': 78, 'w': 40, 'h': 96}

    # trial and error
    template_rect = {'x': 96, 'y': 40, 'w': 32, 'h': 78}

    # visualize bounding boxes at targets 1 & 2 initial state
    target = cv2.imread('input_images/follow/000.jpg')

    cv2.rectangle(target, (int(template_rect['x'] - template_rect['w'] / 2), int(template_rect['y'] - template_rect['h'] / 2)),
                  (int(template_rect['x'] + template_rect['w'] / 2), int(template_rect['y'] + template_rect['h'] / 2)),
                  (224, 255, 255), 2)

    cv2.imwrite('target_bounding_box.png', target)

    save_frames = {60: os.path.join(output_dir, 'ps5-6-a-1.png'),
                   160: os.path.join(output_dir, 'ps5-6-a-2.png'),
                   186: os.path.join(output_dir, 'ps5-6-a-3.png')}

    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.02  # Set a value for alpha

    run_particle_filter6(ps5.AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "follow"),
                        # input video
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect)  # Add more if you need to

    # raise NotImplementedError

if __name__ == '__main__':
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    # part_4()
    # part_5()
    part_6()
